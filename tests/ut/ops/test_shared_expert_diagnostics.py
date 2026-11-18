import ast
import logging
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def load_class(path, name, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = []
    if methods is not None:
        node.body = [item for item in node.body if getattr(item, "name", None) in methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


class TestSharedExpertDiagnostics(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.logger.isEnabledFor.return_value = True
        self.main = SimpleNamespace(device="npu:2", npu_stream=10)
        self.compute = SimpleNamespace(device="npu:2", npu_stream=20)
        self.communication = SimpleNamespace(device="npu:2", npu_stream=30)
        self.current = self.main
        self.context = SimpleNamespace(cudagraph_runtime_mode="NONE", in_profile_run=False, is_draft_model=False)
        self.torch = SimpleNamespace(
            compiler=SimpleNamespace(is_compiling=Mock(return_value=False)),
            npu=SimpleNamespace(
                current_stream=Mock(side_effect=lambda: self.current),
                is_current_stream_capturing=Mock(return_value=False),
            ),
        )
        self.namespace = dict(
            torch=self.torch,
            logging=logging,
            logger=self.logger,
            get_forward_context=lambda: self.context,
            get_ascend_config=lambda: SimpleNamespace(multistream_overlap_shared_expert=True),
        )
        self.cls = load_class(
            "vllm_ascend/ops/fused_moe/shared_expert_diagnostics.py", "SharedExpertDiagnostics", self.namespace
        )
        self.config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
            model_config=SimpleNamespace(runner_type="generate"),
        )

    def messages(self):
        return [call.args[0] % call.args[1:] for call in self.logger.info_once.call_args_list]

    def test_info_logging_reports_each_decision_once_without_debug(self):
        self.logger.isEnabledFor.side_effect = lambda level: level >= logging.INFO
        diagnostic = self.cls("layer.1", self.config)
        self.assertTrue(diagnostic.enabled)
        self.logger.isEnabledFor.assert_called_once_with(logging.INFO)
        diagnostic.record("eligibility", schedule="generic", rejected="mc2")
        diagnostic.record("eligibility", schedule="generic", rejected="mc2")
        self.logger.info_once.assert_called_once()
        self.logger.debug.assert_not_called()
        self.assertIn("stage=eligibility", self.messages()[0])
        self.assertIn("schedule=generic rejected=mc2", self.messages()[0])

    def test_disabled_logging_never_queries_runtime(self):
        self.logger.isEnabledFor.return_value = False
        diagnostic = self.cls("layer.1", self.config)
        diagnostic.record("ag")
        self.torch.compiler.is_compiling.assert_not_called()
        self.torch.npu.current_stream.assert_not_called()
        self.torch.npu.is_current_stream_capturing.assert_not_called()
        self.logger.info_once.assert_not_called()

    def test_prefill_draft_and_combined_roles_are_excluded(self):
        for role in ["kv_producer", "kv_both"]:
            with self.subTest(role=role):
                self.config.kv_transfer_config.kv_role = role
                self.assertFalse(self.cls("layer.1", self.config).enabled)
        self.config.kv_transfer_config = None
        self.assertFalse(self.cls("layer.1", self.config).enabled)
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
        self.config.model_config.runner_type = "draft"
        self.assertFalse(self.cls("layer.1", self.config).enabled)

    def test_decode_bench_and_legacy_consumer_flags(self):
        for transfer in [
            SimpleNamespace(kv_connector="DecodeBenchConnector"),
            SimpleNamespace(is_kv_consumer=True, is_kv_producer=False),
        ]:
            self.config.kv_transfer_config = transfer
            self.assertTrue(self.cls("layer.1", self.config).enabled)

    def test_compiler_and_draft_contexts_do_not_log(self):
        diagnostic = self.cls("layer.1", self.config)
        self.torch.compiler.is_compiling.return_value = True
        diagnostic.record("ag")
        self.torch.npu.current_stream.assert_not_called()
        self.torch.compiler.is_compiling.return_value = False
        self.context.is_draft_model = True
        diagnostic.record("ag")
        self.logger.info_once.assert_not_called()

    def test_reports_actual_stream_and_capture_separately(self):
        diagnostic = self.cls("layer.1", self.config)
        diagnostic.record("ag", tokens=4, ready_event=True)
        diagnostic.record("ag", tokens=8, ready_event=True)
        self.assertEqual(self.logger.info_once.call_count, 1)
        self.current = self.communication
        self.torch.npu.is_current_stream_capturing.return_value = True
        diagnostic.record("ag", tokens=4, ready_event=True)
        self.assertEqual(self.logger.info_once.call_count, 2)
        self.assertIn("stream=10 capturing=False", self.messages()[0])
        self.assertIn("stream=30 capturing=True", self.messages()[1])

    def test_profile_warmup_and_real_execution_are_distinct(self):
        diagnostic = self.cls("layer.1", self.config)
        self.context.in_profile_run = True
        diagnostic.record("selection", schedule="generic", overlap=True)
        self.context.in_profile_run = False
        diagnostic.record("selection", schedule="generic", overlap=True)
        self.assertEqual(self.logger.info_once.call_count, 2)

    def test_changed_fallback_and_schedule_are_not_suppressed(self):
        diagnostic = self.cls("layer.1", self.config)
        diagnostic.record("eligibility", schedule="generic", rejected="shared_bf16_or_mxfp4")
        diagnostic.record("eligibility", schedule="generic", rejected="mc2")
        diagnostic.record("eligibility", schedule="kimi_decode", rejected="none")
        self.assertEqual(self.logger.info_once.call_count, 3)
        self.assertIn("rejected=shared_bf16_or_mxfp4", self.messages()[0])

    def test_ag_logs_inside_communication_stream(self):
        diagnostic = self.cls("layer.1", self.config)
        tensor = SimpleNamespace(shape=(8, 16), record_stream=Mock())
        self.main.record_event = Mock(return_value="input_ready")
        self.communication.record_event = Mock(return_value="ag_done")
        self.communication.wait_event = Mock()

        @contextmanager
        def switch(stream, enabled=True):
            previous = self.current
            self.current = stream if enabled else previous
            try:
                yield
            finally:
                self.current = previous

        modes = SimpleNamespace(SEQUENCE_PARALLEL_ONLY=object())
        namespace = dict(
            self.namespace,
            SharedExpertParallelMode=modes,
            PreparedSharedExpertInput=SimpleNamespace,
            shared_experts_calculation_stream=lambda: self.compute,
            npu_stream_switch=switch,
            tensor_model_parallel_all_gather=Mock(return_value=tensor),
        )
        shared_cls = load_class(
            "vllm_ascend/ops/fused_moe/shared_experts.py", "AscendSharedExperts", namespace, {"prepare_input_async"}
        )
        shared = shared_cls()
        shared._diagnostics = diagnostic
        shared.multistream_overlap = True
        shared.parallel_mode = lambda: modes.SEQUENCE_PARALLEL_ONLY
        result = shared.prepare_input_async(tensor, comm_stream=self.communication)
        self.assertEqual(result.ready_event, "ag_done")
        self.communication.wait_event.assert_called_once_with("input_ready")
        self.assertIn("stage=ag device=npu:2 stream=30", self.messages()[0])
        self.assertIn("dedicated_comm_stream=True", self.messages()[0])

    def test_rs_logs_current_stream_and_preserves_padding(self):
        diagnostic = self.cls("layer.1", self.config)
        self.current = self.communication
        reduced = object()
        pad = Mock(return_value=SimpleNamespace(shape=(8, 16)))
        reduce_scatter = Mock(return_value=reduced)
        namespace = dict(
            self.namespace, F=SimpleNamespace(pad=pad), tensor_model_parallel_reduce_scatter=reduce_scatter
        )
        shared_cls = load_class(
            "vllm_ascend/ops/fused_moe/shared_experts.py", "AscendSharedExperts", namespace, {"_pad_and_reduce_scatter"}
        )
        shared = shared_cls()
        shared._diagnostics = diagnostic
        shared.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=8))
        self.assertIs(shared._pad_and_reduce_scatter(SimpleNamespace(shape=(5, 16))), reduced)
        self.assertIn("stage=rs device=npu:2 stream=30", self.messages()[0])
        self.assertIn("tokens=5 padded_tokens=8", self.messages()[0])
        pad.assert_called_once()
        reduce_scatter.assert_called_once_with(pad.return_value, dim=0)

    def test_fallback_ag_reports_calling_stream(self):
        tensor = SimpleNamespace(shape=(8, 16))
        modes = SimpleNamespace(SEQUENCE_PARALLEL_ONLY=object())
        namespace = dict(
            self.namespace,
            SharedExpertParallelMode=modes,
            PreparedSharedExpertInput=SimpleNamespace,
            tensor_model_parallel_all_gather=Mock(return_value=tensor),
        )
        shared_type = load_class(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            "AscendSharedExperts",
            namespace,
            {"prepare_input_before_routed"},
        )
        shared = shared_type()
        shared._diagnostics = self.cls("layer.1", self.config)
        shared.multistream_overlap = True
        shared.parallel_mode = lambda: modes.SEQUENCE_PARALLEL_ONLY
        self.assertTrue(shared.prepare_input_before_routed(tensor).is_gathered)
        self.assertIn("stage=ag device=npu:2 stream=10", self.messages()[0])
        self.assertIn("input_path=before_routed dedicated_comm_stream=False", self.messages()[0])

    def test_missing_event_still_fails_without_logging(self):
        diagnostic = self.cls("layer.1", self.config)
        shared_cls = load_class(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            "AscendSharedExperts",
            self.namespace,
            {"_wait_for_milestone"},
        )
        shared = shared_cls()
        shared._diagnostics = diagnostic
        shared.multistream_overlap = True
        with self.assertRaisesRegex(RuntimeError, "Missing router_output_ready"):
            shared._wait_for_milestone(None, "router_output_ready")
        self.logger.info_once.assert_not_called()

    def test_decode_down_and_rs_report_their_respective_streams(self):
        diagnostic = self.cls("layer.1", self.config)
        tensor = SimpleNamespace(record_stream=Mock())
        for stream, event in [(self.main, "gmm1_done"), (self.compute, "down_done"), (self.communication, "rs_done")]:
            stream.record_event = Mock(return_value=event)
            stream.wait_event = Mock()

        @contextmanager
        def switch(stream):
            previous = self.current
            self.current = stream
            try:
                yield
            finally:
                self.current = previous

        namespace = dict(
            self.namespace,
            shared_experts_calculation_stream=lambda: self.compute,
            npu_stream_switch=switch,
            moe_superkernel_scope=lambda *args: nullcontext(),
        )
        schedule_type = load_class(
            "vllm_ascend/ops/fused_moe/kimi_decode_schedule.py", "KimiDecodeSchedule", namespace, {"after_gmm1"}
        )
        schedule = schedule_type()
        schedule.shared = SimpleNamespace(
            part2=Mock(return_value=tensor), _pad_and_reduce_scatter=Mock(return_value=tensor)
        )
        schedule.diagnostics = diagnostic
        schedule.enable_superkernel = False
        schedule.comm_stream = self.communication
        schedule.down_input = object()
        schedule.submitted = SimpleNamespace(hidden_states=object())
        schedule.after_gmm1()
        self.assertIn("stage=down device=npu:2 stream=20", self.messages()[0])
        self.assertIn("stage=rs_wait_down device=npu:2 stream=30", self.messages()[1])
        self.compute.wait_event.assert_called_once_with("gmm1_done")
        self.communication.wait_event.assert_called_once_with("down_done")
        self.assertEqual(schedule.output_done, "rs_done")

    def test_decode_eligibility_reports_actual_fallback_conditions(self):
        routed_type = type("Routed", (), {})
        shared_type = type("Shared", (), {})
        latent_type = type("Latent", (), {})
        router_type = type("Router", (), {})
        modules = {
            "vllm_ascend.ops.fused_moe.router.fused_topk_router": SimpleNamespace(AscendFusedTopKRouter=router_type),
            "vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4": SimpleNamespace(
                AscendW4A8MXFPDynamicFusedMoEMethod=routed_type,
                AscendW4A8MXFPDynamicLinearMethod=shared_type,
            ),
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8": SimpleNamespace(
                AscendW8A8MXFP8DynamicLinearMethod=latent_type
            ),
        }
        self.torch.bfloat16 = object()
        self.context.attn_metadata = {"mla": SimpleNamespace(num_prefills=0)}
        communication = SimpleNamespace(moe_comm_type="mc2", is_draft_model=False)
        namespace = dict(
            self.namespace,
            get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda capability: True),
            HardwareCapability=SimpleNamespace(CANN_MEGAMOE_MXFP=object()),
            _EXTRA_CTX=communication,
            MoECommType=SimpleNamespace(MC2="mc2"),
            QuantType=SimpleNamespace(W4A8MXFP="fp4"),
        )
        runner_type = load_class(
            "vllm_ascend/ops/fused_moe/fused_moe.py", "AscendMoERunner", namespace, {"_can_run_kimi_decode"}
        )
        runner = runner_type()
        runner._shared_expert_diagnostics = self.cls("layer.1", self.config)
        runner.quant_type = "fp4"
        runner.moe_config = SimpleNamespace(is_sequence_parallel=True)
        runner._can_overlap_sp_shared_with = lambda transform: True
        runner.routed_input_transform = SimpleNamespace(quant_method=SimpleNamespace(quant_method=latent_type()))
        runner.routed_output_transform = object()
        router = router_type()
        router.custom_routing_function = router.tid2eid = router.bias_vl = None
        scheme = routed_type()
        scheme.use_expert_weight_list = False
        runner.routed_experts = SimpleNamespace(
            quant_method=SimpleNamespace(quant_method=scheme), router=router, dynamic_eplb=False
        )
        linear = SimpleNamespace(weight=SimpleNamespace(dtype=self.torch.bfloat16))
        runner.ascend_shared_experts = SimpleNamespace(
            layer=SimpleNamespace(gate_up_proj=linear, down_proj=linear), situ_activation=object()
        )
        hidden_states = SimpleNamespace(dtype=self.torch.bfloat16)
        with patch.dict("sys.modules", modules):
            self.assertTrue(runner._can_run_kimi_decode(hidden_states))
            self.assertIn("schedule=kimi_decode rejected=none", self.messages()[-1])
            for quantization, accepted in [(shared_type, True), (latent_type, False)]:
                linear.weight_scale = object()
                linear.quant_method = SimpleNamespace(quant_method=quantization())
                self.assertEqual(runner._can_run_kimi_decode(hidden_states), accepted)
            self.assertIn("rejected=shared_bf16_or_mxfp4", self.messages()[-1])
            communication.moe_comm_type = "alltoall"
            self.assertFalse(runner._can_run_kimi_decode(hidden_states))
            self.assertIn("rejected=mc2,shared_bf16_or_mxfp4", self.messages()[-1])


if __name__ == "__main__":
    unittest.main()
