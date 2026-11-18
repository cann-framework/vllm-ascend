import ast
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from test_shared_expert_diagnostics import ROOT, load_class


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


class TestKimiOverlapControls(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(multistream_overlap_shared_expert=False)
        self.namespace = {"get_ascend_config": lambda: self.config}

    def test_stream_creation_and_reuse_obey_switch_with_cached_streams(self):
        for name in ["_get_kimi_router_stream", "_get_kimi_shared_comm_stream"]:
            with self.subTest(name=name):
                factory = Mock(side_effect=lambda: object())
                namespace = dict(self.namespace, torch=SimpleNamespace(npu=SimpleNamespace(Stream=factory)))
                select = load_function("vllm_ascend/models/kimi_k3.py", name, namespace)
                model = SimpleNamespace()
                args = (model, object()) if "shared" in name else (model,)
                self.config.multistream_overlap_shared_expert = False
                self.assertIsNone(select(*args))
                factory.assert_not_called()
                self.config.multistream_overlap_shared_expert = True
                stream = select(*args)
                self.assertIs(select(*args), stream)
                factory.assert_called_once()
                self.config.multistream_overlap_shared_expert = False
                self.assertIsNone(select(*args))
                factory.assert_called_once()

    def test_shared_stream_requires_shared_experts(self):
        factory = Mock()
        namespace = dict(self.namespace, torch=SimpleNamespace(npu=SimpleNamespace(Stream=factory)))
        select = load_function("vllm_ascend/models/kimi_k3.py", "_get_kimi_shared_comm_stream", namespace)
        self.config.multistream_overlap_shared_expert = True
        self.assertIsNone(select(SimpleNamespace(), None))
        factory.assert_not_called()

    def core_scope(self, runtime=None):
        self.logger = Mock()
        namespace = dict(
            self.namespace,
            contextmanager=contextmanager,
            torch_npu=runtime if runtime is not None else SimpleNamespace(),
            logger=self.logger,
        )
        return load_function(
            "vllm_ascend/ops/fused_moe/shared_expert_diagnostics.py", "shared_expert_core_limit", namespace
        )

    def test_disabled_core_limit_does_not_access_npugraph_ex(self):
        scope = self.core_scope()
        entered = Mock()
        with scope("router", (32, 1)):
            entered()
        entered.assert_called_once()
        self.logger.info_once.assert_not_called()

    def test_enabled_core_limit_logs_parameters_and_exits_on_exception(self):
        actions = []

        @contextmanager
        def limit(*core_limit):
            actions.append(("enter", core_limit))
            try:
                yield
            finally:
                actions.append(("exit", core_limit))

        runtime = SimpleNamespace(
            npu=SimpleNamespace(npugraph_ex=SimpleNamespace(scope=SimpleNamespace(limit_core_num=limit)))
        )
        scope = self.core_scope(runtime)
        self.config.multistream_overlap_shared_expert = True
        for stage, core_limit in [("router", (32, 1)), ("shared_gate_up", (32, 1)), ("shared_gate_up", (16, 32))]:
            with self.subTest(stage=stage, core_limit=core_limit):
                with self.assertRaisesRegex(RuntimeError, "kernel submission"), scope(stage, core_limit):
                    raise RuntimeError("kernel submission")
                self.assertEqual(actions[-2:], [("enter", core_limit), ("exit", core_limit)])
                call = self.logger.info_once.call_args.args
                message = call[0] % call[1:]
                self.assertIn(f"stage={stage} aic_num={core_limit[0]} aiv_num={core_limit[1]}", message)

    def test_disabled_decode_schedule_rejects_before_stream_allocation(self):
        stream = Mock(side_effect=AssertionError("allocated a stream"))
        namespace = dict(self.namespace, shared_experts_calculation_stream=stream)
        schedule = load_class(
            "vllm_ascend/ops/fused_moe/kimi_decode_schedule.py", "KimiDecodeSchedule", namespace, {"__init__"}
        )
        with self.assertRaisesRegex(RuntimeError, "requires multistream_overlap_shared_expert"):
            schedule(object(), object(), object())
        stream.assert_not_called()

    def test_disabled_decode_and_precomputed_topk_exit_before_runtime_access(self):
        runner_type = load_class(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            "AscendMoERunner",
            dict(self.namespace),
            {"_can_run_kimi_decode", "_prepare_kimi_decode_topk"},
        )
        runner = runner_type()
        self.assertFalse(runner._can_run_kimi_decode(object()))
        self.assertIsNone(runner._prepare_kimi_decode_topk(object(), object()))

    def test_disabled_topk_ignores_cached_router_stream(self):
        select = Mock(return_value=("weights", "ids"))
        namespace = dict(self.namespace)
        routed_type = load_class(
            "vllm_ascend/ops/fused_moe/routed_experts.py",
            "AscendRoutedExperts",
            namespace,
            {"_select_experts_with_stream"},
        )
        routed = routed_type()
        routed._select_experts = select
        self.assertEqual(
            routed._select_experts_with_stream("hidden", "logits", False, None, object()), ("weights", "ids")
        )
        select.assert_called_once_with("hidden", "logits", False, None)

    def test_enabled_topk_uses_router_stream_and_joins_result(self):
        self.config.multistream_overlap_shared_expert = True
        main = SimpleNamespace(record_event=Mock(return_value="prepared"), wait_event=Mock())
        router = SimpleNamespace(record_event=Mock(return_value="done"), wait_event=Mock())
        current = [main]
        output = SimpleNamespace(record_stream=Mock())

        @contextmanager
        def switch(stream):
            previous = current[0]
            current[0] = stream
            try:
                yield
            finally:
                current[0] = previous

        def select(*args):
            self.assertIs(current[0], router)
            return output, output

        namespace = dict(
            self.namespace,
            torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: current[0])),
            _EXTRA_CTX=SimpleNamespace(moe_comm_type="mc2"),
            MoECommType=SimpleNamespace(MC2="mc2"),
            npu_stream_switch=switch,
        )
        routed_type = load_class(
            "vllm_ascend/ops/fused_moe/routed_experts.py",
            "AscendRoutedExperts",
            namespace,
            {"_select_experts_with_stream"},
        )
        routed = routed_type()
        routed.moe_config = SimpleNamespace(is_sequence_parallel=True)
        routed._select_experts = select
        routed._select_experts_with_stream(output, output, False, None, router)
        router.wait_event.assert_called_once_with("prepared")
        main.wait_event.assert_called_once_with("done")

    def test_disabled_front_clears_stale_topk_and_uses_calling_stream(self):
        states, projected, logits = object(), object(), object()
        compute = Mock(return_value=logits)
        stream_switch = Mock(return_value=nullcontext())
        namespace = dict(
            self.namespace,
            nullcontext=nullcontext,
            npu_stream_switch=stream_switch,
            PreparedSharedExpertInput=lambda hidden_states: SimpleNamespace(
                hidden_states=hidden_states, ready_event=None
            ),
        )
        runner_type = load_class(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            "AscendMoERunner",
            namespace,
            {"apply_routed_input_transform", "_can_overlap_sp_shared_with"},
        )
        runner = runner_type()
        runner.ascend_shared_experts = SimpleNamespace(multistream_overlap=True)
        runner.routed_input_transform = object()
        runner._kimi_gate = object()
        runner._kimi_router_stream = object()
        runner._kimi_shared_comm_stream = object()
        runner._kimi_topk = object()
        runner._kimi_shared_output_done = object()
        runner._can_run_kimi_decode = Mock(side_effect=AssertionError("Decode schedule queried while disabled"))
        runner._compute_router_logits = compute
        self.assertEqual(runner.apply_routed_input_transform(states, routed_input=projected), (projected, states))
        stream_switch.assert_called_once_with(None, enabled=False)
        self.assertIsNone(runner._kimi_topk)
        self.assertIsNone(runner._kimi_shared_output_done)
        self.assertIsNone(runner._kimi_shared_input_ready)
        self.assertIsNone(runner._kimi_router_done)
        self.assertFalse(runner._kimi_decode_active)
        compute.assert_called_once_with(states, states)


if __name__ == "__main__":
    unittest.main()
