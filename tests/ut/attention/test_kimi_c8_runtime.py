from types import SimpleNamespace as Ns
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.attention import kimi_c8_decode as projections
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.kimi_c8_static import write_static_fp8_cache
from vllm_ascend.attention.mla_v1 import AscendMLAImpl, AscendMLAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


def runtime(graph=False, world=8, padding=2):
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.layer_name, impl.num_heads, impl.scale = "mla", 2, 0.125
    impl.kimi_c8_static_fp8, impl.kimi_c8_graph, impl.dcp_enabled = True, graph, world > 1
    impl.pcp_enabled = impl.use_mla_rope = False
    impl.dcp_size, impl.dcp_rank, impl.fak_descale_float = world, 1 if world > 1 else 0, torch.tensor([0.25])
    impl.kimi_c8_manager_block_size, impl.kimi_c8_page_stride_bytes = 1536, (12 + padding) * 73728
    impl.vllm_config = Ns(model_config=Ns(max_model_len=133120), parallel_config=Ns(cp_kv_cache_interleave_size=1536))
    starts, lengths = torch.tensor([0, 1, 3], dtype=torch.int32), torch.tensor([1537, 7], dtype=torch.int32)
    metadata = Ns(
        num_actual_tokens=3,
        num_decode_tokens=3,
        num_decodes=2,
        num_prefills=0,
        query_start_loc=starts,
        c8_query_start_loc_cpu=starts.clone(),
        seq_lens_cpu=lengths.clone(),
        c8_global_seq_lens=lengths,
        c8_input_positions=torch.tensor([1536, 5, 6]),
        slot_mapping=torch.arange(3),
        c8_step_cache={},
        decode=Ns(
            seq_lens_list=[1537, 7],
            actual_seq_lengths_q=[1, 3],
            block_table=torch.arange(24, dtype=torch.int32).reshape(2, 12).flip(0),
        ),
    )
    return impl, metadata, torch.zeros(2 * (12 + padding), 1, 18, 128, 32, dtype=torch.uint8)


@pytest.fixture
def metadata_op(monkeypatch):
    import cann_ops_transformer.ops as ops

    op = MagicMock(side_effect=lambda **kwargs: torch.ones(1, dtype=torch.int32))
    monkeypatch.setattr(ops, "quant_flash_mla_with_kvcache_metadata", op, raising=False)
    return op


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("world,padding", [(1, 0), (8, 2)])
def test_decode_q1_geometry_scale_and_local_v_projection(monkeypatch, metadata_op, graph, world, padding):
    impl, meta, raw = runtime(graph, world, padding)
    impl._dcp_all_gather_fragments = MagicMock(side_effect=lambda value, dim: (torch.cat([value] * world, dim),))
    impl._merge_dcp_attention_output = MagicMock(return_value=torch.ones(3, 2, 512))

    def local_v(value):
        assert value.shape == (3, 2, 512)  # Merge must precede TP-local V projection.
        return value

    impl._v_up_proj_batch_major = local_v
    fa = MagicMock(return_value=(torch.ones(3, 2 * world, 512), torch.zeros(2 * world, 3)))
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "quant_flash_mla_with_kvcache", fa, raising=False)
    projected = Ns(q=torch.zeros(3, 2, 576, dtype=torch.float8_e4m3fn), q_descale=torch.ones(3, 2))
    if graph:
        meta.c8_global_seq_lens = torch.cat((meta.c8_global_seq_lens, torch.zeros(6, dtype=torch.int32)))[
            : 8 if padding else 1
        ]
        meta.decode.input_positions, meta.c8_input_positions = meta.c8_input_positions, None
        meta.decode.seq_lens_list = meta.decode.actual_seq_lengths_q = None
        meta.query_start_loc = torch.tensor([0, 1, 3, 3, 3], dtype=torch.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError("Graph decode must not synchronize device values")

    with monkeypatch.context() as guard:
        if graph:
            guard.setattr(torch.Tensor, "item", forbidden)
            guard.setattr(torch.Tensor, "tolist", forbidden)
        result, _ = impl._run_kimi_c8_decode(torch.ones(3, 1), (raw,), meta, projection=projected)
    q, cache, descale, kv_descale, table, lens, *_ = fa.call_args.args
    expected_table = meta.decode.block_table + torch.div(meta.decode.block_table, 12, rounding_mode="floor") * padding
    assert result.shape == (3, 2, 512) and cache.data_ptr() == raw.data_ptr()
    assert q.shape == (3, 2 * world, 576) and descale.shape == q.shape[:2]
    assert cache.dtype == q.dtype == torch.float8_e4m3fn and kv_descale is impl.fak_descale_float
    torch.testing.assert_close(table, expected_table.repeat_interleave(torch.tensor([1, 2]), dim=0))
    assert lens.tolist() == ([1, 0, 0] if world > 1 else [1537, 6, 7])
    assert fa.call_args.kwargs["softmax_scale"] == 0.125 and fa.call_args.kwargs["mask_mode"] == 0
    assert fa.call_args.kwargs["cu_seqlens_q"].tolist() == [0, 1, 2, 3]
    assert metadata_op.call_args.kwargs["num_heads_q"] == 2 * world
    if world > 1:
        assert impl._merge_dcp_attention_output.call_args.args[1].shape == (3, 16, 1)
        assert [call.args[0].dtype for call in impl._dcp_all_gather_fragments.call_args_list] == [
            torch.uint8,
            torch.float32,
        ]


@pytest.mark.parametrize("graph", [False, True])
def test_metadata_refreshes_values_shares_only_within_group_and_skips_empty(metadata_op, graph):
    impl, meta, raw = runtime(graph)
    other, separate, _ = runtime(graph)
    other.dcp_rank = 7
    if not graph:
        meta.decode.seq_lens_list, meta.decode.actual_seq_lengths_q = [1537], None
        meta.query_start_loc = torch.empty(3, device="meta")
    first = impl._prepare_kimi_c8_step(meta, start_step=True)
    empty_rank = other._prepare_kimi_c8_step(separate, start_step=True)
    assert empty_rank[0].cache_seqlens.eq(0).all()
    impl.layer_name = "next"
    assert impl._prepare_kimi_c8_step(meta, start_step=True) is first
    meta.num_actual_tokens = meta.num_decode_tokens = 0
    impl._prelaunch_kimi_c8_metadata({"mla": meta})
    output = torch.ones(3, 1)
    impl._forward_kimi_c8("mla", output, (raw,), meta, output)
    assert output.eq(0).all() and metadata_op.call_count == 2
    meta.num_actual_tokens = meta.num_decode_tokens = 3
    meta.c8_global_seq_lens.add_(1)
    meta.c8_input_positions.add_(1)
    meta.decode.seq_lens_list = [1538] if not graph else [1538, 8]
    meta.seq_lens_cpu.add_(1)
    meta.decode.block_table.fill_(0)
    impl.layer_name = "mla"
    updated = impl._prepare_kimi_c8_step(meta, start_step=True)
    assert updated is not first and updated[0].cache_seqlens.tolist() == [2, 0, 0]
    assert updated[0].block_table.eq(0).all() and metadata_op.call_count == 3
    assert other._prepare_kimi_c8_step(separate) is empty_rank


@pytest.fixture
def builder(monkeypatch):
    config = Ns(
        model_config=Ns(
            max_model_len=16384, dtype=torch.bfloat16, get_head_size=lambda: 576, hf_text_config=Ns(qk_rope_head_dim=64)
        ),
        scheduler_config=Ns(max_num_seqs=32, max_num_batched_tokens=16384, enable_chunked_prefill=False),
        parallel_config=Ns(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        speculative_config=None,
        cache_config=Ns(block_size=1536),
        kv_transfer_config=None,
        compilation_config=Ns(static_forward_context={"mla": Ns(impl=Ns(use_mla_rope=False, use_flashmla=True))}),
    )

    def parent(self, spec, names, config, device, metadata_cls, *args):
        self.kv_cache_spec, self.metadata_cls, self.device = spec, metadata_cls, device
        self.vllm_config, self.model_config, self.chunked_prefill_workspace_size = config, config.model_config, 16384

    monkeypatch.setattr(mla_v1.MLACommonMetadataBuilder, "__init__", parent)
    monkeypatch.setattr(mla_v1, "enable_dcp", lambda: False)
    monkeypatch.setattr(mla_v1, "get_ascend_config", lambda: Ns())
    spec = AscendMLAAttentionSpec(
        block_size=1536, num_kv_heads=1, head_size=576, dtype=torch.float8_e4m3fn, static_fp8_pa_nz=True
    )
    result = AscendMLAMetadataBuilder(spec, ["mla"], config, torch.device("cpu"))
    result.get_block_table_size = lambda common, phase: common.num_reqs
    return result


@pytest.mark.parametrize("query_lens", [[1, 1], [512] * 32, [16384]])
@pytest.mark.parametrize("device_lengths", [False, True])
def test_builder_aliases_runner_inputs_and_routes_profile_to_prefill(builder, query_lens, device_lengths):
    starts = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
    lengths = torch.tensor(query_lens, dtype=torch.int32) + 10
    positions = torch.cat([torch.arange(length - query, length) for length, query in zip(lengths, query_lens)])
    source = AscendCommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        seq_lens=lengths if device_lengths else None,
        seq_lens_cpu=lengths.clone(),
        _seq_lens_cpu=lengths.clone(),
        num_reqs=len(query_lens),
        num_actual_tokens=int(starts[-1]),
        num_input_tokens=int(starts[-1]),
        max_query_len=max(query_lens),
        max_seq_len=int(lengths.max()),
        positions=positions,
        graph_pad_size=-1,
        block_table_tensor=torch.zeros(len(query_lens), 8, dtype=torch.int32),
        slot_mapping=torch.arange(int(starts[-1])),
        attn_state=AscendAttentionState.DecodeOnly,
    )
    result = builder.build(0, source)
    launcher, _, _ = runtime()
    launcher._prepare_kimi_c8_step = MagicMock()
    launcher._prelaunch_kimi_c8_metadata({"other": result})
    launcher._prelaunch_kimi_c8_metadata({"mla": result})
    assert launcher._prepare_kimi_c8_step.call_count == int(max(query_lens) <= 16)
    assert builder.flashmla_state is None and result.external_flashmla is None
    assert result.c8_input_positions.data_ptr() == positions.data_ptr()
    assert result.c8_query_start_loc_cpu is source.query_start_loc_cpu
    if max(query_lens) > 16:
        assert result.num_prefills == len(query_lens) and result.num_decode_tokens == 0
    else:
        assert result.decode.seq_lens_list == [11, 11]
        assert (
            result.c8_global_seq_lens is None
            if not device_lengths
            else result.c8_global_seq_lens.data_ptr() == lengths.data_ptr()
        )


@pytest.mark.parametrize("omit_fields", [False, True])
def test_real_hybrid_capture_refreshes_persistent_buffers(builder, monkeypatch, metadata_op, omit_fields):
    from vllm_ascend.worker.v2 import input_batch as batches
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

    monkeypatch.setattr(batches, "update_cos_sin", lambda _: None)
    buffers = batches.AscendInputBuffers(32, 64, torch.device("cpu"))
    batch = batches.AscendInputBatch.make_dummy(8, 8, buffers)
    state = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    state.vllm_config, state.max_model_len = builder.vllm_config, 16384
    group = Ns(layer_names=["mla"], get_metadata_builder=lambda _: builder)
    cfg = Ns(kv_cache_groups=[Ns(kv_cache_spec=builder.kv_cache_spec)])
    meta = state.prepare_attn(
        batch,
        CUDAGraphMode.NONE,
        (torch.zeros(32, 8, dtype=torch.int32),),
        torch.zeros(1, 8, dtype=torch.int64),
        [[group]],
        cfg,
        for_capture=True,
    )["mla"]
    assert meta.c8_input_positions.data_ptr() == buffers.positions.data_ptr()
    assert meta.query_start_loc.data_ptr() == buffers.query_start_loc.data_ptr()
    impl, _, _ = runtime(True, 1, 0)
    impl.vllm_config = builder.vllm_config
    if omit_fields:
        meta.c8_input_positions = meta.c8_global_seq_lens = None
    assert impl._prepare_kimi_c8_step(meta, start_step=True)[0].cache_seqlens.tolist() == [1] * 8
    buffers.positions[:8].fill_(5)
    buffers.seq_lens[:8].fill_(6)
    assert impl._prepare_kimi_c8_step(meta, start_step=True)[0].cache_seqlens.tolist() == [6] * 8


@pytest.mark.parametrize("decode_tokens", [0, 2])
def test_real_forward_preserves_write_wait_gate_and_save_order(monkeypatch, decode_tokens):
    impl, meta, raw = runtime(False, 1)
    impl.num_heads, impl.v_head_dim, impl.use_output_gate = 1, 3, True
    meta.num_decode_tokens, meta.num_decodes, meta.num_prefills = decode_tokens, decode_tokens, 1
    events = []
    projection = projections.KimiC8Projection(
        None, None, torch.ones(3, 1), torch.ones(3, 576), (torch.ones(3, 1, 64), torch.ones(3, 1, 512))
    )
    impl._prepare_kimi_c8_step = lambda *a, **kw: None
    impl.layerwise_kv_cache_hook = Ns(wait_for_layer=lambda _: events.append("layer_wait"))
    monkeypatch.setattr(projections, "project_and_write_kimi_c8", lambda *a, **kw: events.append("write") or projection)
    impl._run_kimi_c8_decode = lambda *a, **kw: (events.append("decode") or torch.full((decode_tokens, 3), 3.0), None)
    impl.mla_preprocess_prefill = lambda *a, **kw: events.append("prepare") or Ns(
        **dict.fromkeys(("q_nope", "q_pe", "k_nope", "k_pe", "value"), torch.ones(1))
    )
    impl._forward_prefill = lambda *a, **kw: events.append("prefill") or torch.full((3 - decode_tokens, 3), 5.0)
    impl.g_proj = lambda value: (events.append("gate") or torch.zeros_like(value), None)
    impl.o_proj = lambda value, **kw: (events.append("out") or value + 1, None)
    for name, event in (
        ("wait_for_kv_layer_from_connector", "wait"),
        ("notify_kv_cache_written", "written"),
        ("maybe_save_kv_layer_to_connector", "save"),
    ):
        monkeypatch.setattr(mla_v1, name, lambda *a, event=event: events.append(event))
    output = torch.empty(3, 3)
    assert impl.forward("mla", torch.ones_like(output), (raw,), meta, output) is output
    torch.testing.assert_close(output[:decode_tokens], torch.full_like(output[:decode_tokens], 2.5))
    torch.testing.assert_close(output[decode_tokens:], torch.full_like(output[decode_tokens:], 3.5))
    assert events == ["wait", "layer_wait", "write", "written"] + (["decode"] if decode_tokens else []) + [
        "prepare",
        "prefill",
        "gate",
        "out",
        "save",
    ]


@pytest.mark.parametrize("padding", [0, 2])
def test_prefill_reads_full_precision_history_exactly_once(monkeypatch, padding):
    impl, _, cache = runtime(False, 1, padding)
    impl.fa_quant_layer = impl.support_fp8_attention = True
    impl.kv_lora_rank, impl.qk_rope_head_dim, impl.head_padding = 512, 64, 0
    impl.num_heads = impl.v_head_dim = impl.qk_nope_head_dim = 1
    values = torch.cat((torch.full((1, 512), 2.0), torch.full((1, 64), 6.0)), -1)
    write_static_fp8_cache(values, cache, torch.tensor([(12 + padding) * 128]), impl.fak_descale_float)
    chunk = Ns(
        seq_tot=[1],
        starts=torch.tensor([[0]]),
        starts_cpu=torch.tensor([[0]]),
        chunk_seq_lens_npu=torch.tensor([[1]]),
        chunk_seq_lens=torch.tensor([[1]]),
        chunk_actual_seq_lengths_kv_list=[[1]],
    )

    def project(value):
        torch.testing.assert_close(value, values[:, :512].to(torch.bfloat16))
        return torch.ones(1, 2, dtype=torch.bfloat16), None

    def attention(*args, **kwargs):
        torch.testing.assert_close(kwargs["key_rope"], values[:, 512:].to(torch.bfloat16).unsqueeze(1))
        return torch.full((1, 1, 1), 5.0), torch.zeros(1, 1)

    impl.kv_b_proj = project
    monkeypatch.setattr(mla_v1.torch_npu, "npu_fused_infer_attention_score", attention)
    monkeypatch.setattr(
        mla_v1, "merge_prefill_chunks", lambda outputs, *a, **kw: (torch.stack(outputs).mean(0), None, None)
    )
    prefill = Ns(
        chunked_context=chunk,
        block_table=torch.arange(12, 24, dtype=torch.int32).reshape(1, 12),
        actual_seq_lengths_q=[1],
    )
    output, _ = impl._compute_prefill_context(
        torch.ones(1, 1, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 64, dtype=torch.bfloat16),
        (cache,),
        64,
        Ns(prefill=prefill),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1),
    )
    torch.testing.assert_close(output, torch.full((1, 1, 1), 2.5))


@pytest.mark.parametrize("inner_stride", [1, 2])
def test_large_page_native_writer_uses_bounded_workspace(monkeypatch, inner_stride):
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.kv_lora_rank, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 4, 2, 2
    impl.v_head_dim, impl.num_heads, impl.num_kv_heads = 2, 1, 1
    impl.kv_b_proj = Ns(weight=torch.zeros(4, 4), quant_method=UnquantizedLinearMethod())
    impl.fa_quant_layer = impl.enable_mlapo = impl.use_flashmla = impl.is_draft_cache = impl.dcp_enabled = False
    impl.use_mla_rope, impl.dtype = True, torch.bfloat16
    impl.vllm_config = Ns(cache_config=Ns(block_size=1536), scheduler_config=Ns(max_num_batched_tokens=16))
    impl._flashmla_writer_buffers = None
    impl.supports_prolog_v3_writer = lambda: True
    impl._prepare_prolog_v3_weights_after_loading = impl._prepare_strided_kv_workspace = lambda: None
    monkeypatch.setattr(mla_v1.torch_npu, "npu_format_cast", lambda value, *a: value)
    monkeypatch.setattr(mla_v1, "maybe_trans_nz", lambda value: value)
    impl.process_weights_after_loading(torch.bfloat16)
    assert impl._flashmla_writer_buffers[0].shape[1] == 128
    backing = torch.full((2, 1540, 1, 4 * inner_stride), 17.0)
    cache = backing[:, :1536, :, ::inner_stride]
    impl.q_a_layernorm = impl.kv_a_layernorm = Ns(weight=torch.ones(4), variance_epsilon=1e-6)
    impl._prepare_prolog_v3_token = lambda value: (
        value,
        {},
        dict.fromkeys(("weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr"), torch.ones(1)),
    )

    def prolog(**kwargs):
        assert kwargs["kv_cache"].shape[1] == 128
        kwargs["kv_cache"][0, :2].fill_(3)
        return (torch.zeros(2),)

    scatter = MagicMock()
    monkeypatch.setattr(mla_v1.torch_npu, "npu_mla_prolog_v3", prolog, raising=False)
    monkeypatch.setattr(mla_v1.DeviceOperator, "reshape_and_cache", scatter)
    slots = torch.tensor([0, 1536])
    impl._run_prolog_v3(
        torch.zeros(2, 4), torch.ones(2, 4), torch.zeros(2, 4), (cache, cache), slots, quantize_query=False
    )
    assert scatter.call_args.kwargs["key_cache"] is cache and scatter.call_args.kwargs["slot_mapping"] is slots
    assert backing[:, 1536:].eq(17).all()


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY])
def test_post_load_preserves_static_scale_address(monkeypatch, mode):
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.kv_lora_rank, impl.qk_nope_head_dim, impl.v_head_dim, impl.num_heads = 512, 128, 512, 1
    impl.kv_b_proj = Ns(weight=torch.zeros(640, 512), quant_method=UnquantizedLinearMethod())
    impl.kimi_c8_mode, impl.fa_quant_layer, impl.pcp_enabled, impl.kimi_c8_kv_scale_value = True, False, False, 0.25
    impl.vllm_config = Ns(compilation_config=Ns(cudagraph_mode=mode), kv_transfer_config=Ns(is_kv_producer=True))
    monkeypatch.setattr(mla_v1.torch_npu, "npu_format_cast", lambda value, *a: value)
    monkeypatch.setattr(projections, "prepare_kimi_c8_prolog_runtime", lambda _: False)
    impl.process_weights_after_loading(torch.bfloat16)
    pointer = impl.fak_descale_float.data_ptr()
    impl.kimi_c8_kv_scale_value = 0.5
    impl.process_weights_after_loading(torch.bfloat16)
    assert impl.fak_descale_float.data_ptr() == pointer and impl.fak_descale_float.item() == 0.5
    assert impl.kimi_c8_graph == (mode == CUDAGraphMode.FULL_DECODE_ONLY) and not impl.use_flashmla


def test_c8_graph_never_updates_uncaptured_legacy_fia(monkeypatch):
    from vllm_ascend.attention.context_parallel import mla_cp

    context = Ns(attn_metadata={"mla": Ns(decode=Ns(), external_flashmla=None, c8_static_fp8=True)})
    for module, impl in ((mla_v1, AscendMLAImpl), (mla_cp, mla_cp.AscendMlaDCPImpl)):
        monkeypatch.setattr(module, "_EXTRA_CTX", Ns(is_draft_model=False))
        monkeypatch.setattr(module, "get_graph_params", lambda: None)
        impl.update_graph_params(None, context, 2)


@pytest.mark.parametrize("rope,decode_tokens", [(False, 0), (True, 1)])
def test_native_projection_keeps_bf16_prefill_and_rotates_query_and_kv(rope, decode_tokens):
    class Linear(torch.nn.Linear):
        def forward(self, value):
            return super().forward(value), None

    fused, query = Linear(4, 580, bias=False), Linear(4, 12 * 192, bias=False)
    torch.nn.init.constant_(fused.weight, 0.01)
    torch.nn.init.constant_(query.weight, 0.005)
    impl, _, _ = runtime(False, 1)
    impl.num_heads, impl.num_kv_heads, impl.q_lora_rank = 12, 1, 4
    impl.kv_lora_rank, impl.qk_rope_head_dim, impl.qk_nope_head_dim, impl.qk_head_dim = 512, 64, 128, 192
    impl.use_mla_rope, impl.fused_qkv_a_proj, impl.q_proj = rope, fused, query
    impl.q_a_layernorm, impl.kv_a_layernorm = lambda value: value * 2, lambda value: value * 3
    impl.W_UK_T = torch.full((12, 128, 512), 0.001)
    impl.rope_single = lambda value, cos, sin: value + cos - sin
    cache = torch.full((1, 1, 18, 128, 32), 17, dtype=torch.uint8)
    result = projections.project_and_write_kimi_c8(
        impl,
        torch.ones(2, 4),
        cache,
        torch.tensor([0, -1]),
        2.0,
        num_decode_tokens=decode_tokens,
        cos=torch.full((2, 1, 64), 2.0),
        sin=torch.ones(2, 1, 64),
    )
    kv = fused(torch.ones(2, 4))[0][:, 4:]
    expected = torch.cat((kv[:, :512] * 3, kv[:, 512:] + int(rope)), -1)
    stored = cache[0, 0, :, 0].flatten().view(torch.float8_e4m3fn).float() * 2
    torch.testing.assert_close(stored, expected[0], rtol=0.07, atol=0.002)
    assert cache[0, 0, :, 127].eq(17).all()
    if decode_tokens:
        nope, aux = impl._q_proj_and_k_up_proj(result.q_c[:1])
        torch.testing.assert_close(
            result.q.float() * result.q_descale[..., None], torch.cat((nope, aux + 1), -1), rtol=0.07, atol=0.002
        )
    else:
        assert result.q is None and result.q_descale is None
