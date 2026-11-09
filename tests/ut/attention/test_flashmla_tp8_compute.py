from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.attention import flashmla
from vllm_ascend.attention import mla_v1 as mla


@pytest.mark.parametrize("q", [1, 8])
@pytest.mark.parametrize("batch,rank,capture", [(1, 7, False), (9, 4, False), (16, 7, True)])
@pytest.mark.parametrize("capacity", [2, 4])
def test_runtime_and_capture_attach_local_flashmla(monkeypatch, q, batch, rank, capture, capacity):
    count, total = 8 * capacity, 8 * capacity * q
    lengths = torch.tensor([128] * batch + [0] * (count - batch), dtype=torch.int32)
    common = NS(
        num_reqs=count,
        max_query_len=q,
        num_input_tokens=total,
        num_actual_tokens=batch * q,
        query_start_loc=torch.arange(count + 1) * q,
        query_start_loc_cpu=torch.arange(count + 1) * q,
        _seq_lens_cpu=lengths,
        seq_lens_cpu=lengths,
        seq_lens=lengths,
        slot_mapping=torch.arange(total),
        positions=torch.arange(total),
        block_table_tensor=torch.arange(count * 2, dtype=torch.int32).reshape(count, 2),
        is_prefilling=torch.zeros(count, dtype=torch.bool),
        kimi_mla_num_actual_reqs=batch,
        kimi_mla_full_graph=True,
        kimi_mla_dummy=False,
        graph_pad_size=total,
        attn_state=None,
        causal=True,
    )
    builder = object.__new__(mla.AscendMLAMetadataBuilder)
    builder.kimi_compute_enabled = True
    builder.flashmla_state = object()
    builder.pcp_enabled = builder.dcp_enabled = False
    builder.decode_threshold = q
    builder.reorder_batch_threshold = q
    builder.speculative_config = NS(num_speculative_tokens=q - 1) if q > 1 else None
    builder.vllm_config = NS(compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY")))
    builder.model_config = NS(get_head_size=lambda: 576)
    builder.metadata_cls = mla.AscendMLAMetadata
    builder.attn_mask_builder = NS(get_splitfuse_attn_mask=lambda: None)
    builder.get_block_table_size = lambda *args: count
    builder.build_decode_metadata = lambda *args: NS(seq_lens_list=lengths.tolist(), cos=None, sin=None)
    monkeypatch.setattr(mla, "split_decodes_and_prefills", lambda shared, **kw: (count, 0, shared.num_actual_tokens, 0))
    monkeypatch.setattr(mla, "get_tp_group", lambda: NS(world_size=8, rank_in_group=rank))
    monkeypatch.setattr(
        flashmla.FlashMLAImpl,
        "load",
        lambda config: flashmla.FlashMLAImpl(
            config, lambda *a, **kw: None, lambda lengths, **kw: torch.zeros(32, dtype=torch.int32)
        ),
    )
    impl = NS(
        num_heads=96,
        scale=1.0,
        dtype=torch.bfloat16,
        use_mla_rope=False,
        vllm_config=NS(speculative_config=None, parallel_config=NS(cp_kv_cache_interleave_size=1)),
    )
    builder.compute_flashmla_state = flashmla.FlashMLAMetadataBuilder(
        impl, torch.device("cpu"), count, torch.zeros(2048, 2048, dtype=torch.int8)
    )
    pointers = None
    for step, actual in enumerate((count if capture else batch, 1, batch)):
        common.kimi_mla_num_actual_reqs = actual
        common.num_actual_tokens = actual * q
        common.slot_mapping.copy_(torch.arange(total))
        lengths[actual:] = 0
        lengths[:actual] = 131072
        result = builder.build_for_cudagraph_capture(common) if capture and step == 0 else builder.build(0, common)
        local = result.compute_decode.local_metadata
        partition = result.compute_decode.partition
        flash = local.external_flashmla
        valid = max(0, min(capacity, actual - rank * capacity)) * q
        assert partition.capacity == capacity * q and partition.valid_tokens == valid
        assert flash.query.shape == (capacity * q, 96, 576)
        assert flash.block_table.shape == (capacity, 2)
        assert flash.cache_lens.shape == flash.used_q.shape == (capacity,)
        assert flash.cu.shape == (capacity + 1,)
        assert flash.used_q.sum() == valid
        start = rank * capacity
        torch.testing.assert_close(
            flash.block_table[: valid // q], common.block_table_tensor[start : start + valid // q]
        )
        assert flash.cache_lens[: valid // q].eq(131072).all()
        assert flash.cache_lens[valid // q :].eq(0).all()
        torch.testing.assert_close(flash.slots[:valid], torch.arange(start * q, start * q + valid))
        assert flash.slots[valid:].eq(-1).all()
        addresses = tuple(x.data_ptr() for x in (flash.query, flash.schedule, flash.cu))
        assert pointers is None or addresses == pointers
        pointers = addresses


@pytest.mark.parametrize("q", [1, 8])
def test_fullhead_reader_preserves_fused_cache_and_output_order(monkeypatch, q):
    tokens = 2 * q
    fused = torch.arange(2 * 128 * 576, dtype=torch.float32).reshape(2, 128, 1, 576)
    original = fused.clone()
    query = torch.empty(tokens, 96, 576)
    nope = torch.arange(tokens * 96 * 512, dtype=torch.float32).reshape(tokens, 96, 512)
    rope = torch.arange(tokens * 96 * 64, dtype=torch.float32).reshape(tokens, 96, 64)
    latent = torch.arange(96 * tokens * 512, dtype=torch.float32).reshape(96, tokens, 512)
    observed = []

    def attention(actual_query, actual_cache, **kwargs):
        assert actual_cache is fused
        torch.testing.assert_close(actual_query[..., :512], nope)
        torch.testing.assert_close(actual_query[..., 512:], rope)
        observed.append(kwargs)
        return latent, None

    flash = NS(
        query=query,
        adapter=NS(attention=attention),
        block_table=object(),
        cache_lens=object(),
        cu=object(),
        used_q=object(),
        schedule=object(),
        attn_mask=object(),
    )
    impl = NS(kv_lora_rank=512, _v_up_proj=lambda value: value.transpose(0, 1).reshape(tokens, 96 * 512))
    monkeypatch.setattr(mla, "record_attention_compute_start", lambda: pytest.fail("compute reader released early"))
    result = mla.AscendMLAImpl._forward_external_flashmla(
        impl, NS(ql_nope=nope, q_pe=rope), fused, NS(external_flashmla=flash), defer_kv_ready_fence=True
    )
    torch.testing.assert_close(result, latent.transpose(0, 1).reshape(tokens, -1))
    torch.testing.assert_close(fused, original)
    assert observed[0]["metadata"] is flash.schedule
    assert observed[0]["block_table"] is flash.block_table


@pytest.mark.parametrize("q", [1, 8])
def test_native_capture_keeps_standard_backend(q):
    builder = object.__new__(mla.AscendMLAMetadataBuilder)
    builder.reorder_batch_threshold = q
    builder.dcp_enabled = False
    builder.flashmla_state = object()
    builder.kimi_compute_enabled = False
    builder._build_external_flashmla = lambda *args, **kw: ("native", kw["retain_for_graph"])
    common = NS(num_reqs=8, num_actual_tokens=8 * q, max_query_len=q, attn_state=None, is_prefilling=None)
    assert builder.build_for_cudagraph_capture(common) == ("native", True)
