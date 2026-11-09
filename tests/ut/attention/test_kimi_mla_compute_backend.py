# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.attention import kimi_mla_dp_backend as backend


def test_dspark_target_config_does_not_enable_draft_compute(monkeypatch):
    config = NS(
        model_config=NS(hf_config=NS(model_type="kimi_k3", architectures=["KimiK3ForCausalLM"])),
        kv_transfer_config=NS(is_kv_consumer=True, is_kv_producer=False),
    )
    monkeypatch.setattr(backend, "get_ascend_config", lambda: NS(enable_kimi_mla_decode_dp=True))
    monkeypatch.setattr(backend.compilation_backends, "model_tag", "backbone")
    assert backend.compute_enabled(config)
    monkeypatch.setattr(backend.compilation_backends, "model_tag", "dspark_head")
    draft = NS(vllm_config=config, g_proj=None)
    draft.kimi_compute_enabled = backend.compute_enabled(draft)
    assert not draft.kimi_compute_enabled
    backend.prepare_compute_weights(draft)
    monkeypatch.setattr(backend.compilation_backends, "model_tag", "backbone")
    assert not backend.compute_enabled(draft)
    backend.prepare_compute_weights(draft)
    assert not hasattr(draft, "_kimi_compute_view")
    assert backend.compute_enabled(config)


class NativeMethod(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class Group:
    world_size = 2
    rank_in_group = 0

    def __init__(self):
        self.calls = []

    def all_gather(self, value, dim=0):
        self.calls.append((tuple(value.shape), dim))
        return torch.cat((value, value + 100), dim=dim)


def _impl(method=None):
    q = torch.nn.Module()
    q.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(3, 2), requires_grad=False)
    q.weight.output_dim, q.weight.checkpoint_tag = 0, "native"
    q.bias = None
    q.quant_method = method or NativeMethod()
    q.output_size_per_partition, q.output_size = 3, 6
    q.input_size = q.input_size_per_partition = 2
    return NS(
        q_proj=q,
        g_proj=object(),
        num_heads=1,
        num_kv_heads=1,
        W_UK_T=torch.arange(4.0).reshape(1, 2, 2),
        W_UV=torch.arange(4.0).reshape(1, 2, 2),
    )


def test_full_query_view_preserves_native_parameters(monkeypatch):
    group, impl = Group(), _impl()
    monkeypatch.setattr(backend, "get_tp_group", lambda: group)
    monkeypatch.setattr(backend, "compute_enabled", lambda value: True)
    monkeypatch.setattr(torch.npu, "Stream", lambda: "gate-stream")
    original = impl.q_proj.weight
    backend.prepare_compute_weights(impl)
    view = impl._kimi_compute_view
    assert impl.num_heads == 1 and view.num_heads == 2
    assert impl.q_proj.weight is original and impl.q_proj.weight.checkpoint_tag == "native"
    assert view.q_proj.layer._parameters is not impl.q_proj._parameters
    assert view.q_proj.layer.weight.checkpoint_tag == "native"
    assert view.q_proj.layer.output_size_per_partition == 6
    actual, bias = view.q_proj.as_linear(torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(actual, F.linear(torch.tensor([[1.0, 2.0]]), torch.cat((original, original + 100))))
    assert bias is None and group._kimi_mla_gate_stream == "gate-stream"
    torch.testing.assert_close(view.W_UK_T, torch.cat((impl.W_UK_T, impl.W_UK_T + 100)))
    assert len(group.calls) == 3
    backend.prepare_compute_weights(impl)
    assert len(group.calls) == 3


def test_quantized_query_gathers_all_declared_metadata(monkeypatch):
    specs = (NS(attr_name="weight", gather_dim=1), NS(attr_name="weight_scale", gather_dim=0))
    scheme = NS(
        supports_weight_switch=True,
        weight_switch_output_gather_specs=specs,
        weight_switch_output_repeat_specs=(NS(attr_name="input_scale", repeat_dim=0),),
    )
    impl, group = _impl(NS(quant_method=scheme)), Group()
    impl.q_proj.weight = torch.nn.Parameter(impl.q_proj.weight.T.contiguous(), requires_grad=False)
    impl.q_proj.weight_scale = torch.nn.Parameter(torch.arange(3.0), requires_grad=False)
    impl.q_proj.input_scale = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    monkeypatch.setattr(backend, "get_tp_group", lambda: group)
    monkeypatch.setattr(backend, "compute_enabled", lambda value: True)
    monkeypatch.setattr(torch.npu, "Stream", lambda: "gate-stream")
    backend.prepare_compute_weights(impl)
    full = impl._kimi_compute_view.q_proj.layer
    assert full.weight.shape == (2, 6)
    torch.testing.assert_close(full.weight_scale, torch.tensor([0.0, 1.0, 2.0, 100.0, 101.0, 102.0]))
    torch.testing.assert_close(full.input_scale, torch.tensor([2.0, 2.0]))
    assert impl.q_proj.weight.shape == (2, 3) and impl.q_proj.input_scale.shape == (1,)


def test_slot_mask_uses_live_buffer_values():
    slots = torch.tensor([0, 1, -1, -1])
    values = torch.tensor([[2.0, 3.0], [4.0, 5.0], [float("nan")] * 2, [float("nan")] * 2])
    result = backend._zero_padding(values, slots)
    torch.testing.assert_close(result, torch.tensor([[2.0, 3.0], [4.0, 5.0], [0.0, 0.0], [0.0, 0.0]]))
    slots[1] = -1
    assert torch.equal(backend._zero_padding(values, slots)[1], torch.zeros(2))


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_query_gather_preserves_payload_and_parameter_attrs(monkeypatch, dtype):
    impl, group, casts = _impl(), Group(), []
    bits = torch.tensor([[0, 128], [127, 255], [1, 129]], dtype=torch.uint8)
    impl.q_proj.weight = torch.nn.Parameter(bits.view(dtype), requires_grad=False)
    impl.q_proj.weight.output_dim, impl.q_proj.weight.tag = 0, "fp8"

    def cast(value, fmt):
        assert fmt == 2
        casts.append(value.dtype)
        return value

    def gather(value, dim=0):
        if value.element_size() == 1:
            assert value.dtype == torch.uint8
            return torch.cat((value, value.bitwise_xor(128)), dim)
        return torch.cat((value, value), dim)

    group.all_gather = gather
    monkeypatch.setattr(backend, "get_tp_group", lambda: group)
    monkeypatch.setattr(backend, "compute_enabled", lambda value: True)
    monkeypatch.setattr(backend, "torch_npu", NS(npu_format_cast=cast), raising=False)
    monkeypatch.setattr(torch.npu, "Stream", lambda: "gate-stream")
    backend.prepare_compute_weights(impl)
    full = impl._kimi_compute_view.q_proj.layer.weight
    torch.testing.assert_close(full.view(torch.uint8), torch.cat((bits, bits.bitwise_xor(128))))
    assert full.dtype == dtype and full.tag == "fp8" and casts[0] == dtype
    torch.testing.assert_close(impl.q_proj.weight.view(torch.uint8), bits)


@pytest.mark.parametrize(
    "dim,token_stride,block_stride,last_stride,offset,valid",
    [
        (512, 512, 1536, 1, 0, True),
        (512, 512, 1552, 1, 16, True),
        (64, 64, 208, 1, 16, True),
        (512, 576, 1728, 1, 16, False),
        (64, 576, 1728, 1, 512, False),
        (512, 1024, 3072, 2, 16, False),
    ],
)
def test_fia_cache_reader_contract(dim, token_stride, block_stride, last_stride, offset, valid):
    size = offset + block_stride + 2 * token_stride + (dim - 1) * last_stride + 1
    backing = torch.empty_strided((size,), (1,))
    component = backing.as_strided((2, 3, 1, dim), (block_stride, token_stride, dim, last_stride), offset)
    cache = (component, torch.empty(2, 3, 1, 0))
    if valid:
        backend._validate_fia_cache(cache)
        assert component.storage_offset() == offset
    else:
        with pytest.raises(ValueError, match="shape=.*stride="):
            backend._validate_fia_cache(cache)


def test_upstream_groups_target_and_draft_without_splitting_kv_pool(monkeypatch):
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.worker.gpu import attn_utils

    from vllm_ascend.attention.mla_v1 import AscendKimiComputeMLABackend, AscendMLABackend

    layers = {
        "target": NS(num_heads=16, get_attn_backend=lambda: AscendKimiComputeMLABackend),
        "draft": NS(num_heads=16, get_attn_backend=lambda: AscendMLABackend),
    }
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda config: {})
    monkeypatch.setattr(attn_utils, "add_kv_sharing_layers_to_kv_cache_groups", lambda *args: None)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *args: layers)
    monkeypatch.setattr(attn_utils, "prepare_kernel_block_sizes", lambda config, groups: [128])
    monkeypatch.setattr(attn_utils, "get_num_ubatches", lambda config: 1, raising=False)
    created = []
    monkeypatch.setattr(
        attn_utils.AttentionGroup,
        "create_metadata_builders",
        lambda self, **kw: (created.append(self.layer_names), setattr(self, "metadata_builders", [])),
    )
    monkeypatch.setattr(
        attn_utils.AttentionGroup,
        "get_metadata_builder",
        lambda *args: NS(get_cudagraph_support=lambda *args: AttentionCGSupport.ALWAYS),
    )
    cache = NS(
        kv_cache_groups=[
            NS(layer_names=["target", "draft"], kv_cache_spec=type("Spec", (), {"has_layer_views": True})())
        ]
    )
    groups, _, blocks = attn_utils.init_attn_backend(
        cache, NS(cache_config=NS(kv_sharing_fast_prefill=False), parallel_config=NS()), torch.device("cpu")
    )
    assert len(groups) == 1 and blocks == [128]
    assert created == [["target"], ["draft"]]
    assert groups[0][0].kv_cache_group_id == groups[0][1].kv_cache_group_id


@pytest.mark.parametrize("bias", [False, True])
def test_kv_only_projection_preserves_native_weights_and_output(monkeypatch, bias):
    linear = torch.nn.Linear(8, 12, bias=bias)
    linear.quant_method = NativeMethod()
    original = linear.weight
    monkeypatch.setattr(backend.torch_npu, "npu_format_cast", lambda value, fmt: value)
    monkeypatch.setattr(backend.torch_npu, "get_npu_format", lambda value: 2, raising=False)
    impl = NS(_kimi_compute_view=object(), fused_qkv_a_proj=linear, q_lora_rank=8, kv_lora_rank=3, qk_rope_head_dim=1)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    expected = linear(x)[:, 8:]
    backend.prepare_compute_kv(impl)
    actual = backend._source_kv(impl, x)
    torch.testing.assert_close(actual, expected)
    assert linear.weight is original and linear.weight.shape == (12, 8)
    assert impl._kimi_compute_kv_proj.weight.shape == (4, 8)
    assert impl._kimi_compute_kv_proj.weight.storage_offset() == 0
    assert impl._kimi_compute_kv_proj.output_size_per_partition == 4


def test_mxfp8_output_axis_scales_are_sliced_together(monkeypatch):
    class Scheme:
        def apply(self, layer, x, bias=None):
            return x @ layer.weight

    monkeypatch.setattr(backend, "AscendW8A8MXFP8DynamicLinearMethod", Scheme)
    monkeypatch.setattr(backend.torch_npu, "npu_format_cast", lambda value, fmt: value)
    monkeypatch.setattr(backend.torch_npu, "get_npu_format", lambda value: 2, raising=False)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.arange(96.0).reshape(8, 12))
    layer.weight_scale = torch.nn.Parameter(torch.arange(24.0).reshape(1, 12, 2))
    layer.bias = None
    layer.quant_method = Scheme()
    impl = NS(_kimi_compute_view=object(), fused_qkv_a_proj=layer, q_lora_rank=8, kv_lora_rank=3, qk_rope_head_dim=1)
    backend.prepare_compute_kv(impl)
    projected = impl._kimi_compute_kv_proj
    torch.testing.assert_close(projected.weight, layer.weight[:, 8:])
    torch.testing.assert_close(projected.weight_scale, layer.weight_scale[:, 8:])
    assert layer.weight_scale.shape == (1, 12, 2)


def test_unsupported_scheme_keeps_full_projection(monkeypatch):
    impl = NS(_kimi_compute_view=object(), fused_qkv_a_proj=NS(quant_method=object()))
    backend.prepare_compute_kv(impl)
    assert not hasattr(impl, "_kimi_compute_kv_proj")
    monkeypatch.setattr(backend, "_source_a", lambda impl, x: (None, x + 3))
    torch.testing.assert_close(backend._source_kv(impl, torch.ones(2, 4)), torch.full((2, 4), 4.0))


def test_prepare_fused_prolog_keeps_native_and_profiles_fullhead_weights():
    class View:
        def _process_weights_for_fused(self, dtype):
            assert dtype == torch.bfloat16
            assert self.num_heads == self.num_heads_padded == 96
            assert self.head_padding == 0
            assert self.q_proj is full_query
            self._parameters["fused"] = "full96"
            self.weight_uq_qr = "packed96"

    full_query = object()
    view = View()
    view.q_proj = NS(layer=full_query)
    view.num_heads = 96
    view._parameters = {"native": "original"}
    view._buffers = {"source": "original"}
    impl = NS(
        _kimi_compute_view=view,
        use_flashmla=True,
        enable_mlapo=True,
        _mlapo_uses_native_weights=True,
        _flashmla_writer_buffers=object(),
    )
    backend.prepare_compute_prolog(impl, torch.bfloat16)
    fused = impl._kimi_compute_prolog_view
    assert fused is not view
    assert view._parameters == {"native": "original"}
    assert not hasattr(view, "weight_uq_qr")
    assert fused.weight_uq_qr == "packed96"
    assert fused._flashmla_writer_buffers is impl._flashmla_writer_buffers


def test_fused_prolog_finishes_before_hidden_ag(monkeypatch):
    events = []
    stream = NS(wait_stream=lambda other: None)
    group = NS(world_size=2, rank_in_group=0, device_group=None, _kimi_mla_gate_stream=stream)
    partition = NS(total_tokens=4, capacity=2, world_size=2)
    local = NS(external_flashmla=object(), slot_mapping=torch.tensor([0, 1]))
    metadata = NS(
        compute_decode=NS(partition=partition, local_metadata=local),
        slot_mapping=torch.arange(4),
        decode=NS(cos=None, sin=None),
    )

    def fused(hidden, cache, local_metadata):
        events.append("prolog")
        return object(), None

    def attention(*args, **kwargs):
        events.append("flash")
        return torch.ones(2, 2)

    impl = NS(
        _kimi_compute_view=NS(num_heads=96, _forward_external_flashmla=attention),
        _kimi_compute_prolog_view=NS(mla_preprocess_only_decode=fused),
        kv_lora_rank=512,
        layerwise_kv_cache_hook=None,
        g_proj=lambda x: (torch.zeros(4, 2), None),
        exec_kv_decode=lambda *args: events.append("refresh"),
        o_proj=lambda x, **kwargs: (x, None),
    )
    monkeypatch.setattr(backend, "get_tp_group", lambda: group)
    monkeypatch.setattr(backend, "wait_for_kv_layer_from_connector", lambda *args: None)
    monkeypatch.setattr(backend, "_source_a", lambda impl, x: (None, torch.zeros(4, 576)))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)

    def gather(output, local_hidden, **kwargs):
        events.append("ag")
        output.copy_(local_hidden.repeat(2, 1))
        return NS(wait=lambda: None)

    def exchange(value, partition, group, after_submit):
        events.append("a2a")
        after_submit()
        return torch.ones(4, 2)

    monkeypatch.setattr(backend.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(backend, "dp_to_tp_output", exchange)
    for name in ("record_attention_compute_start", "notify_kv_cache_written", "maybe_save_kv_layer_to_connector"):
        monkeypatch.setattr(backend, name, lambda *args: None)
    output = torch.empty(4, 2)
    backend.forward_compute(impl, "layer", torch.ones(2, 2), torch.zeros(1, 128, 1, 576), metadata, output, 4)
    assert events == ["prolog", "ag", "flash", "a2a", "refresh"]
