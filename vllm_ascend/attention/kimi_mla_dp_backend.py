# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi target MLA compute over the existing SP shards and native KV cache."""

from copy import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.compilation import backends as compilation_backends
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.kimi_mla_dp import DecodePartition, dp_to_tp_output
from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND
from vllm_ascend.weight_switch import WeightSwitchConfig


def compute_enabled(impl_or_config) -> bool:
    if hasattr(impl_or_config, "kimi_compute_enabled"):
        return impl_or_config.kimi_compute_enabled
    if compilation_backends.model_tag != "backbone":
        return False
    config = getattr(impl_or_config, "vllm_config", impl_or_config)
    if not getattr(get_ascend_config(), "enable_kimi_mla_decode_dp", False):
        return False
    model = config.model_config
    text = getattr(model, "hf_text_config", getattr(model, "hf_config", None))
    model_type = getattr(text, "model_type", "").lower()
    architectures = getattr(getattr(model, "hf_config", text), "architectures", None) or ()
    is_kimi = "kimi" in model_type or any("KimiK3" in name for name in architectures)
    transfer = config.kv_transfer_config
    return bool(
        is_kimi
        and transfer is not None
        and transfer.is_kv_consumer
        and not transfer.is_kv_producer
        and not getattr(impl_or_config, "is_draft_model", False)
        and getattr(model, "runner_type", None) != "draft"
    )


class FullHeadQuery:
    """Run a private full-weight query linear without TP output communication."""

    def __init__(self, layer):
        self.layer = layer

    def as_linear(self, x):
        return self.layer.quant_method.apply(self.layer, x, bias=self.layer.bias), None

    __call__ = as_linear


def _gather_tensor(group, tensor, dim):
    native = torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_ND).contiguous()
    byte_view = native.is_floating_point() and native.element_size() == 1
    payload = native.view(torch.uint8) if byte_view else native
    full = group.all_gather(payload, dim=dim)
    return full.view(native.dtype) if byte_view else full


def _set_tensor(layer, name, value, original):
    if isinstance(original, torch.nn.Parameter):
        value = torch.nn.Parameter(value, requires_grad=False)
        value.__dict__.update(original.__dict__)
    setattr(layer, name, value)


@torch.no_grad()
def prepare_compute_weights(impl) -> None:
    """Gather native projection shards once, leaving original parameters intact."""
    if not compute_enabled(impl) or getattr(impl, "_kimi_compute_view", None) is not None:
        return
    if getattr(impl, "g_proj", None) is None:
        raise ValueError("Kimi MLA compute requires an output gate")
    group = get_tp_group()
    original = impl.q_proj
    method = original.quant_method
    scheme = getattr(method, "quant_method", method)
    if isinstance(method, UnquantizedLinearMethod):
        gathers = [("weight", getattr(original.weight, "output_dim", 0))]
        repeats = []
    else:
        if not getattr(scheme, "supports_weight_switch", False):
            raise ValueError("Kimi full query requires weight-switch support")
        resolve = getattr(scheme, "_get_weight_switch_specs", None)
        if resolve is None:
            gather_specs = scheme.weight_switch_output_gather_specs
            repeat_specs = scheme.weight_switch_output_repeat_specs
        else:
            gather_specs, repeat_specs, _ = resolve(original, WeightSwitchConfig.from_group(group, shard_axis="output"))
        gathers = [(spec.attr_name, spec.gather_dim) for spec in gather_specs]
        repeats = [(spec.attr_name, spec.repeat_dim) for spec in repeat_specs]
        if not gathers:
            raise ValueError("Kimi full query requires output weight-switch gather specs")
    if original.bias is not None and not any(name == "bias" for name, _ in gathers):
        gathers.append(("bias", 0))
    full_query = copy(original)
    full_query._parameters = original._parameters.copy()
    full_query._buffers = original._buffers.copy()
    for name, dim in gathers:
        tensor = getattr(original, name)
        _set_tensor(full_query, name, _gather_tensor(group, tensor.detach(), dim), tensor)
    for name, dim in repeats:
        tensor = getattr(original, name)
        sizes = [1] * tensor.ndim
        sizes[dim] = group.world_size
        _set_tensor(full_query, name, tensor.detach().repeat(*sizes), tensor)
    full_query.output_size_per_partition = original.output_size_per_partition * group.world_size
    if hasattr(original, "output_partition_sizes"):
        full_query.output_partition_sizes = [size * group.world_size for size in original.output_partition_sizes]
    view = copy(impl)
    view.num_heads = impl.num_heads * group.world_size
    view.num_heads_padded = 1 << (view.num_heads - 1).bit_length()
    view.head_padding = view.num_heads_padded - view.num_heads
    view.num_queries_per_kv = view.num_heads // view.num_kv_heads
    view.mlapo_num_heads = view.num_heads
    view.W_UK_T = _gather_tensor(group, impl.W_UK_T, 0)
    view.W_UV = _gather_tensor(group, impl.W_UV, 0)
    view.mlapo_W_UK_T = view.W_UK_T
    view.q_proj = FullHeadQuery(full_query)
    if not hasattr(group, "_kimi_mla_gate_stream"):
        group._kimi_mla_gate_stream = torch.npu.Stream()
    impl._kimi_compute_view = view


@torch.no_grad()
def prepare_compute_prolog(impl, act_dtype):
    view = getattr(impl, "_kimi_compute_view", None)
    if view is None or not impl.use_flashmla or not impl.enable_mlapo:
        return
    fused = copy(view)
    if hasattr(fused, "_parameters"):
        fused._parameters = fused._parameters.copy()
    if hasattr(fused, "_buffers"):
        fused._buffers = fused._buffers.copy()
    fused.q_proj = view.q_proj.layer
    fused.head_padding = 0
    fused.num_heads_padded = fused.num_heads
    fused.enable_mlapo = impl.enable_mlapo
    fused._mlapo_uses_native_weights = impl._mlapo_uses_native_weights
    fused._process_weights_for_fused(act_dtype)
    fused._flashmla_writer_buffers = impl._flashmla_writer_buffers
    impl._kimi_compute_prolog_view = fused


@torch.no_grad()
def prepare_compute_kv(impl):
    if getattr(impl, "_kimi_compute_view", None) is None:
        return
    original = impl.fused_qkv_a_proj
    method = original.quant_method
    scheme = getattr(method, "quant_method", method)
    if isinstance(method, UnquantizedLinearMethod):
        tensors = [("weight", 0)]
    elif type(scheme) is AscendW8A8MXFP8DynamicLinearMethod:
        if original.weight.ndim != 2 or original.weight_scale.ndim != 3:
            return
        tensors = [("weight", 1), ("weight_scale", 1)]
    else:
        return
    width = impl.kv_lora_rank + impl.qk_rope_head_dim
    start = impl.q_lora_rank
    for name, axis in tensors:
        if getattr(original, name).shape[axis] != start + width:
            return
    if original.bias is not None:
        tensors.append(("bias", 0))
    projected = copy(original)
    projected._parameters = original._parameters.copy()
    projected._buffers = original._buffers.copy()
    for name, axis in tensors:
        tensor = getattr(original, name)
        values = torch_npu.npu_format_cast(tensor.detach(), ACL_FORMAT_FRACTAL_ND)
        values = values.narrow(axis, start, width).clone(memory_format=torch.contiguous_format)
        if name == "weight":
            values = torch_npu.npu_format_cast(values, torch_npu.get_npu_format(tensor))
        _set_tensor(projected, name, values, tensor)
    projected.output_size = projected.output_size_per_partition = width
    if hasattr(projected, "output_partition_sizes"):
        projected.output_partition_sizes = [width]
    impl._kimi_compute_kv_proj = projected


def _source_kv(impl, hidden):
    projected = getattr(impl, "_kimi_compute_kv_proj", None)
    if projected is None:
        return _source_a(impl, hidden)[1]
    return projected.quant_method.apply(projected, hidden, bias=projected.bias)


def _source_a(impl, hidden):
    projected = impl.fused_qkv_a_proj(hidden)[0]
    return projected.split([impl.q_lora_rank, impl.kv_lora_rank + impl.qk_rope_head_dim], dim=-1)


def _zero_padding(value, slots):
    mask = (slots[: value.shape[0]] >= 0).unsqueeze(-1)
    return torch.where(mask, value, 0)


def _validate_fia_cache(cache):
    for component in cache:
        if component.numel() and (component.stride(-1) != 1 or component.stride(1) != component.shape[-1]):
            raise ValueError(
                f"Kimi FIA requires dense token rows: shape={tuple(component.shape)}, stride={component.stride()}"
            )


def forward_compute(impl, layer_name, local_hidden, cache, metadata, output, global_tokens):
    """Run local full-head FA while refreshing the original TP cache and gate."""
    plan = metadata.compute_decode
    partition, local_metadata = plan.partition, plan.local_metadata
    if global_tokens != partition.total_tokens or local_hidden.shape[0] != partition.capacity:
        raise ValueError("Kimi compute hidden shape must match the SP partition")
    view = impl._kimi_compute_view
    group = get_tp_group()
    fused_cache = cache if isinstance(cache, torch.Tensor) else None
    flash = local_metadata.external_flashmla
    if flash is not None:
        view = copy(view)
        view.head_padding = 0
        view.num_heads_padded = view.num_heads
    if fused_cache is not None:
        cache = (fused_cache[..., : impl.kv_lora_rank], fused_cache[..., impl.kv_lora_rank :])
    if flash is None:
        _validate_fia_cache(cache)
    elif fused_cache is None or fused_cache.shape[-1] != 576:
        raise ValueError("Kimi FlashMLA compute requires the original token-fused 576-lane KV cache")
    fused_view = getattr(impl, "_kimi_compute_prolog_view", None) if flash is not None else None
    if fused_view is None:
        q_c, local_kv = _source_a(impl, local_hidden)
        q_c = impl.q_a_layernorm(q_c)
    wait_for_kv_layer_from_connector(layer_name)
    if impl.layerwise_kv_cache_hook is not None:
        impl.layerwise_kv_cache_hook.wait_for_layer(layer_name)
    if fused_view is not None:
        preprocess, _ = fused_view.mla_preprocess_only_decode(local_hidden, cache, local_metadata)
    else:
        preprocess = view.mla_preprocess_decode(q_c, local_kv.contiguous(), cache, local_metadata)
    global_hidden = local_hidden.new_empty((global_tokens, local_hidden.shape[-1]))
    work = None
    if partition.world_size > 1:
        work = dist.all_gather_into_tensor(
            global_hidden, local_hidden.contiguous(), group=group.device_group, async_op=True
        )
    else:
        global_hidden.copy_(local_hidden)
    gather_done = work is None
    gate = None
    gate_started = False
    main_stream = torch.npu.current_stream()
    gate_stream = group._kimi_mla_gate_stream
    global_slots = metadata.slot_mapping[:global_tokens]

    def wait_hidden():
        nonlocal gather_done
        if not gather_done:
            work.wait()
            gather_done = True

    def refresh_and_gate():
        nonlocal gate, gate_started
        wait_hidden()
        gate_stream.wait_stream(main_stream)
        global_hidden.record_stream(gate_stream)
        gate_started = True
        with torch.npu.stream(gate_stream):
            gate = _zero_padding(torch.sigmoid(impl.g_proj(global_hidden)[0]), global_slots)
        full_kv = _source_kv(impl, global_hidden)
        impl.exec_kv_decode(
            full_kv.contiguous(),
            metadata.decode.cos,
            metadata.decode.sin,
            cache,
            metadata.slot_mapping[:global_tokens],
        )

    try:
        if flash is not None:
            local_attention = view._forward_external_flashmla(
                preprocess,
                fused_cache,
                local_metadata,
                defer_kv_ready_fence=True,
                output_partitions=partition.world_size,
            )
        else:
            local_attention = view._forward_decode(
                preprocess,
                cache[0].shape[1],
                local_metadata,
                graph_num_tokens=global_tokens,
                defer_kv_ready_fence=True,
            )
        if local_attention.ndim == 3:
            live = (local_metadata.slot_mapping[: partition.capacity] >= 0).view(1, partition.capacity, 1)
            local_attention = torch.where(live, local_attention, 0)
        else:
            local_attention = _zero_padding(local_attention, local_metadata.slot_mapping[: partition.capacity])
        tp_attention = dp_to_tp_output(local_attention, partition, group.device_group, after_submit=refresh_and_gate)
    finally:
        wait_hidden()
        if gate_started:
            main_stream.wait_stream(gate_stream)
            if gate is not None:
                gate.record_stream(main_stream)
    tp_attention.mul_(gate)
    output.copy_(impl.o_proj(tp_attention, is_prefill=False)[0])
    record_attention_compute_start()
    notify_kv_cache_written(layer_name)
    maybe_save_kv_layer_to_connector(layer_name, [fused_cache] if fused_cache is not None else list(cache))
    return output


def _profile_projection(layer, hidden, live):
    method = layer.quant_method
    scheme = getattr(method, "quant_method", method)
    prepare = getattr(scheme, "prepare_input_for_overlap", None)
    prepared = prepare(layer, hidden) if prepare is not None else None
    if prepared is not None:
        live.append(prepared)
        hidden = prepared
    value = method.apply(layer, hidden, bias=layer.bias)
    live.append(value)
    return value


def profile_compute(impl, local_hidden, global_tokens):
    """Exercise bounded projection/communication peaks without cache or FIA."""
    group, view = get_tp_group(), copy(impl._kimi_compute_view)
    if global_tokens < group.world_size:
        return
    partition = DecodePartition.from_sp_layout(
        global_tokens, local_hidden.shape[0], [1] * global_tokens, group.world_size, group.rank_in_group
    )
    live = [local_hidden]
    projected = _profile_projection(impl.fused_qkv_a_proj, local_hidden, live)
    q_c = impl.q_a_layernorm(projected[:, : impl.q_lora_rank])
    local_kv = impl.kv_a_layernorm(projected[:, impl.q_lora_rank : impl.q_lora_rank + impl.kv_lora_rank])
    query_layer = view.q_proj.layer
    view.q_proj = lambda value: (_profile_projection(query_layer, value, live), None)
    q_nope, q_pe = view._q_proj_and_k_up_proj(q_c)
    live.extend((q_c, local_kv, q_nope, q_pe))
    if view.head_padding:
        live.extend(F.pad(value, (0, 0, 0, view.head_padding)) for value in (q_nope, q_pe))
    global_hidden = local_hidden.new_empty((global_tokens, local_hidden.shape[-1]))
    work = (
        dist.all_gather_into_tensor(global_hidden, local_hidden.contiguous(), group=group.device_group, async_op=True)
        if group.world_size > 1
        else None
    )
    if work is None:
        global_hidden.copy_(local_hidden)
    main, side = torch.npu.current_stream(), group._kimi_mla_gate_stream
    gate, gate_started = None, False

    def refresh_and_gate():
        nonlocal gate, gate_started
        if work is not None:
            work.wait()
        side.wait_stream(main)
        global_hidden.record_stream(side)
        gate_started = True
        with torch.npu.stream(side):
            gate = torch.sigmoid(_profile_projection(impl.g_proj, global_hidden, live))
        full = _profile_projection(impl.fused_qkv_a_proj, global_hidden, live)
        live.append(impl.kv_a_layernorm(full[:, impl.q_lora_rank : impl.q_lora_rank + impl.kv_lora_rank]))

    try:
        latent = local_hidden.new_zeros((view.num_heads_padded, partition.capacity, view.kv_lora_rank))
        local_attention = view._v_up_proj(latent[: view.num_heads])
        live.extend((global_hidden, latent, local_attention))
        tp_attention = dp_to_tp_output(local_attention, partition, group.device_group, after_submit=refresh_and_gate)
    finally:
        if work is not None:
            work.wait()
        if gate_started:
            main.wait_stream(side)
            if gate is not None:
                gate.record_stream(main)
    tp_attention.mul_(gate)
    return impl.o_proj(tp_attention, is_prefill=False)[0]
