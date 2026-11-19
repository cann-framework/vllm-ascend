# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import AbstractContextManager
from copy import copy
from dataclasses import replace
from typing import Any, cast

import torch
import torch.distributed as dist
from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.distributed import get_dp_group
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream
from vllm.v1.worker.gpu.attn_utils import get_attn_cg_support as _get_attn_cg_support
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp as _dispatch_cg_and_sync_dp
from vllm.v1.worker.ubatch_utils import get_num_ubatches

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    _is_kv_consumer_only,
    _kimi_megamoe_weights_supported,
    get_mc2_tokens_capacity,
    use_cann_megamoe,
)
from vllm_ascend.attention.context_parallel.common_cp import is_pcp_decode_sharding_enabled
from vllm_ascend.core.kv_cache_interface import is_sliding_window_gqa_draft_group
from vllm_ascend.worker.block_table import max_num_blocks_per_req_for_group
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace
from vllm_ascend.worker.v2.async_utils import AscendAsyncOutput
from vllm_ascend.worker.v2.moe_utils import (
    V2MoEPhaseMetadata,
    agree_v2_moe_batch,
    get_v2_moe_batch_state,
)


def _sync_v2_moe_phase(num_tokens, dp_size, dp_rank, allow_ubatching):
    state = get_v2_moe_batch_state()
    if state is None:
        return None
    vllm_config = state.vllm_config
    parallel = vllm_config.parallel_config
    capacity = min(get_mc2_tokens_capacity() or 0, vllm_config.scheduler_config.max_num_batched_tokens)
    eligible = (
        state.allow_megamoe
        and not allow_ubatching
        and parallel.prefill_context_parallel_size == 1
        and not _is_kv_consumer_only(vllm_config)
        and use_cann_megamoe(vllm_config)
        and _kimi_megamoe_weights_supported(vllm_config)
    )
    local = V2MoEPhaseMetadata(
        num_tokens, state.has_real_work, state.batch_contains_decode, state.is_profile, eligible, capacity
    )
    if dp_size > 1:
        metadata = torch.zeros((dp_size, len(local)), dtype=torch.int32, device="cpu")
        metadata[dp_rank] = torch.tensor(local, dtype=torch.int32, device="cpu")
        dist.all_reduce(metadata, group=get_dp_group().cpu_group)
        peers = metadata.tolist()
    else:
        peers = [list(local)]
    agreed = agree_v2_moe_batch(peers, parallel.tensor_parallel_size)
    if agreed.prefill_ready:
        reason = "ready"
    elif agreed.batch_contains_decode:
        reason = "decode_or_unknown_phase"
    elif not all(V2MoEPhaseMetadata(*peer).eligible for peer in peers):
        reason = "ineligible_rank"
    else:
        reason = "token_capacity_or_padding"
    logger.info_once("Kimi V2 MegaMoE phase agreement: %s", reason)
    logger.debug(
        "Kimi V2 MegaMoE phase: rank=%s, local=%s, peers=%s, allow_megamoe=%s, "
        "allow_ubatching=%s, pcp_size=%s, padded_tokens=%s, decision=%s",
        dp_rank,
        local,
        peers,
        state.allow_megamoe,
        allow_ubatching,
        parallel.prefill_context_parallel_size,
        agreed.padded_num_tokens,
        reason,
    )
    return agreed


# TODO: Remove once upstream GPUModelRunner dispatches sharded PCP decode with
# rank-local request counts.
def dispatch_cg_and_sync_dp(
    cudagraph_manager,
    num_reqs,
    num_tokens,
    uniform_token_count,
    dp_size,
    dp_rank,
    max_query_len=None,
    need_eager=False,
    num_active_loras=0,
    parallel_config=None,
    allow_ubatching=False,
    uniform_decode=False,
    dp_sync=None,
    **kwargs,
):
    vllm_config = getattr(cudagraph_manager, "vllm_config", None)
    if uniform_token_count and vllm_config is not None and is_pcp_decode_sharding_enabled(vllm_config):
        # num_tokens is already the largest rank-local count, and each sharded
        # decode request stays whole on one rank.
        num_reqs = num_tokens // uniform_token_count
    state = _sync_v2_moe_phase(num_tokens, dp_size, dp_rank, allow_ubatching)
    prefill_ready = state is not None and state.prefill_ready
    batch_desc, dp_sync = _dispatch_cg_and_sync_dp(
        cudagraph_manager,
        num_reqs,
        num_tokens,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=max_query_len,
        need_eager=need_eager or prefill_ready,
        num_active_loras=num_active_loras,
        parallel_config=parallel_config,
        allow_ubatching=allow_ubatching,
        uniform_decode=uniform_decode,
        dp_sync=dp_sync,
        **kwargs,
    )
    if prefill_ready:
        batch_desc = replace(batch_desc, num_tokens=state.padded_num_tokens)
        if dp_sync is not None:
            dp_sync = replace(
                dp_sync,
                num_tokens_across_dp=torch.full_like(dp_sync.num_tokens_across_dp, state.padded_num_tokens),
            )
    return batch_desc, dp_sync


# Adapted from vLLM ced6857afa GPUModelRunner.initialize_kv_cache.
def _check_attention_cp_compatibility(
    vllm_config,
    target_layer_names,
    replicated_draft_layer_names: frozenset[str] = frozenset(),
) -> None:
    """Upstream check_attention_cp_compatibility, CP assertions scoped.

    Upstream asserts MTP-with-CP-interleave and DCP decode-LSE support on every
    attention layer. A Window-GQA DSpark draft is replicated on every DCP rank
    and runs its attention under a local DCP size of one (the same criterion
    that builds cp_size=1 block tables for its group), so the worker-wide CP
    assertions do not describe it. Keep the full upstream contract for every
    other layer, including MLA drafts that share the worker-wide DCP.
    """
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size <= 1:
        return
    layer_type = cast(Any, AttentionLayerBase)
    layers = get_layers_from_vllm_config(vllm_config, layer_type)
    for layer_name, layer in layers.items():
        check_pcp = target_layer_names is None or layer_name in target_layer_names
        get_attn_backend = getattr(layer, "get_attn_backend", None)
        if pcp_size > 1 and check_pcp and get_attn_backend is not None:
            backend = get_attn_backend()
            assert backend.supports_pcp(), (
                "PCP requires attention backend support, "
                f"but {backend.get_name()} does not support PCP."
            )
        if layer_name in replicated_draft_layer_names:
            continue
        layer_impl = getattr(layer, "impl", None)
        if layer_impl is None:
            continue
        if vllm_config.speculative_config is not None and interleave_size > 1:
            assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                "MTP with cp_kv_cache_interleave_size > 1 is not "
                f"supported in {layer_impl.__class__.__name__}."
            )
        if dcp_size > 1:
            assert layer_impl.need_to_return_lse_for_decode, (
                "Decode Context Parallelism (DCP) requires attention "
                "implementations to return the softmax LSE during decode, "
                f"but {layer_impl.__class__.__name__} does not. "
                "Try a different backend by setting "
                "--attention-backend or disable DCP."
            )


def initialize_kv_cache(
    self,
    kv_cache_config: KVCacheConfig,
    is_profiling: bool = False,
    kv_cache_allocation_context: AbstractContextManager | None = None,
) -> None:
    # GPUWorker finalizes the PD interleave before KV cache initialization.
    self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size
    kv_cache_config = upstream.deepcopy(kv_cache_config)
    self.kv_cache_config = kv_cache_config

    block_table_max_model_len = self.max_model_len
    if self.is_encoder_decoder:
        # Cross-attention block tables need to index encoder tokens, which
        # can exceed the decoder's max_model_len.
        block_table_max_model_len = max(
            block_table_max_model_len,
            self.scheduler_config.max_num_encoder_input_tokens,
            getattr(self.model_config.hf_config, "max_source_positions", 0),
        )

    block_sizes = []
    max_num_blocks_per_group = []
    cp_sizes = []
    cp_ranks = []
    cp_interleaves = []
    slot_mapping_enabled = []
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        spec = kv_cache_group.kv_cache_spec
        block_sizes.append(spec.block_size)
        layer_spec = spec.first_spec if isinstance(spec, upstream.UniformTypeKVCacheSpecs) else spec
        slot_mapping_enabled.append(layer_spec.uses_slot_mapping)
        # Let each cache type account for CP. Attention KV is DCP-sharded,
        # while Mamba/GDN recurrent state is replicated across DCP ranks.
        is_replicated_window = is_sliding_window_gqa_draft_group(kv_cache_group)
        max_num_blocks = max_num_blocks_per_req_for_group(
            spec,
            self.vllm_config,
            block_table_max_model_len,
            kv_cache_group,
        )
        cp_sizes.append(1 if is_replicated_window else self.dcp_size)
        cp_ranks.append(0 if is_replicated_window else self.dcp_rank)
        cp_interleaves.append(1 if is_replicated_window else self.cp_interleave)
        # Preserve each cache type's alignment requirements after applying
        # its topology-aware block-table width.
        max_num_blocks = upstream.get_block_table_width(
            max_num_blocks,
            spec.block_size,
            token_alignment=spec.block_table_token_alignment,
        )
        max_num_blocks_per_group.append(max_num_blocks)

    target_attn_layer_names = None
    if isinstance(self.speculator, upstream.DraftModelSpeculator):
        # Adaptive verification validates target attention separately.
        target_attn_layer_names = {
            layer_name for group in self.kv_cache_config.kv_cache_groups for layer_name in group.layer_names
        } - self.speculator.draft_attn_layer_names
    draft_attn_layer_names = None
    if isinstance(self.speculator, upstream.DraftModelSpeculator):
        draft_attn_layer_names = self.speculator.draft_attn_layer_names
    # Metadata builders select attention kernels that need JIT warmup.
    with self.jit_warmup_registry.activate():
        (
            self.attn_groups,
            attn_cg_support,
            self.kernel_block_sizes,
        ) = upstream.init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
            draft_layer_names=draft_attn_layer_names,
        )
    # init_attn_backend selects each group's builder class from the *current*
    # config, which here is the worker-wide target config (DCP > 1). A
    # replicated Window-GQA DSpark draft never consumes DCP execution: its
    # block tables are built with cp_size=1 below and its speculator keeps a
    # local DCP=1 attention config. Building DCP metadata builders for those
    # layers makes graph capture run the DCP-only decode path (including the
    # DCP CPU seq-lens partition over device tensors) against a topology the
    # draft never uses. Rebuild exactly those groups' builders under the
    # draft's local DCP=1 topology, mirroring the speculator's set_attn.
    replicated_draft_layer_names = frozenset(
        layer_name
        for group in kv_cache_config.kv_cache_groups
        if is_sliding_window_gqa_draft_group(group)
        for layer_name in group.layer_names
    )
    if replicated_draft_layer_names:
        draft_local_config = copy(self.vllm_config)
        draft_local_config.parallel_config = replace(
            self.parallel_config,
            decode_context_parallel_size=1,
            dcp_kv_cache_interleave_size=1,
            cp_kv_cache_interleave_size=1,
        )
        num_metadata_builders = get_num_ubatches(self.parallel_config)
        for groups in self.attn_groups:
            for group in groups:
                if not any(name in replicated_draft_layer_names for name in group.layer_names):
                    continue
                with set_current_vllm_config(draft_local_config):
                    group.create_metadata_builders(
                        vllm_config=draft_local_config,
                        device=self.device,
                        kernel_block_size=self.kernel_block_sizes[group.kv_cache_group_id],
                        num_metadata_builders=num_metadata_builders,
                    )
        attn_cg_support = _get_attn_cg_support(self.attn_groups, self.vllm_config)
    additional_attn_cg_support = self.model_state.get_additional_cg_support()
    attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)
    # The speculator clears the flag at load time when the checkpoint has
    # no confidence head, so it holds the effective value.
    self.adaptive_verification = upstream.maybe_create_adaptive_verification_manager(
        enable_adaptive_verification=getattr(self.speculator, "enable_adaptive_verification", False),
        attn_groups=self.attn_groups,
        attn_cg_support=attn_cg_support,
        req_states=self.req_states,
        query_start_loc=self.input_buffers.query_start_loc,
        num_bonus_tokens=self.model_state.num_new_sampled_tokens_per_step,
        max_total_logits=upstream.get_max_chunk_logits(self.vocab_size),
        vllm_config=self.vllm_config,
        target_layer_names=target_attn_layer_names,
        additional_attn_cg_support=additional_attn_cg_support,
    )

    self.block_tables = upstream.BlockTables(
        block_sizes=block_sizes,
        max_num_reqs=self.max_num_reqs,
        max_num_batched_tokens=self.max_num_tokens,
        max_num_blocks_per_group=max_num_blocks_per_group,
        device=self.device,
        kernel_block_sizes=self.kernel_block_sizes,
        slot_mapping_enabled=slot_mapping_enabled,
        cp_size=self.dcp_size,
        cp_rank=self.dcp_rank,
        cp_interleave=self.cp_interleave,
        cp_sizes=cp_sizes,
        cp_ranks=cp_ranks,
        cp_interleaves=cp_interleaves,
    )
    self.pcp_manager = upstream.pcp.maybe_build_pcp_manager(
        self.vllm_config,
        self.device,
        self.supports_mm_inputs,
        self.block_tables,
        cls=self.pcp_manager_cls,
    )
    self.ubatch_runner = upstream.maybe_build_ubatch_runner(
        self.vllm_config,
        self.device,
        self.model_state,
        self.attn_groups,
        self.kv_cache_config,
        self.max_num_reqs,
    )
    if self.speculator is not None:
        self.speculator.pcp_manager = self.pcp_manager
    upstream.initialize_mamba_ssu_backend(
        self.vllm_config.mamba_config,
        self.kv_cache_config,
        use_replayssm=self.cache_config.use_replayssm,
    )
    piecewise_capture_available = bool(
        upstream.envs.VLLM_USE_BREAKABLE_CUDAGRAPH or upstream.has_compiled_submodule(self.model)
    )
    if self.adaptive_verification is not None:
        self.compilation_config.cudagraph_mode = upstream.resolve_adaptive_cudagraph_mode(
            self.compilation_config.cudagraph_mode,
            piecewise_capture_available=piecewise_capture_available,
        )
    cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
        attn_cg_support.min_cg_support,
        attn_cg_support.min_cg_attn_backend,
        self.decode_query_len,
        use_v2_model_runner=True,
        tensor_parallel_size=self.parallel_config.tensor_parallel_size,
        kv_cache_config=self.kv_cache_config,
        max_num_reqs=self.max_num_reqs,
        is_profiling=is_profiling,
        piecewise_capture_available=piecewise_capture_available,
    )
    self.cudagraph_manager = upstream.ModelCudaGraphManager(
        self.vllm_config,
        self.device,
        cudagraph_mode,
        decode_query_len=self.decode_query_len,
        lora_capture_cases=self.lora_capture_cases,
        varlen_decode=self.adaptive_verification is not None,
        ubatch_runner=self.ubatch_runner,
    )
    if self.cache_config.kv_sharing_fast_prefill and self.pcp_manager is None:
        self.fast_prefill = upstream.FastPrefillHelper(self.cudagraph_manager, self.max_num_tokens)
    # A replicated Window-GQA DSpark draft runs with a local DCP size of one
    # (cp_size=1 block tables above), so the worker-wide CP interleave and
    # DCP LSE assertions must not apply to its layers. Every other
    # configuration keeps the exact upstream check. The layer set was already
    # collected while rebuilding its metadata builders above.
    if replicated_draft_layer_names:
        _check_attention_cp_compatibility(
            self.vllm_config, target_attn_layer_names, replicated_draft_layer_names
        )
    else:
        upstream.check_attention_cp_compatibility(self.vllm_config, target_attn_layer_names)
    if isinstance(self.speculator, upstream.DraftModelSpeculator):
        # HACK(woosuk)
        self.speculator.set_attn(
            self.model_state,
            self.kv_cache_config,
            self.block_tables,
            self.input_buffers,
            self.attn_groups,
        )
    if self.speculator is not None:
        # After set_attn, so the speculator can size its cudagraph mode
        # to its own attention support.
        self.speculator.init_cudagraph_manager(cudagraph_mode)

    # Capture warmup providers that depend on allocated KV-cache strides.
    with self.jit_warmup_registry.activate():
        kv_caches_dict = upstream.init_kv_cache(
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.device,
            self.kernel_block_sizes,
            self.vllm_config,
            kv_cache_allocation_context=kv_cache_allocation_context,
            block_tables=self.block_tables,
        )
    register_sparse_kv_caches = getattr(self, "_register_sparse_kv_caches", None)
    if register_sparse_kv_caches is not None:
        register_sparse_kv_caches(kv_caches_dict)
    self.kv_caches = [
        tensor
        for cache in kv_caches_dict.values()
        for tensor in (cache if isinstance(cache, (tuple, list)) else (cache,))
        if tensor is not None and tensor.device == self.device
    ]
    if is_profiling:
        self.kv_connector = upstream.NO_OP_KV_CONNECTOR
    else:
        self.kv_connector = upstream.get_kv_connector(self.vllm_config, kv_caches_dict)


def _copy_ascend_cache_blocks(kv_caches, num_blocks, copies):
    return copy_kv_cache_blocks_inplace(
        kv_caches,
        num_blocks,
        copies,
        static_c8=getattr(get_ascend_config(), "mla_cache_mode", "C16") == "C8",
    )


upstream.copy_kv_cache_blocks_inplace = _copy_ascend_cache_blocks

upstream.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp

upstream.GPUModelRunner.initialize_kv_cache = initialize_kv_cache

# Replace the constructor used inside GPUModelRunner.sample_tokens, before it
# submits output copies. All sampling, state updates and connector hooks remain
# in the upstream implementation.
upstream.AsyncOutput = AscendAsyncOutput
