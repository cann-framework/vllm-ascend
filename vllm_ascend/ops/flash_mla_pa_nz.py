# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""BF16 Kimi K3 MLA decode operators over the merged PA_NZ latent cache.

This module mirrors the v0.27 ``ops/flash_mla.py`` BF16 subset.  It is only
imported when ``VLLM_ASCEND_ENABLE_FLASH_MLA`` is enabled, so installations
without the optional ``cann_ops_transformer`` / ``cannbot-dsl`` operators keep
working on the native PA_BBND path.

The fused prolog is ``ops.mla_prolog_bf16`` (BF16, no quantization); the
attention operator is ``flash_mla`` with ``layout_kv="PA_NZ"``.  W8A8C8 / FP8
cache operators are intentionally not part of this module.
"""

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module

import torch

PA_NZ_TILE = 16
_MASK_MODE_NONE = 0
_MASK_MODE_CAUSAL = 3


def _get_cann_op(namespace, *names: str) -> Callable:
    """Return the first available operator name from a CANN namespace."""
    for name in names:
        operator = getattr(namespace, name, None)
        if operator is not None:
            return operator
    raise AttributeError(f"none of the operators are available: {', '.join(names)}")


@lru_cache
def _get_pa_nz_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load the 0.27 BF16 FlashMLA decode and metadata operators."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return (
            _get_cann_op(namespace, "flash_mla", "flash_mla_with_kvcache"),
            _get_cann_op(
                namespace,
                "flash_mla_metadata",
                "flash_mla_with_kvcache_metadata",
            ),
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_FLASH_MLA=1 requires the 0.27 PA_NZ "
            "flash_mla + flash_mla_metadata operators from a matching "
            "cann_ops_transformer install."
        ) from exc


@lru_cache
def _get_bf16_mla_prolog_op() -> Callable:
    """Load the Kimi K3 BF16 fused MLA prolog from cannbot-dsl."""
    try:
        module = import_module("ops.mla_prolog_bf16")
        return module.mla_prolog_bf16
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_FLASH_MLA=1 requires mla_prolog_bf16 from "
            "the cannbot-dsl package."
        ) from exc


def pa_nz_cache_view(kv_cache: torch.Tensor, tile: int = PA_NZ_TILE) -> torch.Tensor:
    """View prolog's logical PA storage as FlashMLA's physical PA_NZ layout.

    Prolog writes ``[block_num, block_size, num_kv, cache_dim]`` while
    FlashMLA reads the same storage as
    ``[block_num, num_kv, cache_dim // tile, block_size, tile]``.
    """
    block_num, block_size, num_kv, cache_dim = kv_cache.shape
    if cache_dim % tile:
        raise ValueError(f"PA_NZ cache width {cache_dim} is not aligned to {tile}.")
    return kv_cache.view(
        block_num,
        num_kv,
        cache_dim // tile,
        block_size,
        tile,
    )


def pa_nz_tile_indices(slots: torch.Tensor, block_size: int, tile_count: int) -> torch.Tensor:
    """Flat NZ-tile row indices for each (slot, tile) pair of a merged cache.

    The merged cache is written through its flattened
    ``[block_num * tile_count * block_size, tile]`` NZ view.  For a token at
    ``slot`` in block ``b`` with in-block offset ``s``, tile ``t`` lives at
    flat row ``((b * tile_count + t) * block_size + s)``.
    """
    flat_slots = slots.reshape(-1).to(torch.int64)
    block_ids = torch.div(flat_slots, block_size, rounding_mode="floor")
    offsets = torch.remainder(flat_slots, block_size)
    tile_ids = torch.arange(tile_count, dtype=torch.int64, device=flat_slots.device)
    return (block_ids[:, None] * tile_count * block_size + tile_ids[None, :] * block_size + offsets[:, None]).reshape(
        -1
    )


def prepare_partial_block_tail_metadata(
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_size: int,
    num_requests: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare fixed-shape metadata for clearing partial cache blocks."""
    if num_requests is None:
        num_requests = min(block_table.shape[0], cache_seqlens.numel())
    if num_requests <= 0:
        return block_table.new_empty((0,), dtype=torch.long), torch.empty(
            (0, block_size), dtype=torch.bool, device=cache_seqlens.device
        )

    block_table = block_table[:num_requests]
    cache_seqlens = cache_seqlens[:num_requests].reshape(-1).to(torch.long)
    request_ids = torch.arange(num_requests, device=block_table.device, dtype=torch.long)
    last_columns = torch.div(
        torch.clamp_min(cache_seqlens, 1) - 1,
        block_size,
        rounding_mode="floor",
    )
    last_block_ids = block_table[request_ids, last_columns].to(torch.long)
    valid_tail = torch.remainder(cache_seqlens, block_size)
    valid_tail.masked_fill_(valid_tail == 0, block_size)
    offsets = torch.arange(block_size, device=cache_seqlens.device, dtype=torch.long)
    invalid = offsets[None, :] >= valid_tail[:, None]
    return last_block_ids, invalid


def clear_partial_block_tail(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    num_requests: int | None = None,
    tail_metadata: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Clear unused slots after the last valid token of each request.

    The request and slot dimensions stay fixed for a captured decode graph.
    Only BF16 caches are handled here; the C8 path is out of scope.
    """
    if num_requests is None:
        num_requests = min(block_table.shape[0], cache_seqlens.numel())
    if num_requests <= 0:
        return
    block_size = kv_cache.shape[1]
    if tail_metadata is None:
        tail_metadata = prepare_partial_block_tail_metadata(
            block_table,
            cache_seqlens,
            block_size,
            num_requests=num_requests,
        )
    last_block_ids, invalid = tail_metadata

    last_blocks = kv_cache.index_select(0, last_block_ids)
    last_blocks_nz = pa_nz_cache_view(last_blocks, tile=PA_NZ_TILE)
    mask_shape = (num_requests, 1, 1, block_size, 1)
    invalid_mask = invalid.view(mask_shape).expand_as(last_blocks_nz)
    last_blocks_nz.masked_fill_(invalid_mask, 0)
    kv_cache.index_copy_(0, last_block_ids, last_blocks)


def merged_pa_nz_gather(
    merged_cache: torch.Tensor,
    kv_lora_rank: int,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the history blocks of a BF16 merged PA_NZ cache for paged reads.

    Only the blocks referenced by ``block_table`` are copied.  Stale scheduler
    columns beyond each request's kv length are redirected to an appended
    all-zero block, so paged reads past the request's context see zeros
    instead of stale KV.
    """
    _, block_size, num_kv, cache_dim = merged_cache.shape
    if cache_dim % PA_NZ_TILE:
        raise ValueError(f"PA_NZ cache width {cache_dim} is not aligned to {PA_NZ_TILE}.")
    write_tile_count = cache_dim // PA_NZ_TILE
    unique_blocks, inverse = torch.unique(block_table.reshape(-1), return_inverse=True)
    compact = merged_cache.index_select(0, unique_blocks.to(torch.int64))
    num_unique = unique_blocks.numel()
    device = merged_cache.device
    nope_tiles = kv_lora_rank // PA_NZ_TILE
    rope_tiles = write_tile_count - nope_tiles
    block_offsets = torch.arange(num_unique, device=device) * write_tile_count * block_size
    nope_rows = (block_offsets[:, None] + torch.arange(nope_tiles * block_size, device=device)).reshape(-1)
    rope_rows = (
        block_offsets[:, None] + nope_tiles * block_size + torch.arange(rope_tiles * block_size, device=device)
    ).reshape(-1)
    flat_rows = compact.view(-1, PA_NZ_TILE)
    nope_view = flat_rows.index_select(0, nope_rows).view(num_unique, num_kv, nope_tiles, block_size, PA_NZ_TILE)
    rope_view = flat_rows.index_select(0, rope_rows).view(num_unique, num_kv, rope_tiles, block_size, PA_NZ_TILE)
    remapped_table = inverse.view_as(block_table).to(block_table.dtype)
    valid_cols = (kv_lens.reshape(-1, 1) + block_size - 1) // block_size
    stale_columns = torch.arange(block_table.shape[1], device=device)[None, :] >= valid_cols
    remapped_table.masked_fill_(stale_columns, num_unique)
    nope_view = torch.cat([nope_view, nope_view.new_zeros((1,) + nope_view.shape[1:])], dim=0)
    rope_view = torch.cat([rope_view, rope_view.new_zeros((1,) + rope_view.shape[1:])], dim=0)
    return nope_view, rope_view, remapped_table


def mla_prolog_no_rope(
    hidden_states: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    q_norm_eps: float,
):
    """Run the BF16 fused no-RoPE prolog with a merged PA_NZ cache update.

    ``mla_prolog_bf16`` takes 10 positional arguments.  The merged ``ckv+kr``
    cache is updated in place and the returned query already has its rope
    segment concatenated, ready for FlashMLA.
    """
    prolog_op = _get_bf16_mla_prolog_op()
    return prolog_op(
        hidden_states.contiguous(),
        weight_dq.detach(),
        weight_uq_qr.detach(),
        weight_dkv_kr.detach(),
        weight_uk.detach(),
        q_norm_weight.float().contiguous(),
        kv_norm_weight.float().contiguous(),
        kv_cache,
        cache_index.reshape(-1).to(torch.int64),
        q_norm_eps,
    )


def flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    cu_seqlens_q: torch.Tensor,
    q_len: int,
    head_dim_qk: int,
    head_dim_v: int,
    mask_mode: int,
    seqused_q: torch.Tensor | None = None,
):
    """Build the BF16 FlashMLA metadata tensor for one decode step."""
    _, metadata_op = _get_pa_nz_flash_mla_ops()
    return metadata_op(
        cache_seqlens,
        num_heads_q=num_heads_q,
        num_heads_kv=1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=q_len,
        max_seqlen_kv=-1,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        layout_q="TND",
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q_len: int,
    softmax_scale: float,
    head_dim_v: int,
    attn_mask: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    mask_mode: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Run BF16 FlashMLA against the merged PA_NZ latent cache."""
    if mask_mode is None:
        mask_mode = _MASK_MODE_NONE if q_len == 1 else _MASK_MODE_CAUSAL
    if mask_mode == _MASK_MODE_CAUSAL and attn_mask is None:
        raise ValueError("FlashMLA mask_mode=3 requires attn_mask")
    if metadata is None:
        metadata = flash_mla_metadata(
            cache_seqlens,
            num_heads_q=q.shape[1],
            cu_seqlens_q=cu_seqlens_q,
            q_len=q_len,
            head_dim_qk=q.shape[-1],
            head_dim_v=head_dim_v,
            mask_mode=mask_mode,
            seqused_q=seqused_q,
        )
    attention_op, _ = _get_pa_nz_flash_mla_ops()
    output, lse = attention_op(
        q.contiguous(),
        pa_nz_cache_view(kv_cache),
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        attn_mask=attn_mask,
        metadata=metadata,
        head_dim_v=head_dim_v,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="TND",
        layout_kv="PA_NZ",
        # This operator version only accepts NTD output for TND queries.
        # Callers that need token-major output transpose the leading two dims.
        layout_out="NTD",
        return_softmax_lse=return_softmax_lse,
    )
    if return_softmax_lse:
        return output, lse
    return output, None
