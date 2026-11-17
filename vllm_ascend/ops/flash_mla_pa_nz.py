# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""BF16 PA_NZ Prolog and cache helpers for the native FlashMLA backend.

The optional DSL prolog uses A16C16, PA_NZ and NO_ROPE. Attention and
metadata use the DCP-capable operator pair selected by attention.utils.
"""

from collections.abc import Callable
from functools import lru_cache, partial
from importlib import import_module

import torch

PA_NZ_TILE = 16


@lru_cache
def _get_bf16_mla_prolog_op() -> Callable:
    """Load and specialize the current cannbot-dsl BF16 prolog once."""
    try:
        module = import_module("ops.mla_prolog_v4")
        return partial(
            module.mla_prolog_v2,
            qkv_quant_mode=module.QkvQuantMode.A16C16,
            cache_mode=module.CacheMode.PA_NZ,
            rope_mode=module.RopeMode.NO_ROPE,
            return_qr=False,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_PA_NZ=1 requires ops.mla_prolog_v4.mla_prolog_v2 "
            "with A16C16, PA_NZ and NO_ROPE support from a matching cannbot-dsl install."
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


def gather_pa_nz_chunk(
    merged_cache: torch.Tensor,
    kv_lora_rank: int,
    block_table: torch.Tensor,
    chunk_seq_lens: torch.Tensor,
    starts: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one bounded prefill history chunk in token order.

    DCP supplies padded local chunk lengths; its existing KV reorganization
    discards padding after the collective. Never dereference stale page IDs
    or columns outside the table for those padded rows. No unique/nonzero or
    device-to-host shape discovery is needed.
    """
    pages, block_size, num_kv, cache_dim = merged_cache.shape
    if num_kv != 1 or cache_dim % PA_NZ_TILE:
        raise ValueError("PA_NZ chunk gather requires one KV head and a tile-aligned cache")
    lengths = chunk_seq_lens.reshape(-1).long()
    boundaries = lengths.cumsum(0) - lengths
    requests = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=merged_cache.device),
        lengths,
        output_size=num_tokens,
    )
    positions = (
        torch.arange(num_tokens, device=merged_cache.device) - boundaries[requests] + starts.reshape(-1)[requests]
    )
    columns = torch.div(positions, block_size, rounding_mode="floor")
    valid = (positions >= 0) & (columns < block_table.shape[1])
    block_ids = block_table[requests, columns.clamp(0, block_table.shape[1] - 1).long()].long()
    valid &= (block_ids >= 0) & (block_ids < pages)
    slots = torch.where(valid, block_ids * block_size + positions.remainder(block_size), 0)
    indices = pa_nz_tile_indices(slots, block_size, cache_dim // PA_NZ_TILE)
    rows = merged_cache.view(-1, PA_NZ_TILE).index_select(0, indices)
    values = rows.view(num_tokens, num_kv, cache_dim).masked_fill(~valid[:, None, None], 0)
    latent, auxiliary = values.split((kv_lora_rank, cache_dim - kv_lora_rank), dim=-1)
    return latent.contiguous(), auxiliary.contiguous()


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
    """Return ``(q, descale_q, qr)`` and update the PA_NZ cache in place.

    The current ``mla_prolog_v2`` takes nine positional tensors and keyword
    attributes. NZ projection weights retain logical ``[K, O]`` shapes;
    Q-B and W_UK contain only this TP rank's unpadded heads, so the operator
    infers and computes the local head count without a separate TP argument.
    In A16C16/NO_ROPE mode, q already contains the auxiliary 64 dimensions;
    query quantization and the separate QR output are disabled.
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
        norm_eps=q_norm_eps,
    )
