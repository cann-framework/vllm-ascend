from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KimiC8DCPQueryPlan:
    """One causal Q1 row per input query, using the resident rank-local KV."""

    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_q: torch.Tensor
    block_table: torch.Tensor
    max_seqlen_kv: int
    host_cache_seqlens: tuple[int, ...]


def _host_int_vector(value: Sequence[int] | torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise ValueError(f"{name} must use the existing host metadata, without a device synchronization")
    result = torch.as_tensor(value, dtype=torch.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def build_c8_dcp_query_plan(
    global_seqlens: Sequence[int] | torch.Tensor,
    cu_seqlens_q: Sequence[int] | torch.Tensor,
    block_table: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    interleave_size: int,
    num_tokens: int | None = None,
) -> KimiC8DCPQueryPlan:
    """Expand causal decode into Q1 rows for any 128-aligned DCP interleave.

    QFMLA's native DCP metadata accepts only interleave 128. For a larger
    interleave, query i can still read exactly its visible prefix on this rank:
    S = global_len - query_len + i + 1, and local_len = floor(S / (W*C))*C
    + clamp(S % (W*C) - rank*C, 0, C). Each Q1 row uses mask_mode=0 and the
    original request's kernel block table. No KV data is gathered or copied.

    Lengths come from the runner's host metadata. Returned tensors live on the
    block table's device; max_seqlen_kv stays a host scalar for the op API.
    """
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("DCP world_size/rank is invalid")
    if interleave_size <= 0 or interleave_size % 128:
        raise ValueError("DCP interleave must contain whole 128-token kernel blocks")
    lengths = _host_int_vector(global_seqlens, "global_seqlens")
    cu = _host_int_vector(cu_seqlens_q, "cu_seqlens_q")
    if cu.numel() != lengths.numel() + 1 or cu[0] != 0:
        raise ValueError("cu_seqlens_q must contain B+1 offsets with a leading zero")
    query_lens = cu[1:] - cu[:-1]
    if bool((query_lens < 0).any()):
        raise ValueError("C8 query offsets must be nondecreasing")
    if bool((lengths < query_lens).any()):
        raise ValueError("global sequence lengths must cover their query rows")
    if block_table.ndim != 2 or block_table.shape[0] < lengths.numel():
        raise ValueError("block_table must contain one row per request")
    token_to_req = torch.repeat_interleave(torch.arange(lengths.numel()), query_lens)
    token_count = token_to_req.numel()
    query_offset = torch.arange(token_count) - cu[token_to_req]
    visible_global = lengths[token_to_req] - query_lens[token_to_req] + query_offset + 1
    cycle = world_size * interleave_size
    local_lens = torch.div(visible_global, cycle, rounding_mode="floor") * interleave_size + (
        visible_global.remainder(cycle) - rank * interleave_size
    ).clamp(0, interleave_size)
    expanded_table = block_table.index_select(0, token_to_req.to(block_table.device))
    if num_tokens is not None:
        if num_tokens < token_count:
            raise ValueError("C8 query offsets exceed the projected token count")
        padding = num_tokens - token_count
        if padding:
            if block_table.shape[0] == 0:
                raise ValueError("C8 padded queries require a block-table row")
            local_lens = torch.cat((local_lens, torch.zeros(padding, dtype=local_lens.dtype)))
            expanded_table = torch.cat((expanded_table, block_table[:1].expand(padding, -1)))
            token_count = num_tokens
    device = block_table.device
    return KimiC8DCPQueryPlan(
        cache_seqlens=local_lens.to(device=device, dtype=torch.int32),
        cu_seqlens_q=torch.arange(token_count + 1, device=device, dtype=torch.int32),
        seqused_q=torch.ones(token_count, device=device, dtype=torch.int32),
        block_table=expanded_table,
        max_seqlen_kv=max(1, int(local_lens.max())) if token_count else 1,
        host_cache_seqlens=tuple(local_lens.tolist()),
    )
