# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-request compute partitions over the existing SP token shards."""

from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DecodePartition:
    """A rank's real requests and padded capacity in an existing SP shard."""

    token_start: int
    capacity: int
    total_tokens: int
    valid_tokens: int
    query_len: int
    world_size: int
    rank: int

    @classmethod
    def from_sp_layout(
        cls,
        total_tokens: int,
        local_tokens: int,
        query_lengths: Sequence[int],
        world_size: int,
        rank: int,
        *,
        allow_empty: bool = False,
    ) -> "DecodePartition":
        """Keep uniform positive-length requests inside their original SP token ranges.

        Query lengths describe real requests on the CPU; padding follows them.
        Empty shards require explicit Graph padding support. Zero-KV FIA was
        used for fixed-capacity graph partitions; eager keeps the strict default.
        """
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world_size")
        if local_tokens < 1 or total_tokens != world_size * local_tokens:
            raise ValueError("total_tokens must equal world_size * positive local_tokens")
        if len(query_lengths) == 0:
            raise ValueError("query_lengths must not be empty")
        query_len = query_lengths[0]
        if any(length != query_len for length in query_lengths):
            raise ValueError("query_lengths must be uniform")
        if any(not isinstance(length, Integral) for length in query_lengths) or query_len < 1:
            raise ValueError("uniform query length must be a positive integer")
        query_len = int(query_len)
        if local_tokens % query_len:
            raise ValueError("SP shard capacity must be request-aligned")
        valid_total = len(query_lengths) * query_len
        if valid_total > total_tokens:
            raise ValueError("real query tokens exceed total_tokens")
        counts = tuple(
            max(0, min(local_tokens, valid_total - source_rank * local_tokens)) for source_rank in range(world_size)
        )
        if not allow_empty and any(count == 0 for count in counts):
            raise ValueError("SP layout has an empty rank with no real requests")
        token_start = rank * local_tokens
        return cls(
            token_start=token_start,
            capacity=local_tokens,
            total_tokens=total_tokens,
            valid_tokens=counts[rank],
            query_len=query_len,
            world_size=world_size,
            rank=rank,
        )


def dp_to_tp_output(
    local_output: torch.Tensor,
    partition: DecodePartition,
    process_group: dist.ProcessGroup | None,
    output: torch.Tensor | None = None,
    after_submit: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Exchange [S, H*V] DP outputs into [T, H*V/P] in source-rank order.

    The caller initializes padded rows in local_output. The callback runs after
    the asynchronous collective is submitted; its failure still waits for work.
    """
    world_size, capacity = partition.world_size, partition.capacity
    packed = local_output.ndim == 3
    if packed:
        if tuple(local_output.shape[:2]) != (world_size, capacity) or not local_output.is_contiguous():
            raise ValueError("packed output must be contiguous [world_size, SP capacity, local channels]")
        channels = local_output.shape[2]
    else:
        if local_output.ndim != 2 or local_output.shape[0] != capacity:
            raise ValueError("local_output must have shape [SP capacity, H*V]")
        if local_output.shape[1] % world_size:
            raise ValueError("local_output channels must be divisible by world_size")
        channels = local_output.shape[1] // world_size
    result_shape = (partition.total_tokens, channels)
    if output is not None and (
        tuple(output.shape) != result_shape
        or output.dtype != local_output.dtype
        or output.device != local_output.device
    ):
        raise ValueError("output must match the TP output shape, dtype and device")
    if world_size == 1:
        if packed:
            local_output = local_output.reshape(result_shape)
        if after_submit is not None:
            after_submit()
        if output is None:
            return local_output
        output.copy_(local_output)
        return output

    send = (
        local_output if packed else local_output.reshape(capacity, world_size, channels).permute(1, 0, 2).contiguous()
    )
    receive = torch.empty_like(send)
    work = dist.all_to_all_single(receive, send, group=process_group, async_op=True)
    try:
        if after_submit is not None:
            after_submit()
    finally:
        work.wait()
    if output is None:
        return receive.reshape(result_shape)
    output.copy_(receive.reshape(result_shape))
    return output


def build_compute_decode(metadata, common, world_size: int, rank: int):
    """Attach common geometry to every unsupported decode partition error."""
    batch = getattr(common, "kimi_mla_num_actual_reqs", None)
    total = metadata.num_input_tokens
    capacity = total // world_size if world_size > 0 else 0
    prefix = common.query_start_loc_cpu
    query_len = "unknown"
    if prefix.device.type == "cpu" and prefix.numel() > 1:
        query_len = (prefix[1:2] - prefix[:1]).tolist()[0]
    try:
        return _build_compute_decode(metadata, common, world_size, rank)
    except ValueError as exc:
        raise ValueError(
            f"compute_decode_unsupported: B={batch}, q={query_len}, TP={world_size}, T={total}, S={capacity}: {exc}"
        ) from exc


def _build_compute_decode(metadata, common, world_size: int, rank: int):
    """Build a fixed-capacity FIA view without changing physical block IDs."""
    batch = common.kimi_mla_num_actual_reqs
    if batch is None or batch < 1:
        raise ValueError("compute decode requires the real request count")
    prefix = common.query_start_loc_cpu[: batch + 1]
    if prefix.device.type != "cpu" or prefix.numel() != batch + 1:
        raise ValueError("compute decode requires complete CPU query boundaries")
    total = metadata.num_input_tokens
    capacity = total // world_size if world_size > 0 else 0
    part = DecodePartition.from_sp_layout(
        total,
        capacity,
        (prefix[1:] - prefix[:-1]).tolist(),
        world_size,
        rank,
        allow_empty=common.kimi_mla_full_graph,
    )
    if int(prefix[0]) != 0 or int(prefix[-1]) != metadata.num_actual_tokens:
        raise ValueError("real query boundaries must match the actual token count")
    query_len = part.query_len
    request_capacity = capacity // query_len
    request_start = rank * request_capacity
    request_stop = request_start + request_capacity
    token_start, token_stop = rank * capacity, (rank + 1) * capacity
    # Preserve the runner-owned global table and its possibly padded row stride.
    table = common.block_table_tensor
    if table.shape[0] < total // query_len:
        raise ValueError("global block table is smaller than the fixed request capacity")
    native = metadata.decode
    if native is None:
        raise ValueError("compute decode requires native decode metadata")
    lengths = native.seq_lens_list[:batch] + [0] * (total // query_len - batch)
    if len(lengths) != total // query_len:
        raise ValueError("native decode KV lengths do not cover the real requests")
    slots = metadata.slot_mapping
    if slots.numel() < total:
        if common.slot_mapping.numel() >= total:
            slots = common.slot_mapping[:total]
        elif common.kimi_mla_full_graph:
            raise ValueError("FULL compute graphs require stable full-capacity slot buffers")
        else:
            slots = torch.cat((slots, slots.new_full((total - slots.numel(),), -1)))
    metadata.slot_mapping = slots[:total]
    metadata.slot_mapping[metadata.num_actual_tokens :].fill_(-1)
    cos, sin = native.cos, native.sin
    if cos is not None and cos.shape[0] < total:
        if cos.device.type != "cpu":
            # Native MLA owns the persistent full-length RoPE buffers. Expanding
            # its real-token slice must not allocate a temporary replay input.
            from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla

            cos, sin = get_cos_and_sin_mla(common.positions[:total].long(), use_cache=True)
        else:
            cos = torch.cat((cos, cos.new_zeros((total - cos.shape[0],) + cos.shape[1:])))
            sin = torch.cat((sin, sin.new_zeros((total - sin.shape[0],) + sin.shape[1:])))
    native.cos = cos[:total] if cos is not None else None
    native.sin = sin[:total] if sin is not None else None
    local = copy(metadata)
    local.decode = copy(native)
    local.num_input_tokens = local.num_decode_tokens = capacity
    local.num_actual_tokens = part.valid_tokens
    local.num_decodes = request_capacity
    local.num_prefills = 0
    local.prefill = None
    local.query_lens = [query_len] * request_capacity
    local.query_start_loc = torch.arange(request_capacity + 1, dtype=prefix.dtype, device=table.device) * query_len
    local.slot_mapping = metadata.slot_mapping[token_start:token_stop]
    local.block_tables = table[request_start:request_stop]
    local_lengths = lengths[request_start:request_stop]
    local.seq_lens = torch.tensor(local_lengths, dtype=prefix.dtype)
    local.seq_lens_cpu = local.seq_lens
    local.decode.block_table = local.block_tables
    local.decode.seq_lens = local.seq_lens
    local.decode.seq_lens_list = local_lengths
    local.decode.max_seq_lens = max(local_lengths)
    local.decode.actual_seq_lengths_q = [query_len * (i + 1) for i in range(request_capacity)]
    local.decode.input_positions = common.positions[token_start:token_stop]
    local.decode.cos = native.cos[token_start:token_stop] if native.cos is not None else None
    local.decode.sin = native.sin[token_start:token_stop] if native.sin is not None else None
    return SimpleNamespace(partition=part, local_metadata=local)
