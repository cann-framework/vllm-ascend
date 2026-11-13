from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_ascend.attention.kimi_c8_metadata import build_quant_mla_metadata_kwargs


@dataclass(frozen=True)
class KimiC8GraphQueryPlan:
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_q: torch.Tensor
    block_table: torch.Tensor
    max_seqlen_kv: int
    live_mask: torch.Tensor


def build_c8_graph_query_plan(
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    global_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    interleave_size: int,
    max_model_len: int,
) -> KimiC8GraphQueryPlan:
    """Build fixed-T Q1 rows entirely on device for capture and every replay.

    Inputs must be views of the runner's persistent device input buffers. The
    block table is already expanded to logical 128-token kernel IDs by V2;
    the caller applies static C8 physical-page mapping before quant FA.
    Padded rows retain Q1 shape, but read zero KV and produce ignored LSE.
    """
    if world_size <= 0 or not 0 <= rank < world_size or interleave_size <= 0 or interleave_size % 128:
        raise ValueError("C8 Graph DCP geometry requires a valid rank and 128-aligned interleave")
    if max_model_len <= 0:
        raise ValueError("C8 Graph requires a positive static model-length bound")
    if positions.ndim != 1 or query_start_loc.ndim != 1 or global_seq_lens.ndim != 1:
        raise ValueError("C8 Graph positions, query offsets and sequence lengths must be vectors")
    if global_seq_lens.numel() == 0 or block_table.ndim != 2 or block_table.shape[0] < global_seq_lens.numel():
        raise ValueError("C8 Graph requires one block-table row per request")
    if query_start_loc.numel() != global_seq_lens.numel() + 1:
        raise ValueError("C8 Graph query offsets must include leading zero and one end per request")
    if any(tensor.device != positions.device for tensor in (query_start_loc, global_seq_lens, block_table)):
        raise ValueError("C8 Graph inputs must share the runner device")
    token_count = positions.numel()
    token_rows = torch.arange(token_count, device=positions.device)
    request_rows = torch.searchsorted(query_start_loc[1:].contiguous(), token_rows, right=True)
    safe_rows = request_rows.clamp(max=global_seq_lens.numel() - 1)
    live = (request_rows < global_seq_lens.numel()) & (token_rows < query_start_loc[-1])
    live = live & (global_seq_lens[safe_rows] > 0) & (positions >= 0)
    visible = positions.long() + 1
    cycle = world_size * interleave_size
    local = torch.div(visible, cycle, rounding_mode="floor") * interleave_size
    local = local + (visible.remainder(cycle) - rank * interleave_size).clamp(0, interleave_size)
    lengths = torch.where(live, local, 0).to(torch.int32)
    # This scalar is configuration-derived, never obtained from a device max.
    remainder = max_model_len % cycle
    maximum = max_model_len // cycle * interleave_size + min(
        max(remainder - rank * interleave_size, 0), interleave_size
    )
    return KimiC8GraphQueryPlan(
        cache_seqlens=lengths,
        cu_seqlens_q=torch.arange(token_count + 1, device=positions.device, dtype=torch.int32),
        seqused_q=torch.ones(token_count, device=positions.device, dtype=torch.int32),
        block_table=block_table.index_select(0, safe_rows),
        max_seqlen_kv=max(1, maximum),
        live_mask=live,
    )


def prepare_c8_graph_metadata(plan: KimiC8GraphQueryPlan, metadata_op: Callable, *, num_heads_q: int) -> torch.Tensor:
    """Record metadata generation in the graph; share its result across layers.

    The caller caches this plan/result only within the current model step and
    geometry. No value-based Python cache may skip the recorded device op.
    """
    return metadata_op(
        **build_quant_mla_metadata_kwargs(
            plan.cache_seqlens,
            plan.cu_seqlens_q,
            plan.seqused_q,
            num_heads_q=num_heads_q,
            max_seqlen_q=1,
            max_seqlen_kv=plan.max_seqlen_kv,
        )
    )


class C8MetadataEvents:
    """Load-time stream/events reused by eager steps and ACL Graph capture.

    The first MLA layer starts the producer; the step's shared metadata cache
    keeps both its output and this producer alive for every consuming layer.
    The CPU path is synchronous and implements the identical start/wait API.
    """

    def __init__(self, device: torch.device | str, *, npu_runtime=None):
        self.device = torch.device(device)
        self._metadata: torch.Tensor | None = None
        self._runtime = npu_runtime
        if self._runtime is None and self.device.type in ("npu", "privateuseone"):
            self._runtime = torch.npu
        if self._runtime is None:
            self.stream = self.input_ready = self.output_ready = None
        else:
            self.stream = self._runtime.Stream(device=self.device)
            self.input_ready = self._runtime.Event()
            self.output_ready = self._runtime.Event()

    def start(self, plan: KimiC8GraphQueryPlan, metadata_op: Callable, *, num_heads_q: int) -> torch.Tensor:
        self._metadata = None
        if self._runtime is None:
            self._metadata = prepare_c8_graph_metadata(plan, metadata_op, num_heads_q=num_heads_q)
            return self._metadata
        current = self._runtime.current_stream(device=self.device)
        self.input_ready.record(current)
        self.stream.wait_event(self.input_ready)
        for tensor in (plan.cache_seqlens, plan.cu_seqlens_q, plan.seqused_q):
            tensor.record_stream(self.stream)
        with self._runtime.stream(self.stream):
            self._metadata = prepare_c8_graph_metadata(plan, metadata_op, num_heads_q=num_heads_q)
            self._metadata.record_stream(self.stream)
            self.output_ready.record(self.stream)
        return self._metadata

    def wait(self, expected_metadata: torch.Tensor | None = None) -> torch.Tensor:
        metadata = self._metadata if expected_metadata is None else expected_metadata
        if metadata is None:
            raise RuntimeError("C8 metadata producer has no successful start")
        if self._runtime is not None:
            current = self._runtime.current_stream(device=self.device)
            # A newer start may have advanced the event; waiting for it also
            # covers earlier work on the same stream, but must still return A.
            current.wait_event(self.output_ready)
            metadata.record_stream(current)
        return metadata
