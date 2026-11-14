from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.kimi_c8_static import FP8_E4M3_MAX


@triton.jit
def _scatter_pa_nz_bytes_kernel(raw, payload, slots, NUM_BYTES: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    active = index < NUM_BYTES
    token, feature = index // WIDTH, index % WIDTH
    slot = tl.load(slots + token, mask=active, other=-1).to(tl.int64)
    safe_slot = tl.maximum(slot, 0)
    offset = safe_slot // 128 * (128 * WIDTH) + feature // 32 * 4096 + safe_slot % 128 * 32 + feature % 32
    value = tl.load(payload + index, mask=active, other=0)
    tl.store(raw + offset, value, mask=active & (slot >= 0))


def _scatter_pa_nz_bytes(raw: torch.Tensor, payload: torch.Tensor, slots: torch.Tensor) -> None:
    count, width = payload.shape
    _scatter_pa_nz_bytes_kernel[(triton.cdiv(count * width, 256),)](
        raw,
        payload,
        slots,
        NUM_BYTES=count * width,
        WIDTH=width,
        TILE=256,
    )


def write_static_fp8_cache_graph(merged: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, k_descale) -> None:
    """Fixed-shape quantization plus masked byte scatter for ACL Graph.

    k_descale is a loaded, validated positive scalar. Slot mapping has already
    converted logical manager pages into aligned physical pages. Every token
    retains its row, including padded negative slots; no dynamic nonzero/index
    tensor or host read is introduced during capture/replay.
    """
    if merged.ndim != 2 or merged.shape[1] != 576 or slots.numel() != merged.shape[0]:
        raise ValueError("C8 Graph writer requires [T,576] and one slot per token")
    if cache.ndim != 5 or tuple(cache.shape[1:]) != (1, 18, 128, 32) or not cache.is_contiguous():
        raise ValueError("C8 Graph writer requires contiguous physical PA_NZ kernel pages")
    if cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("C8 Graph writer requires E4M3 byte storage")
    scale = torch.as_tensor(k_descale, device=merged.device, dtype=torch.float32)
    if scale.numel() != 1:
        raise ValueError("C8 Graph writer requires one prevalidated static KV descale")
    payload = (merged.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    if merged.shape[0]:
        _scatter_pa_nz_bytes(cache.view(torch.uint8).view(-1), payload.contiguous(), slots.reshape(-1).to(torch.int64))
