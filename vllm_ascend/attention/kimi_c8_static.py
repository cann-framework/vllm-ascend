from __future__ import annotations

import torch

FP8_E4M3_MAX = 448.0
PA_NZ_TILE = 32
PA_NZ_BLOCK_SIZE = 128
K3_MLA_CACHE_WIDTH = 576


def quantize_static_fp8(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    s = torch.as_tensor(scale, device=x.device, dtype=torch.float32)
    if s.numel() != 1 or not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
        raise ValueError("static scale must be one finite positive scalar")
    return (x.float() / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def dequantize_static_fp8(x: torch.Tensor, scale: torch.Tensor | float, dtype=torch.bfloat16) -> torch.Tensor:
    s = torch.as_tensor(scale, device=x.device, dtype=torch.float32)
    if s.numel() != 1 or not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
        raise ValueError("static scale must be one finite positive scalar")
    return x.view(torch.float8_e4m3fn).float().mul(s).to(dtype)


def pa_nz_indices(slots: torch.Tensor, block_size: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    if width <= 0 or width % PA_NZ_TILE:
        raise ValueError("width must be a positive multiple of 32")
    _kernel_blocks_per_manager(block_size)
    block_size = PA_NZ_BLOCK_SIZE
    flat = slots.reshape(-1).to(torch.int64)
    valid = flat >= 0
    slot = flat[valid]
    tiles = width // PA_NZ_TILE
    indices = (
        (slot // block_size)[:, None] * tiles * block_size
        + torch.arange(tiles, device=slot.device)[None, :] * block_size
        + slot.remainder(block_size)[:, None]
    ).reshape(-1)
    return valid, indices


def write_static_fp8_cache(merged: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, scale) -> None:
    if merged.ndim != 2 or cache.ndim != 5 or cache.dtype != torch.uint8:
        raise ValueError("merged/cache layout is invalid")
    block_size, tiles = cache.shape[3], cache.shape[2]
    if block_size != PA_NZ_BLOCK_SIZE:
        raise ValueError("C8 PA_NZ cache requires 128-token physical blocks")
    if cache.shape[1] != 1 or cache.shape[4] != PA_NZ_TILE or merged.shape[1] != tiles * PA_NZ_TILE:
        raise ValueError("cache and merged widths do not match PA_NZ")
    if slots.numel() != merged.shape[0]:
        raise ValueError("C8 writer requires one slot per input token")
    if bool((slots >= cache.shape[0] * block_size).any()):
        raise ValueError("C8 writer slot exceeds physical cache capacity")
    valid, indices = pa_nz_indices(slots, block_size, merged.shape[1])
    rows = quantize_static_fp8(merged[valid], scale).reshape(-1, PA_NZ_TILE)
    cache.view(torch.uint8).view(-1, PA_NZ_TILE).index_copy_(0, indices, rows)


def pa_nz_cache_view(
    raw: torch.Tensor,
    num_blocks: int,
    block_size: int,
    width: int = K3_MLA_CACHE_WIDTH,
    *,
    page_stride_bytes: int | None = None,
) -> torch.Tensor:
    """Expose manager pages as consecutive 128-token PA_NZ kernel blocks."""
    if raw.dtype != torch.uint8 or raw.ndim != 1 or raw.stride(0) != 1:
        raise ValueError("raw PA_NZ backing must be a contiguous uint8 vector")
    if width <= 0 or width % PA_NZ_TILE or num_blocks <= 0:
        raise ValueError("invalid PA_NZ geometry")
    page_stride_bytes = block_size * width if page_stride_bytes is None else page_stride_bytes
    ratio = pa_nz_physical_blocks_per_page(block_size, page_stride_bytes, width)
    required = num_blocks * page_stride_bytes
    if raw.numel() < required:
        raise ValueError("raw backing is smaller than PA_NZ payload")
    return raw[:required].view(num_blocks * ratio, 1, width // PA_NZ_TILE, PA_NZ_BLOCK_SIZE, PA_NZ_TILE)


def _kernel_blocks_per_manager(block_size: int) -> int:
    if block_size <= 0 or block_size % PA_NZ_BLOCK_SIZE:
        raise ValueError("C8 manager block size must be a positive multiple of 128")
    return block_size // PA_NZ_BLOCK_SIZE


def pa_nz_prolog_view(cache: torch.Tensor) -> torch.Tensor:
    """Give Prolog its logical PA shape while preserving physical NZ bytes."""
    if (
        cache.ndim != 5
        or tuple(cache.shape[1:]) != (1, K3_MLA_CACHE_WIDTH // PA_NZ_TILE, PA_NZ_BLOCK_SIZE, PA_NZ_TILE)
        or cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or not cache.is_contiguous()
    ):
        raise ValueError("C8 Prolog requires contiguous [blocks,1,18,128,32] PA_NZ cache")
    return cache.view(torch.float8_e4m3fn).view(cache.shape[0], PA_NZ_BLOCK_SIZE, 1, K3_MLA_CACHE_WIDTH)


def read_static_fp8_cache(
    cache: torch.Tensor,
    kernel_block_table: torch.Tensor,
    seq_lens: torch.Tensor | list[int],
    scale: torch.Tensor | float,
    *,
    starts: torch.Tensor | list[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather eager prefill history as request-ordered [sum(lengths),1,576].

    The table contains physical 128-token kernel IDs, not manager IDs. Each length is
    the requested chunk length; starts are absolute positions in each request.
    Only the requested rows are read, so unused negative/padded table entries
    are harmless. Physical NZ tiles are gathered before merging feature lanes.
    """
    pa_nz_prolog_view(cache)
    if kernel_block_table.ndim != 2 or kernel_block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("C8 history block table must be a rank-2 integer tensor")
    # Keep scheduler-owned host lengths/starts on CPU for validation and the
    # output size. Copying them to NPU before sum().item() stalls every layer.
    lengths = torch.as_tensor(seq_lens, dtype=torch.int64)
    offsets = torch.zeros_like(lengths) if starts is None else torch.as_tensor(starts, dtype=torch.int64)
    if lengths.ndim != 1 or offsets.shape != lengths.shape or lengths.numel() != kernel_block_table.shape[0]:
        raise ValueError("C8 history lengths/starts must match block table requests")
    if bool((lengths < 0).any()) or bool((offsets < 0).any()):
        raise ValueError("C8 history lengths/starts must be non-negative")
    host_bounds = lengths.device.type == offsets.device.type == "cpu"
    if host_bounds and bool(
        ((lengths > 0) & (offsets + lengths > kernel_block_table.shape[1] * PA_NZ_BLOCK_SIZE)).any()
    ):
        raise ValueError("C8 history block table does not cover requested positions")
    total = int(lengths.sum().item())
    lengths = lengths.to(device=cache.device)
    offsets = offsets.to(device=cache.device)
    if total == 0:
        return torch.empty((0, 1, K3_MLA_CACHE_WIDTH), dtype=dtype, device=cache.device)
    requests = torch.repeat_interleave(torch.arange(lengths.numel(), device=cache.device), lengths, output_size=total)
    positions = torch.arange(total, device=cache.device) - (lengths.cumsum(0) - lengths)[requests] + offsets[requests]
    columns = positions // PA_NZ_BLOCK_SIZE
    if not host_bounds and bool((columns >= kernel_block_table.shape[1]).any()):
        raise ValueError("C8 history block table does not cover requested positions")
    table = kernel_block_table.to(device=cache.device)
    block_ids = table[requests, columns].long()
    if bool(((block_ids < 0) | (block_ids >= cache.shape[0])).any()):
        raise ValueError("C8 history active block table entry is outside cache bounds")
    tiles = cache.view(torch.uint8)[block_ids, 0, :, positions % PA_NZ_BLOCK_SIZE, :]
    rows = tiles.reshape(total, K3_MLA_CACHE_WIDTH)
    return dequantize_static_fp8(rows, scale, dtype).unsqueeze(1)


def validate_pa_nz_page_stride(block_size: int, page_stride_bytes: int, width: int = K3_MLA_CACHE_WIDTH) -> None:
    """Allow only complete physical kernel pages between manager-page bases."""
    _kernel_blocks_per_manager(block_size)
    if (
        width <= 0
        or width % PA_NZ_TILE
        or page_stride_bytes < block_size * width
        or page_stride_bytes % (PA_NZ_BLOCK_SIZE * width)
    ):
        raise ValueError("C8 PA_NZ page padding must contain whole 128-token kernel pages")


def pa_nz_physical_blocks_per_page(block_size: int, page_stride_bytes: int, width: int = K3_MLA_CACHE_WIDTH) -> int:
    validate_pa_nz_page_stride(block_size, page_stride_bytes, width)
    return page_stride_bytes // (PA_NZ_BLOCK_SIZE * width)


def map_pa_nz_slots(slots: torch.Tensor, block_size: int, page_stride_bytes: int) -> torch.Tensor:
    """Map logical local slots into aligned physical pages; preserve padding -1."""
    physical = pa_nz_physical_blocks_per_page(block_size, page_stride_bytes)
    if physical == _kernel_blocks_per_manager(block_size):
        return slots
    mapped = (slots // block_size) * physical * PA_NZ_BLOCK_SIZE + slots % block_size
    return torch.where(slots >= 0, mapped, slots)


def map_pa_nz_kernel_block_table(kernel_ids: torch.Tensor, block_size: int, page_stride_bytes: int) -> torch.Tensor:
    """Map V2's already-expanded logical kernel IDs to physical kernel IDs."""
    logical = _kernel_blocks_per_manager(block_size)
    physical = pa_nz_physical_blocks_per_page(block_size, page_stride_bytes)
    if physical == logical:
        return kernel_ids
    mapped = (kernel_ids // logical) * physical + kernel_ids % logical
    return torch.where(kernel_ids >= 0, mapped, kernel_ids)
