from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.attention.kimi_c8_static import K3_MLA_CACHE_WIDTH, PA_NZ_TILE, pa_nz_physical_blocks_per_page


@dataclass(frozen=True)
class KimiC8Spec:
    block_size: int
    kernel_block_size: int = 128
    width: int = K3_MLA_CACHE_WIDTH
    quant_mode: int = 1
    head_dim_v: int = 512
    page_stride_bytes: int | None = None

    def __post_init__(self):
        if self.block_size <= 0:
            raise ValueError("block sizes must be positive")
        if self.kernel_block_size != 128:
            raise ValueError("Kimi C8 physical kernel block size must be 128")
        if self.block_size % self.kernel_block_size:
            raise ValueError("scheduler block must contain whole kernel blocks")
        if self.width % PA_NZ_TILE or self.width != 576:
            raise ValueError("Kimi C8 cache width must be 576 and tile aligned")
        if self.quant_mode != 1:
            raise ValueError("Kimi C8 static path only supports E4M3 quant_mode=1")
        _ = self.physical_kernel_blocks_per_page

    @property
    def kernel_blocks_per_page(self):
        return self.block_size // self.kernel_block_size

    @property
    def page_bytes(self):
        return self.block_size * self.width

    @property
    def physical_kernel_blocks_per_page(self):
        stride = self.page_bytes if self.page_stride_bytes is None else self.page_stride_bytes
        return pa_nz_physical_blocks_per_page(self.block_size, stride, self.width)

    def cache_shape(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("PA_NZ requires a positive manager block count")
        return (num_blocks * self.physical_kernel_blocks_per_page, 1, self.width // PA_NZ_TILE, 128, PA_NZ_TILE)

    def validate_cache(self, cache: torch.Tensor):
        # uint8 is the persistent byte view; E4M3 is the zero-copy FA view.
        if (
            cache.ndim != 5
            or cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or tuple(cache.shape[1:]) != self.cache_shape(1)[1:]
            or cache.shape[0] <= 0
            or cache.shape[0] % self.physical_kernel_blocks_per_page
            or not cache.is_contiguous()
        ):
            raise ValueError(f"invalid PA_NZ cache shape/dtype: {tuple(cache.shape)}, {cache.dtype}")
        return cache


def build_quant_mla_metadata_kwargs(
    cache_seqlens,
    cu_seqlens_q,
    seqused_q,
    *,
    num_heads_q: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim_qk: int = 576,
    head_dim_v: int = 512,
    global_seqlens: torch.Tensor | None = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 128,
):
    if head_dim_qk != 576 or head_dim_v != 512:
        raise ValueError("Kimi C8 metadata requires qk=576 and v=512")
    if num_heads_q <= 0:
        raise ValueError("num_heads_q must be positive")
    kwargs = dict(
        cache_seqlens=cache_seqlens,
        num_heads_q=num_heads_q,
        num_heads_kv=1,
        quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_qk=576,
        head_dim_v=512,
        mask_mode=0 if max_seqlen_q == 1 else 3,
        layout_q="TND",
    )
    if enable_dcp:
        if (
            cp_world <= 0
            or not 0 <= cp_rank < cp_world
            or cp_kv_cache_interleave_size <= 0
            or cp_kv_cache_interleave_size % 128
        ):
            raise ValueError("native DCP requires valid world/rank and 128-aligned actual interleave")
        if (
            not isinstance(global_seqlens, torch.Tensor)
            or global_seqlens.ndim != 1
            or global_seqlens.shape != cache_seqlens.shape
            or global_seqlens.dtype != torch.int32
        ):
            raise ValueError("native DCP global_seqlens must be int32 with the same request shape as cache_seqlens")
        # Only the explicitly enabled extension receives these keywords. Older
        # installed operators keep the non-DCP Q1 adapter contract unchanged.
        kwargs.update(
            global_seqlens=global_seqlens,
            enable_dcp=True,
            cp_world=cp_world,
            cp_rank=cp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
    return kwargs
