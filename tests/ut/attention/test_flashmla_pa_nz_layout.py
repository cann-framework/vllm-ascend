"""CPU checks for the PA_NZ cache layout used by VLLM_ASCEND_ENABLE_FLASH_MLA."""

import pytest
import torch

from vllm_ascend.attention.flashmla import FlashMLAConfig
from vllm_ascend.ops.flash_mla_pa_nz import PA_NZ_TILE, pa_nz_cache_view
from vllm_ascend.worker.utils import make_pa_nz_cache_view


def test_make_pa_nz_cache_view_contiguous() -> None:
    blocks, block_size, width = 2, 4, 576
    raw = torch.arange(blocks * block_size * width, dtype=torch.float32).to(torch.bfloat16)
    view = make_pa_nz_cache_view(
        raw,
        (blocks, block_size, 1, width),
        torch.bfloat16,
        block_size * width * 2,
    )
    assert view.shape == (blocks, block_size, 1, width)
    assert view.is_contiguous()
    assert torch.equal(view.reshape(-1), raw)

    nz = pa_nz_cache_view(view)
    assert nz.shape == (blocks, 1, width // PA_NZ_TILE, block_size, PA_NZ_TILE)
    assert torch.equal(nz.reshape(-1), raw)


def test_make_pa_nz_cache_view_rejects_padded_page() -> None:
    blocks, block_size, width = 2, 4, 576
    raw = torch.zeros(blocks * (block_size * width + 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous page"):
        make_pa_nz_cache_view(
            raw,
            (blocks, block_size, 1, width),
            torch.bfloat16,
            block_size * width * 2 + 32,
        )


def test_pa_nz_config_allows_non_power_of_two_heads() -> None:
    # Kimi K3 at TP16 exposes 6 local query heads.
    config = FlashMLAConfig(num_heads=6, softmax_scale=0.1, layout_kv="PA_NZ")
    assert config.num_heads == 6
