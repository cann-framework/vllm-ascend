"""CPU checks for the PA_NZ cache layout used by VLLM_ASCEND_ENABLE_PA_NZ."""

from vllm_ascend.attention.flashmla import FlashMLAConfig


def test_pa_nz_config_allows_non_power_of_two_heads() -> None:
    # Kimi K3 at TP16 exposes 6 local query heads.
    config = FlashMLAConfig(num_heads=6, softmax_scale=0.1, layout_kv="PA_NZ")
    assert config.num_heads == 6
