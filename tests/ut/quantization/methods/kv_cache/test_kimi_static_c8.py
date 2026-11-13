from types import SimpleNamespace as Ns

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode as Mode

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.quantization.methods.kv_cache import kv_c8


@pytest.fixture
def configuration(monkeypatch):
    config = Ns(
        model_config=Ns(
            hf_config=Ns(model_type="kimi_linear", kv_lora_rank=512, qk_rope_head_dim=64, mla_use_output_gate=False)
        ),
        mla_cache_mode="C8",
        compilation_config=Ns(cudagraph_mode=Mode.NONE),
        parallel_config=Ns(
            prefill_context_parallel_size=1, decode_context_parallel_size=8, cp_kv_cache_interleave_size=1536
        ),
        kv_transfer_config=Ns(is_kv_producer=False, kv_connector="MooncakePullConnector"),
        cache_config=Ns(block_size=1536),
        use_v2_model_runner=True,
        speculative_config=None,
    )
    config.model_config.hf_text_config = config.model_config.hf_config
    monkeypatch.setattr(kv_c8, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(kv_c8, "get_ascend_config", lambda: config)
    monkeypatch.setattr(kv_c8, "get_current_hardware_profile", lambda: Ns(supports=lambda _: True))
    return config


def layer(scale=2.0, target=True):
    return Ns(
        impl=Ns(kimi_c8_mode=target),
        num_heads=2,
        num_kv_heads=1,
        fa_k=Ns(
            scale=torch.nn.Parameter(torch.tensor([scale]), requires_grad=False),
            offset=torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False),
        ),
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("scale_zero", "scale"),
        ("scale_nan", "scale"),
        ("offset", "offset"),
        ("model", "Kimi K3"),
        ("graph", "FULL_DECODE_ONLY"),
        ("producer", "producer.*eager"),
    ],
)
def test_incompatible_contract_fails_before_marking(configuration, mutation, error):
    target = layer()
    if mutation.startswith("scale"):
        target.fa_k.scale.data.fill_(0.0 if mutation == "scale_zero" else torch.nan)
    elif mutation == "offset":
        target.fa_k.offset.data.fill_(1)
    elif mutation == "model":
        configuration.model_config.hf_config.model_type = "deepseek_v3"
    else:
        configuration.compilation_config.cudagraph_mode = Mode.FULL if mutation == "graph" else Mode.FULL_DECODE_ONLY
        configuration.kv_transfer_config = Ns(is_kv_producer=mutation == "producer")
    with pytest.raises(ValueError, match=error):
        kv_c8.AscendFAQuantAttentionMethod().process_weights_after_loading(target)
    assert not getattr(target, "kimi_c8_static_fp8", False)


@pytest.mark.parametrize("value,expected", [("default", "C16"), ("c8", "C8"), (" C16 ", "C16")])
def test_configuration_mode_normalization_and_c16_passthrough(value, expected):
    kwargs = {} if value == "default" else {"mla_cache_mode": value}
    options = AscendConfig(sparse_kv_offload_config=Ns(enabled=False), **kwargs)
    assert options.mla_cache_mode == expected
    if expected == "C16":
        options._validate_mla_cache_mode(None)  # Existing models must not be inspected.


@pytest.mark.parametrize("value", ["INT8", "FP8", "", None, True, 8])
def test_configuration_rejects_invalid_mode(value):
    with pytest.raises(ValueError, match="mla_cache_mode"):
        AscendConfig(mla_cache_mode=value, sparse_kv_offload_config=Ns(enabled=False))


@pytest.mark.parametrize(
    "owner,field,value,error",
    [
        (None, None, None, None),
        ("root", "speculative_config", Ns(method="dspark"), None),
        ("model", "model_type", "deepseek_v3", "requires Kimi K3"),
        ("model", "kv_lora_rank", 256, "requires Kimi K3"),
        ("model", "qk_rope_head_dim", 0, "requires Kimi K3"),
        ("root", "use_v2_model_runner", False, "RunnerV2"),
        ("root", "speculative_config", Ns(method="eagle"), "DSpark"),
        ("parallel", "prefill_context_parallel_size", 2, "prefill_context_parallel"),
        ("parallel", "decode_context_parallel_size", 4, "decode_context_parallel"),
        ("parallel", "cp_kv_cache_interleave_size", 128, "interleave"),
        ("transfer", "is_kv_producer", True, "KV producer"),
        ("transfer", "kv_connector", "MooncakeConnectorV1", "MooncakeConnectorV2"),
        ("cache", "block_size", 128, "block_size=1536"),
    ],
)
def test_configuration_checks_model_runner_pd_and_parallel_boundaries(configuration, owner, field, value, error):
    if owner:
        target = {
            "root": configuration,
            "model": configuration.model_config.hf_text_config,
            "parallel": configuration.parallel_config,
            "transfer": configuration.kv_transfer_config,
            "cache": configuration.cache_config,
        }[owner]
        setattr(target, field, value)
    options = AscendConfig(mla_cache_mode="C8", sparse_kv_offload_config=Ns(enabled=False))
    if error:
        with pytest.raises(ValueError, match=error):
            options._validate_mla_cache_mode(configuration)
    else:
        before = vars(configuration.model_config.hf_text_config).copy()
        options._validate_mla_cache_mode(configuration)
        assert vars(configuration.model_config.hf_text_config) == before


@pytest.mark.parametrize(
    "mode,cache_mode,draft,static",
    [
        (Mode.NONE, "C8", False, True),
        (Mode.FULL_DECODE_ONLY, "C8", False, True),
        (Mode.FULL, "C16", False, False),
        (Mode.FULL, "C8", True, False),
    ],
)
def test_real_model_identity_descale_and_legacy_dtype(configuration, mode, cache_mode, draft, static):
    configuration.mla_cache_mode = cache_mode
    configuration.compilation_config.cudagraph_mode = mode
    configuration.kv_transfer_config.is_kv_producer = not static
    target = layer(target=not draft)
    target.kv_cache_torch_dtype = torch.int8
    target.fa_k.offset.data.fill_(0 if static else 3)
    kv_c8.AscendFAQuantAttentionMethod().process_weights_after_loading(target)
    assert getattr(target, "kimi_c8_static_fp8", False) is static
    assert target.kv_cache_torch_dtype == (torch.float8_e4m3fn if static else torch.int8)
    assert target.fak_descale_float.item() == 2
    if not static:
        assert target.fak_offset.item() == 3
