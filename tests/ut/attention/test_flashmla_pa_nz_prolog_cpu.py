# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the DSL prolog interface without torch, vLLM or an NPU.

Run with --confcutdir=tests/ut/attention. These checks validate argument
binding and decode orchestration, not device accuracy or graph execution.
"""

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def tensor(**attributes):
    value = MagicMock(**attributes)
    for method in ("contiguous", "detach", "float", "reshape", "to", "view"):
        getattr(value, method).return_value = value
    value.__getitem__.return_value = value
    return value


@pytest.fixture
def prolog(monkeypatch):
    torch = ModuleType("torch")
    torch.Tensor = object
    torch.int64 = object()
    torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    api = runpy.run_path(str(ROOT / "vllm_ascend/ops/flash_mla_pa_nz.py"))
    calls = []
    output = tensor()
    modes = SimpleNamespace(
        QkvQuantMode=SimpleNamespace(A16C16=object()),
        CacheMode=SimpleNamespace(PA_NZ=object()),
        RopeMode=SimpleNamespace(NO_ROPE=object()),
    )

    # Match PR 1266's nine positional tensors and keyword attributes exactly.
    def provider(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        gamma_qa,
        gamma_kva,
        cache,
        slots,
        *,
        norm_eps,
        qkv_quant_mode,
        cache_mode,
        rope_mode,
        return_qr
    ):
        calls.append((x, wqa, wqb, wkva, wkb, gamma_qa, gamma_kva, cache, slots, norm_eps))
        assert qkv_quant_mode is modes.QkvQuantMode.A16C16
        assert cache_mode is modes.CacheMode.PA_NZ
        assert rope_mode is modes.RopeMode.NO_ROPE
        assert return_qr is False
        return output, None, None

    modes.mla_prolog_v2 = provider
    importer = Mock(return_value=modes)
    api["_get_bf16_mla_prolog_op"].__wrapped__.__globals__["import_module"] = importer
    return SimpleNamespace(api=api, torch=torch, modes=modes, importer=importer, calls=calls, output=output)


def test_latest_prolog_signature_and_inplace_cache(prolog):
    x, wqa, wqb, wkb, wkva, gamma_qa, gamma_kva, cache, slots = [tensor() for _ in range(9)]
    run = prolog.api["mla_prolog_no_rope"]
    for _ in range(2):
        result = run(x, wqa, wqb, wkb, wkva, gamma_qa, gamma_kva, cache, slots, 3e-6)
        assert result == (prolog.output, None, None)
        assert prolog.calls[-1] == (x, wqa, wqb, wkva, wkb, gamma_qa, gamma_kva, cache, slots, 3e-6)
    prolog.importer.assert_called_once_with("ops.mla_prolog_v4")
    slots.to.assert_called_with(prolog.torch.int64)
    cache.clone.assert_not_called()
    cache.copy_.assert_not_called()


@pytest.mark.parametrize("missing", ["module", "mla_prolog_v2", "QkvQuantMode", "CacheMode", "RopeMode"])
def test_incompatible_dsl_install_has_actionable_error(prolog, missing):
    if missing == "module":
        prolog.importer.side_effect = ImportError("missing DSL prolog")
    else:
        delattr(prolog.modes, missing)
    with pytest.raises(RuntimeError, match=r"requires ops\.mla_prolog_v4\.mla_prolog_v2"):
        prolog.api["_get_bf16_mla_prolog_op"]()


def load_method(name, scope):
    path = ROOT / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(path), "exec"), scope
    )
    return scope[name]


def test_pa_nz_rejects_different_rmsnorm_eps_before_launch(prolog):
    scope = {
        "torch": prolog.torch,
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
        "HardwareCapability": SimpleNamespace(MLA_FLASH=object()),
    }
    validate = load_method("_validate_flash_mla_pa_nz_layer", scope)
    impl = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        is_draft_model=False,
        use_mla_rope=False,
        fa_quant_layer=False,
        enable_kv_nz=False,
        q_lora_rank=1536,
        fused_qkv_a_proj=object(),
        q_proj=object(),
        q_a_layernorm=SimpleNamespace(variance_epsilon=1e-5),
        kv_a_layernorm=SimpleNamespace(variance_epsilon=1e-6),
    )
    with pytest.raises(ValueError, match="identical Q/KV RMSNorm epsilon"):
        validate(impl)
    prolog.importer.assert_not_called()


@pytest.mark.parametrize("dcp", [False, True])
def test_decode_consumes_three_outputs_and_preserves_dcp_current_cache(prolog, monkeypatch, dcp):
    module = ModuleType("vllm_ascend.ops.flash_mla_pa_nz")
    module.mla_prolog_no_rope = prolog.api["mla_prolog_no_rope"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    run = load_method(
        "_forward_decode_flash_bf16",
        {
            "torch": prolog.torch,
            "FLASHMLA_BLOCK_SIZE": 128,
            "PA_NZ_TILE_COUNT": 36,
            "record_attention_compute_start": Mock(),
        },
    )
    history_cache, current_cache = tensor(shape=(3, 128, 1, 576)), tensor()
    history_slots, current_slots = tensor(), tensor()
    projected = object()
    flash = SimpleNamespace(
        metadata_async=False,
        metadata_started=True,
        block_columns=None,
        prolog_slots=history_slots,
        current=SimpleNamespace(cache=current_cache) if dcp else None,
        pa_nz_current_slots=current_slots,
        pa_nz_history_slots=history_slots,
        pa_nz_copy_src_indices=tensor(),
        pa_nz_copy_dst_indices=tensor(),
        adapter=SimpleNamespace(attention=Mock(return_value=(prolog.output, None))),
        block_table=object(),
        cache_lens=object(),
        cu=object(),
        used_q=object(),
        schedule=object(),
        q_len=1,
    )
    metadata = SimpleNamespace(num_decode_tokens=3, get_external_flashmla=lambda _: flash)
    impl = SimpleNamespace(
        flash_mla_layout="pa_nz",
        q_a_layernorm=SimpleNamespace(variance_epsilon=1e-6),
        kv_a_layernorm=object(),
        weight_dq=tensor(),
        flash_weight_uq_qr=tensor(),
        flash_weight_uk=tensor(),
        weight_dkv_kr=tensor(),
        flash_gamma_qa=tensor(),
        flash_gamma_kva=tensor(),
        flash_total_heads=96,
        flash_local_head_start=0,
        num_heads=6,
        dcp_enabled=dcp,
        kv_lora_rank=512,
        wait_flash_mla_metadata=Mock(),
        reorg_decode_q=lambda q, pe: (q, pe),
        _copy_pa_nz_rows=Mock(),
        _forward_decode_split_attention=Mock(return_value=projected),
        _v_up_proj=Mock(return_value=projected),
    )
    assert run(impl, tensor(), history_cache, metadata) is projected
    assert prolog.calls[0][7] is (current_cache if dcp else history_cache)
    assert prolog.calls[0][8] is (current_slots if dcp else history_slots)
    impl.wait_flash_mla_metadata.assert_called_once_with(metadata)
    if dcp:
        impl._copy_pa_nz_rows.assert_called_once_with(
            current_cache,
            history_cache,
            flash.pa_nz_copy_src_indices,
            flash.pa_nz_copy_dst_indices,
        )
        split = impl._forward_decode_split_attention.call_args
        assert split.args[2] is history_cache
        assert split.kwargs["current_kv_staged"] is True
    else:
        impl._copy_pa_nz_rows.assert_not_called()
        assert flash.adapter.attention.call_args.args[1] is history_cache
