import math
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from typing import Literal, get_args

import vllm.config.speculative as speculative_config
from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig

from vllm_ascend.utils import is_deepseek_v41, is_gqa_pcp_dcp_config

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override


def _is_redhat_window_gqa(hf_config: PretrainedConfig) -> bool:
    architectures = getattr(hf_config, "architectures", ()) or ()
    return (
        getattr(hf_config, "model_type", None) in ("qwen3", "speculators")
        and (
            "Qwen3DSparkModel" in architectures
            or "DSparkDraftModel" in architectures
        )
        and bool(getattr(hf_config, "use_sliding_window", False))
        and bool(getattr(hf_config, "sample_from_anchor", False))
        and getattr(hf_config, "block_size", None) == 8
        and getattr(hf_config, "sliding_window", None) is not None
    )


# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


# Identify the deployed z-lab/Kimi-K2.5-DFlash checkpoint independently of its
# local directory name. Other DFlash checkpoints may require the new YaRN math.
_KIMI_DFLASH_CONFIG = (
    ("model_type", "qwen3"),
    ("hidden_size", 7168),
    ("vocab_size", 163840),
    ("num_target_layers", 61),
)
_KIMI_DFLASH_TARGET_LAYER_IDS = (1, 12, 24, 35, 47, 58)
_KIMI_DFLASH_YARN_PARAMS = (
    ("rope_type", "yarn"),
    ("factor", 64.0),
    ("mscale", 1.0),
    ("mscale_all_dim", 1.0),
    ("original_max_position_embeddings", 4096),
)


def _normalize_kimi_dflash_rope(hf_config: PretrainedConfig) -> None:
    """Preserve this legacy Kimi draft's vLLM 0.29 YaRN amplitude.

    vLLM #56446 starts honoring mscale/mscale_all_dim for plain YaRN. For
    this checkpoint their ratio is 1, instead of the old 1 + 0.1 * log(64),
    which reduces draft acceptance. Make the old amplitude explicit only for
    the known config; an explicit attention_factor always takes precedence.
    This compatibility shim can go away once the checkpoint specifies it.
    """
    if "DFlashDraftModel" not in (getattr(hf_config, "architectures", None) or ()):
        return
    if any(getattr(hf_config, key, None) != value for key, value in _KIMI_DFLASH_CONFIG):
        return
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if tuple(dflash_config.get("target_layer_ids") or ()) != _KIMI_DFLASH_TARGET_LAYER_IDS:
        return

    # Transformers 5 stores the old rope_scaling field in rope_parameters.
    # Copy before updating so a shared source dict is not modified in place.
    rope_field = "rope_parameters"
    rope_params = getattr(hf_config, rope_field, None)
    if rope_params is None:
        rope_field = "rope_scaling"
        rope_params = getattr(hf_config, rope_field, None)
    if not isinstance(rope_params, dict) or rope_params.get("attention_factor") is not None:
        return
    if any(rope_params.get(key) != value for key, value in _KIMI_DFLASH_YARN_PARAMS):
        return

    legacy_attention_factor = 1.0 + 0.1 * math.log(rope_params["factor"])
    setattr(hf_config, rope_field, {**rope_params, "attention_factor": legacy_attention_factor})


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _orig_hf_config_override(hf_config)
    _normalize_kimi_dflash_rope(hf_config)
    architectures = hf_config.architectures or ()
    is_qwen3_dspark = (
        hf_config.model_type == "qwen3"
        and (
            "DSparkDraftModel" in architectures
            or "Qwen3DSparkModel" in architectures
        )
        or hf_config.model_type == "speculators"
        and (
            "DSparkDraftModel" in architectures
            or "Qwen3DSparkModel" in architectures
        )
    )
    if is_qwen3_dspark:
        # RedHat's Window-GQA checkpoint stores Qwen3 decoder fields in the
        # nested transformer_layer_config. Expose them at the top level before
        # the Qwen3 DSpark model is constructed.
        outer_non_causal = getattr(hf_config, "sliding_window_non_causal", None)
        transformer_config = getattr(hf_config, "transformer_layer_config", None)
        if isinstance(transformer_config, dict):
            hf_config.update(transformer_config)
        if getattr(hf_config, "model_type", None) == "speculators":
            hf_config.model_type = "qwen3"

        dflash_config = dict(getattr(hf_config, "dflash_config", None) or {})
        aux_layer_ids = getattr(hf_config, "aux_hidden_state_layer_ids", None)
        if aux_layer_ids is None:
            aux_layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        target_layer_ids = dflash_config.get("target_layer_ids")
        if target_layer_ids is None:
            target_layer_ids = getattr(hf_config, "target_layer_ids", None)
        if target_layer_ids is None and aux_layer_ids:
            # RedHat stores auxiliary layer IDs as one-based values, while
            # vLLM's target_layer_ids are zero-based.
            target_layer_ids = [int(layer_id) - 1 for layer_id in aux_layer_ids]

        normalized = {
            "architectures": ["Qwen3DSparkModel"],
            "mask_token_id": dflash_config.get(
                "mask_token_id", getattr(hf_config, "mask_token_id", None)
            ),
        }
        if target_layer_ids is not None:
            normalized["target_layer_ids"] = list(target_layer_ids)
            normalized["dspark_target_layer_ids"] = list(target_layer_ids)
        if aux_layer_ids:
            normalized["eagle_aux_hidden_state_layer_ids"] = [int(i) for i in aux_layer_ids]

        non_causal = outer_non_causal
        if non_causal is None and isinstance(transformer_config, dict):
            non_causal = transformer_config.get("sliding_window_non_causal")
        if non_causal is not None:
            causal = not bool(non_causal)
            normalized["is_causal"] = causal
            dflash_config["causal"] = causal
        elif (
            bool(getattr(hf_config, "use_sliding_window", False))
            and getattr(hf_config, "layer_types", None)
            and all(layer_type == "sliding_attention" for layer_type in hf_config.layer_types)
        ):
            dflash_config.setdefault("causal", True)

        # Native Window-GQA samples eight rows from one anchor. FIA's causal
        # window is query-relative, so extend the pre-token span by seven rows;
        # dspark_proposer still crops the actual block table to the W-token
        # history window.
        sliding_window = getattr(hf_config, "sliding_window", None)
        native_width = getattr(hf_config, "block_size", None)
        if (
            sliding_window is not None
            and bool(getattr(hf_config, "use_sliding_window", False))
            and bool(getattr(hf_config, "sample_from_anchor", False))
            and aux_layer_ids is not None
            and native_width == 8
        ):
            dflash_config["use_swa"] = True
            dflash_config["swa_window_size"] = int(sliding_window) + int(native_width) - 1
        if target_layer_ids is not None:
            dflash_config["target_layer_ids"] = list(target_layer_ids)
        if normalized["mask_token_id"] is not None:
            dflash_config.setdefault("mask_token_id", normalized["mask_token_id"])
        if dflash_config:
            normalized["dflash_config"] = dflash_config
        hf_config.update(normalized)
    if hf_config.model_type in ("glm5_next", "glm5_next_text"):
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.model_type = "glm5_next_mtp"
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Glm5NextMTPModel"],
            }
        )
    return hf_config


def _normalize_deepseek_dspark_draft(draft_model_config) -> None:
    """Restore the DSpark draft architecture after VL config conversion.

    DeepSeek-V4-Vision uses the same checkpoint for the target and DSpark
    drafter.  vLLM first rewrites that checkpoint to ``DSparkDraftModel``, but
    rebuilding ``model_arch_config`` with multimodal detection can restore the
    top-level ``*ForConditionalGeneration`` architecture.  The drafter would
    then instantiate a second full VL target and register duplicate attention
    layer names.  Update both config representations without re-running the
    multimodal architecture conversion.
    """
    hf_config = getattr(draft_model_config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", None)
    draft_text_config = text_config if text_config is not None else hf_config
    root_model_type = getattr(hf_config, "model_type", None)
    is_deepseek_v41_model = is_deepseek_v41(hf_config)
    if (
        hf_config is None
        or (root_model_type != "deepseek_v4" and not is_deepseek_v41_model)
        or getattr(draft_text_config, "dspark_target_layer_ids", None) is None
    ):
        return

    architecture = "DeepseekV41DSparkModel" if is_deepseek_v41_model else "DSparkDraftModel"
    if is_deepseek_v41_model:
        # The DeepSeek V4.1 target and draft experts intentionally have different
        # widths.  SpeculativeConfig owns a private config copy, so adapting
        # these fields cannot alter the target model.
        draft_experts_per_token = getattr(draft_text_config, "dspark_num_experts_per_tok", None)
        draft_updates = {
            "n_routed_experts": draft_text_config.dspark_n_routed_experts,
            "n_mtp_layers": getattr(draft_text_config, "num_nextn_predict_layers", 3),
        }
        if draft_experts_per_token is not None:
            draft_updates["num_experts_per_tok"] = draft_experts_per_token
        draft_text_config.update(draft_updates)
    normalized_model_type = "deepseek_v41" if is_deepseek_v41_model else str(root_model_type)
    hf_config.update(
        {
            "architectures": [architecture],
            "model_type": normalized_model_type,
        }
    )
    arch_updates = dict(
        architectures=[architecture],
        model_type=normalized_model_type,
        is_mm_prefix_lm=False,
    )
    if is_deepseek_v41_model:
        arch_updates.update(
            num_experts=draft_text_config.n_routed_experts,
            text_model_type=getattr(draft_text_config, "model_type", None),
        )
        if draft_experts_per_token is not None:
            arch_updates["num_experts_per_token"] = draft_experts_per_token
    draft_model_config.model_arch_config = replace(
        draft_model_config.model_arch_config,
        **arch_updates,
    )
    architectures = draft_model_config.model_arch_config.architectures
    model_info, architecture = draft_model_config.registry.inspect_model_cls(
        architectures,
        draft_model_config,
    )
    draft_model_config._model_info = model_info
    draft_model_config._architecture = architecture


@contextmanager
def _temporarily_disable_dspark_dcp(self: SpeculativeConfig):
    target_parallel_config = self.target_parallel_config
    if getattr(self, "method", None) != "dspark" or target_parallel_config.decode_context_parallel_size <= 1:
        yield
        return

    guard_parallel_config = copy(target_parallel_config)
    guard_parallel_config.decode_context_parallel_size = 1
    self.target_parallel_config = guard_parallel_config
    try:
        yield
    finally:
        self.target_parallel_config = target_parallel_config


def _dspark_post_init(self):
    # TODO: This block can be deleted after the upstream supports the overlay of mla dcp and dspark
    draft_model_config = getattr(self, "draft_model_config", None)
    draft_hf_config = getattr(draft_model_config, "hf_config", None)
    with _temporarily_disable_dspark_dcp(self):
        _orig_post_init(self)
    # The draft config is normally created by the upstream post-init above.
    draft_model_config = getattr(self, "draft_model_config", None)
    draft_hf_config = getattr(draft_model_config, "hf_config", None)
    # RedHat's native Window-GQA width is eight rows; the Ascend verify path
    # exposes seven speculative tokens to the sampler and proposer.
    if draft_hf_config is not None and _is_redhat_window_gqa(draft_hf_config):
        if self.num_speculative_tokens == 8:
            self.num_speculative_tokens = 7
            if hasattr(draft_hf_config, "num_lookahead_tokens"):
                draft_hf_config.num_lookahead_tokens = 7
    if self.use_dspark():
        _normalize_deepseek_dspark_draft(draft_model_config)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore
        _normalize_glm_mla_dspark(self)


def _normalize_glm_mla_dspark(self):
    """Restore MLA architecture erased by the pinned Speculators DSpark converter.

    Remove this narrow compatibility shim when upstream Speculators conversion
    preserves Glm5DSparkForCausalLM and its MLA architecture metadata.
    """
    draft = getattr(self, "draft_model_config", None)
    config = getattr(draft, "hf_config", None)
    if draft is None or not isinstance(config, SpeculatorsConfig):
        return
    source, _ = SpeculatorsConfig.get_config_dict(draft.model)
    if "Glm5DSparkForCausalLM" not in (source.get("architectures") or ()):
        return
    mla_fields = ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
    if any(not isinstance(getattr(config, name, None), int) or getattr(config, name) <= 0 for name in mla_fields):
        raise ValueError("Glm5DSparkForCausalLM requires dense MLA projection dimensions; cannot use GQA weights.")
    config.architectures = ["Glm5DSparkForCausalLM"]
    self.update_arch_()
    draft.model_arch_config = replace(draft.model_arch_config, is_deepseek_mla=True)


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init

# The pinned vLLM revision propagates enable_expert_parallel to the draft
# parallel config (upstream #55914) but no longer disables it for dense
# drafts (upstream #56930 is not on this revision). Non-MoE draft models
# (e.g. Kimi K3 DSpark, VWN eagle3) then fail the
# _verify_with_expert_parallelism check in
# ModelConfig.verify_with_parallel_config. Skip the EP check for non-MoE
# draft model configs; the target EP check and MoE draft models are
# unaffected.

_orig_verify_with_parallel_config = ModelConfig.verify_with_parallel_config


def _ascend_verify_with_parallel_config(self, parallel_config):
    if parallel_config.enable_expert_parallel and not self.is_moe and getattr(self, "runner_type", None) == "draft":
        return
    # TODO: Remove this guard once upstream GQA/MQA DCP validation supports
    # PCP-based KV-head replica groups.
    if is_gqa_pcp_dcp_config(self, parallel_config):
        guard_parallel_config = copy(parallel_config)
        guard_parallel_config.decode_context_parallel_size = 1
        return _orig_verify_with_parallel_config(self, guard_parallel_config)
    return _orig_verify_with_parallel_config(self, parallel_config)


ModelConfig.verify_with_parallel_config = _ascend_verify_with_parallel_config

if "glm5_next_mtp" not in get_args(speculative_config.MTPModelTypes):
    speculative_config.MTPModelTypes = Literal[(*get_args(speculative_config.MTPModelTypes), "glm5_next_mtp")]
