# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.


import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend import envs
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list
from vllm_ascend.quantization.quant_type import QuantType


def apply_moe_mlp(
    mlp_compute_input: MoEMlpComputeInput,
    quant_method,
) -> tuple[torch.Tensor, torch.npu.Event]:
    """
    Unified MoE MLP entry.
    Quant path is dispatched by each FusedMoEMethod with explicit typed kernel flags.
    """

    bypass_mode = envs.VLLM_ASCEND_MOE_MLP_BYPASS
    if bypass_mode is not None:
        return _apply_bypassed_moe_mlp(mlp_compute_input, quant_method, bypass_mode)

    # When LoRA adapter is used in quantized weight, use individual lora impl.
    if mlp_compute_input.quant.is_quant and mlp_compute_input.lora_context is not None:
        from vllm_ascend.lora.fused_moe import has_lora

        if has_lora(mlp_compute_input.lora_context):
            from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora

            return quant_apply_mlp_with_moe_lora(
                mlp_compute_input=mlp_compute_input,
                quant_method=quant_method,
            )

    if quant_method.supports_fused_activation(mlp_compute_input.activation):
        hidden_states, act_out_scale = quant_method.apply_gmm1_act_quant(mlp_compute_input)
    else:
        hidden_states = quant_method.apply_gmm1(mlp_compute_input)
        if mlp_compute_input.shared_expert_schedule is not None:
            mlp_compute_input.shared_expert_schedule.after_gmm1()
        hidden_states = _unified_apply_activation(mlp_compute_input, hidden_states, quant_method)
        hidden_states, act_out_scale = quant_method.apply_act_quant(mlp_compute_input, hidden_states)

    before_gmm2_evt = torch.npu.current_stream().record_event()
    hidden_states = quant_method.apply_gmm2(mlp_compute_input, hidden_states, act_out_scale)
    return hidden_states, before_gmm2_evt


def _apply_bypassed_moe_mlp(
    mlp_compute_input: MoEMlpComputeInput,
    quant_method,
    bypass_mode: int,
) -> tuple[torch.Tensor, torch.npu.Event]:
    input_hidden_states = mlp_compute_input.hidden_states
    input_shape = tuple(input_hidden_states.shape)
    input_device = input_hidden_states.device
    input_dtype = input_hidden_states.dtype
    if input_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        input_dtype = torch.bfloat16

    def debug_tensor(output_size: int) -> torch.Tensor:
        return torch.zeros((*input_shape[:-1], output_size), device=input_device, dtype=input_dtype)

    final_hidden_size = input_shape[-1]

    if bypass_mode == 0:
        hidden_states = debug_tensor(final_hidden_size)
        before_gmm2_evt = torch.npu.current_stream().record_event()
        return hidden_states, before_gmm2_evt

    if bypass_mode in (1, 2):
        gmm1_output_size = _infer_gmm1_output_size(mlp_compute_input, quant_method)

    if bypass_mode == 1:
        hidden_states = debug_tensor(gmm1_output_size)
        _after_gmm1(mlp_compute_input)
        hidden_states = _unified_apply_activation(mlp_compute_input, hidden_states, quant_method)
        hidden_states, act_out_scale = quant_method.apply_act_quant(mlp_compute_input, hidden_states)
    elif bypass_mode == 2:
        hidden_states = quant_method.apply_gmm1(mlp_compute_input)
        _after_gmm1(mlp_compute_input)
        activation_output_size = gmm1_output_size // 2
        hidden_states = debug_tensor(activation_output_size)
        hidden_states, act_out_scale = quant_method.apply_act_quant(mlp_compute_input, hidden_states)
    else:
        if quant_method.supports_fused_activation(mlp_compute_input.activation):
            hidden_states, act_out_scale = quant_method.apply_gmm1_act_quant(mlp_compute_input)
        else:
            hidden_states = quant_method.apply_gmm1(mlp_compute_input)
            _after_gmm1(mlp_compute_input)
            hidden_states = _unified_apply_activation(mlp_compute_input, hidden_states, quant_method)
            hidden_states, act_out_scale = quant_method.apply_act_quant(mlp_compute_input, hidden_states)

    before_gmm2_evt = torch.npu.current_stream().record_event()
    if bypass_mode == 3:
        hidden_states = debug_tensor(final_hidden_size)
    else:
        hidden_states = quant_method.apply_gmm2(mlp_compute_input, hidden_states, act_out_scale)
    return hidden_states, before_gmm2_evt


def _after_gmm1(mlp_compute_input: MoEMlpComputeInput) -> None:
    if mlp_compute_input.shared_expert_schedule is not None:
        mlp_compute_input.shared_expert_schedule.after_gmm1()



def _infer_gmm1_output_size(mlp_compute_input: MoEMlpComputeInput, quant_method) -> int:
    layer = mlp_compute_input.layer
    moe_config = getattr(layer, "moe_config", None)
    intermediate_size = getattr(moe_config, "intermediate_size_per_partition", None)
    if isinstance(intermediate_size, int) and intermediate_size > 0:
        return 2 * intermediate_size
    input_size = mlp_compute_input.hidden_states.shape[-1]
    weight = _get_mlp_weight(mlp_compute_input, quant_method, "w1")
    return _infer_projection_output_size(weight, input_size, "GMM1")


def _get_mlp_weight(mlp_compute_input: MoEMlpComputeInput, quant_method, name: str) -> torch.Tensor | None:
    layer = mlp_compute_input.layer
    layer_names = ("w13_weight_list", "w13_weight") if name == "w1" else ("w2_weight_list", "w2_weight")
    if layer is not None:
        for layer_name in layer_names:
            weight = _first_tensor(getattr(layer, layer_name, None))
            if weight is not None:
                return weight

    weight = _first_tensor(getattr(mlp_compute_input.weights, name, None))
    if weight is not None:
        return weight

    if layer is not None:
        weights = quant_method.get_mlp_weights(layer)
        return _first_tensor(getattr(weights, name, None))
    return None


def _first_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _infer_projection_output_size(weight: torch.Tensor | None, input_size: int, projection_name: str) -> int:
    if weight is None or weight.ndim < 2:
        raise ValueError(f"Cannot infer {projection_name} output size for MoE bypass")

    dimension_a = int(weight.shape[-2])
    dimension_b = int(weight.shape[-1])
    if dimension_a == input_size and dimension_b != input_size:
        return dimension_b
    if dimension_b == input_size and dimension_a != input_size:
        return dimension_a
    raise ValueError(
        f"Cannot infer {projection_name} output size: weight shape={tuple(weight.shape)}, input_size={input_size}"
    )


def _unified_apply_activation(
    mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor, quant_method
) -> torch.Tensor:
    activation = mlp_compute_input.activation

    if activation == MoEActivation.SITU:
        beta = 1.0 if mlp_compute_input.activation_situ_beta is None else mlp_compute_input.activation_situ_beta
        alpha = mlp_compute_input.activation_situ_linear_beta
        if (
            get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP)
            and mlp_compute_input.quant.quant_type == QuantType.NONE
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and mlp_compute_input.group_list_type in (0, 1)
            and beta > 0
            and alpha is not None
            and alpha > 0
            and hasattr(torch.ops.custom, "npu_situ_and_mul_sparse")
        ):
            return torch.ops.custom.npu_situ_and_mul_sparse(
                hidden_states,
                cumsum_group_list(mlp_compute_input.group_list.to(torch.int64), mlp_compute_input.group_list_type, 1),
                beta=beta,
                alpha=alpha,
                high_precision=True,
            )
        hidden_states = _apply_situ(
            hidden_states,
            beta=1.0 if mlp_compute_input.activation_situ_beta is None else mlp_compute_input.activation_situ_beta,
            linear_beta=mlp_compute_input.activation_situ_linear_beta,
        )
    elif activation == MoEActivation.SWIGLUOAI:
        w1, _ = quant_method.get_mlp_weights(mlp_compute_input.layer)
        _, _, hidden_size = w1.shape
        hidden_states = AscendSwigluOAIAndMul.swiglu_oai_forward(hidden_states.view(-1, hidden_size))
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        hidden_states = DeviceOperator.clipped_swiglu(
            hidden_states,
            swiglu_limit=mlp_compute_input.swiglu_limit,
            swiglu_alpha=mlp_compute_input.swiglu_alpha,
            swiglu_beta=mlp_compute_input.swiglu_beta,
        )
    elif activation == MoEActivation.SWIGLUSTEP:
        hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=7.0)
    elif activation == MoEActivation.GELU:
        gate, up = hidden_states.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.gelu(gate) * up
    elif activation == MoEActivation.GELU_TANH:
        gate, up = hidden_states.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.gelu(gate, approximate="tanh") * up
    else:
        if mlp_compute_input.swiglu_limit > 0:
            gate, up = hidden_states.chunk(2, dim=-1)
            gate.clamp_(max=mlp_compute_input.swiglu_limit)
            up.clamp_(min=-mlp_compute_input.swiglu_limit, max=mlp_compute_input.swiglu_limit)
        hidden_states = torch_npu.npu_swiglu(hidden_states)

    return hidden_states


def _apply_situ(
    hidden_states: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Match SituAndMul.forward_native without constructing a CustomOp in forward."""
    gate, up = hidden_states.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(hidden_states.dtype)
