# SPDX-License-Identifier: Apache-2.0
"""Kimi decode MoE fusion scopes."""

from contextlib import contextmanager

import torch

from vllm_ascend.ascend_forward_context import _is_kimi_moe_model
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


def use_kimi_moe_superkernel(vllm_config, ascend_config) -> bool:
    """Reuse 0.30's switch; unsupported paths retain its generic graph policy.

    Include the target-derived draft config here so callers can disable both
    named scopes and the whole-model fallback for that draft.
    """
    return (
        ascend_config.ascend_compilation_config.enable_super_kernel
        and ascend_config.multistream_overlap_shared_expert
        and _is_kimi_moe_model(vllm_config)
        and get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP)
    )


@contextmanager
def moe_superkernel_scope(name: str | None, enabled: bool):
    """None excludes shared communication from the enclosing named scope."""
    if not enabled:
        yield
        return
    torch.npu.super_kernel_scope_begin(name)
    try:
        yield
    finally:
        torch.npu.super_kernel_scope_end(name)


def moe_superkernel_options() -> dict:
    return {"dcci_disable_on_kernel": [".*"], "auto_op_parallel": 1}
