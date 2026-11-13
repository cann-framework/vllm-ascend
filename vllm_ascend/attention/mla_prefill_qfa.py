# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-absorbed MLA prefill QFA, with schedules owned by step metadata."""

import torch
import torch_npu


class MlaPrefillQfa:
    """Share schedules across layers, but never cache layer-specific Q/K/V."""

    def __init__(self, num_heads: int, qk_dim: int, value_dim: int, scale: float, device: torch.device):
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.value_dim = value_dim
        self.scale = scale
        self.device = device
        self._plans: dict = {}
        self._mask = None
        self._p_scale = None

    @staticmethod
    def supports(kv_ends: list[int]) -> bool:
        # Keep irregular/empty segments on FIA. V uses token-axis MX groups;
        # each sequence must start at a 64-token scale-pair boundary.
        return all(end > start and (end - start) % 64 == 0 for start, end in zip([0, *kv_ends], kv_ends))

    @staticmethod
    def prepare_query(q_nope: torch.Tensor, q_pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_npu.npu_dynamic_mx_quant(
            torch.cat((q_nope, q_pe), dim=-1), dst_type=torch.float8_e4m3fn, block_size=32
        )

    def _plan(self, q_ends: list[int], kv_ends: list[int], mask_mode: int) -> dict:
        from cann_ops_transformer.ops import quant_flash_attn_metadata

        key = (tuple(q_ends), tuple(kv_ends), mask_mode)
        if key not in self._plans:
            q_bounds, kv_bounds = [0, *q_ends], [0, *kv_ends]
            kwargs = dict(
                cu_seqlens_q=torch.tensor(q_bounds, dtype=torch.int32, device=self.device),
                cu_seqlens_kv=torch.tensor(kv_bounds, dtype=torch.int32, device=self.device),
                max_seqlen_q=max(b - a for a, b in zip(q_bounds, q_bounds[1:])),
                max_seqlen_kv=max(b - a for a, b in zip(kv_bounds, kv_bounds[1:])),
                quant_mode=1,
                mask_mode=mask_mode,
                win_left=-1,
                win_right=-1,
                layout_q="TND",
                layout_q_descale="TND",
                layout_kv="TND",
                layout_out="TND",
            )
            metadata = quant_flash_attn_metadata(
                self.num_heads,
                self.num_heads,
                self.qk_dim,
                head_dim_v=self.value_dim,
                is_grad_enabled=False,
                **kwargs,
            )
            self._plans[key] = dict(metadata=metadata, **kwargs)
        return self._plans[key]

    def __call__(self, query, k_nope, k_pe, value, q_ends, kv_ends, mask_mode, attn_mask):
        from cann_ops_transformer.ops import quant_flash_attn

        key, key_scale = torch_npu.npu_dynamic_mx_quant(
            torch.cat((k_nope, k_pe), dim=-1), dst_type=torch.float8_e4m3fn, block_size=32
        )
        # Q/K quantize head channels; V quantizes the token axis for P @ V.
        value, value_scale = torch_npu.npu_dynamic_mx_quant(
            value.permute(1, 2, 0).contiguous(), dst_type=torch.float8_e4m3fn, block_size=32
        )
        value = value.view(torch.uint8).permute(2, 0, 1).contiguous().view(torch.float8_e4m3fn)
        value_scale = value_scale.view(torch.uint8).permute(2, 0, 1, 3).contiguous()
        if self._p_scale is None:
            self._p_scale = torch.ones(1, dtype=torch.float32, device=self.device)
        if mask_mode and self._mask is None:
            self._mask = attn_mask.to(torch.int8)
        q, query_scale = query
        # TND QFA returns LSE as [heads, tokens], as does the FIA merge input.
        return quant_flash_attn(
            q,
            key,
            value,
            query_scale.view(torch.float8_e8m0fnu),
            key_scale.view(torch.float8_e8m0fnu),
            value_scale.view(torch.float8_e8m0fnu),
            **self._plan(q_ends, kv_ends, mask_mode),
            attn_mask=self._mask if mask_mode else None,
            p_scale=self._p_scale,
            softmax_scale=self.scale,
            return_softmax_lse=True,
        )
