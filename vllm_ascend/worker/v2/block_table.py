# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
import torch
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_ascend.ops.triton.v2.block_table.compute_slot_mappings import (
    _compute_slot_mappings_kernel,
)


class AscendBlockTables(BlockTables):
    """Block table for Ascend NPUs."""

    block_sizes_tensor: torch.Tensor
    kernel_block_sizes_tensor: torch.Tensor

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int] | None = None,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
        slot_mapping_enabled: list[bool] | None = None,
        cp_sizes: list[int] | None = None,
        cp_ranks: list[int] | None = None,
        cp_interleaves: list[int] | None = None,
    ):
        if kernel_block_sizes is None:
            kernel_block_sizes = block_sizes
        super().__init__(
            block_sizes,
            max_num_reqs,
            max_num_batched_tokens,
            max_num_blocks_per_group,
            device,
            kernel_block_sizes,
            cp_size,
            cp_rank,
            cp_interleave,
            slot_mapping_enabled=slot_mapping_enabled,
        )
        self._triton_block_size = 1024
        # kernel_block_sizes determine the number of block-table entries
        # touched by one token tile. Use the smallest kernel block size to form
        # one safe constexpr window for all groups, without staging a whole
        # row.
        min_kernel_block_size = min(kernel_block_sizes)
        window_size = (self._triton_block_size + min_kernel_block_size - 1) // min_kernel_block_size + 1
        self._block_table_window_size = triton.next_power_of_2(window_size)
        num_groups = self.num_kv_cache_groups
        self.cp_sizes = tuple(cp_sizes) if cp_sizes is not None else (cp_size,) * num_groups
        self.cp_ranks = tuple(cp_ranks) if cp_ranks is not None else (cp_rank,) * num_groups
        self.cp_interleaves = tuple(cp_interleaves) if cp_interleaves is not None else (cp_interleave,) * num_groups
        if not (len(self.cp_sizes) == len(self.cp_ranks) == len(self.cp_interleaves) == num_groups):
            raise ValueError("per-group CP settings must match the number of KV cache groups")
        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.slot_mappings
        # vllm-ascend' reshape_and_cache function requires slot_mappings to be int32.
        # so we need to redefine slot_mappings to be int32.
        self.slot_mappings: torch.Tensor = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        slot_mappings = self.slot_mappings if out is None else out
        slot_mapping_enabled = getattr(self, "slot_mapping_enabled", getattr(self, "_slot_mapping_enabled", None))
        cp_settings = list(zip(self.cp_sizes, self.cp_ranks, self.cp_interleaves))
        if len(set(cp_settings)) == 1:
            launches = [
                (
                    None,
                    self.block_table_ptrs,
                    self.block_table_strides,
                    self.block_sizes_tensor,
                    self.kernel_block_sizes_tensor,
                    slot_mappings,
                    slot_mapping_enabled,
                )
            ]
        else:
            # CP values are constexprs in the Triton kernel, so mixed MLA-DCP
            # and replicated Window-GQA groups require one launch per group.
            launches = [
                (
                    group_id,
                    self.block_table_ptrs[group_id : group_id + 1],
                    self.block_table_strides[group_id : group_id + 1],
                    self.block_sizes_tensor[group_id : group_id + 1],
                    self.kernel_block_sizes_tensor[group_id : group_id + 1],
                    slot_mappings[group_id : group_id + 1],
                    None if slot_mapping_enabled is None else slot_mapping_enabled[group_id : group_id + 1],
                )
                for group_id in range(num_groups)
            ]
        for launch in launches:
            (
                group_id,
                block_table_ptrs,
                block_table_strides,
                block_sizes,
                kernel_block_sizes,
                group_slots,
                enabled,
            ) = launch
            if group_id is None:
                cp_size, cp_rank, cp_interleave = cp_settings[0]
                grid_groups = num_groups
            else:
                cp_size, cp_rank, cp_interleave = cp_settings[group_id]
                grid_groups = 1
            _compute_slot_mappings_kernel[(grid_groups, num_reqs + 1)](
                group_slots.shape[1],
                idx_mapping,
                query_start_loc,
                positions,
                block_table_ptrs,
                block_table_strides,
                block_sizes,
                kernel_block_sizes,
                group_slots,
                group_slots.stride(0),
                cp_rank,
                CP_SIZE=cp_size,
                CP_INTERLEAVE=cp_interleave,
                PAD_ID=PAD_SLOT_ID,
                TRITON_BLOCK_SIZE=self._triton_block_size,
                BLOCK_TABLE_WINDOW_SIZE=self._block_table_window_size,
                slot_mapping_enabled=enabled,
                HAS_SLOT_MAPPING_ENABLED=True,
            )
        return slot_mappings[:, :num_tokens_padded]
