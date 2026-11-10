# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny NPU collective used to verify Lite Actions topology without model loading."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - registers the HCCL backend


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://", timeout=timedelta(seconds=45))
    try:
        value = torch.tensor([rank], dtype=torch.int32, device=f"npu:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size - 1) // 2
        if int(value.cpu().item()) != expected_sum:
            raise RuntimeError(f"DP all_reduce mismatch: expected {expected_sum}, got {int(value.cpu().item())}")
        dist.barrier()
        if rank == 0:
            print(f"DISPATCH_SMOKE_PASS world_size={world_size} all_reduce_sum={expected_sum}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
