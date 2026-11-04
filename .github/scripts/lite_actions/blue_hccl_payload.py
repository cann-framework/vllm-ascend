"""Restored real HCCL payload for the validation-only Blue acceptance PR.

The trusted runner still owns topology, expected sums, timeouts and evidence.
This function is loaded only in leased vllm_hccl_smoke containers from the
verified Head/Base merge source. Keep this acceptance PR unmerged.
"""

PAYLOAD_API_VERSION = 1


def all_reduce(tensor):
    import torch.distributed as dist

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
