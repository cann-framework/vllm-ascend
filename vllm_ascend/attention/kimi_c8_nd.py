from dataclasses import dataclass

import torch

from vllm_ascend.attention.kimi_c8_static import K3_MLA_CACHE_WIDTH


@dataclass(frozen=True)
class C8NDGatherPlan:
    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    row_ranges: tuple[tuple[int, int], ...]
    total_rows: int


def plan_c8_nd_gather(starts: list[int], lengths: list[int], block_size: int) -> C8NDGatherPlan:
    if block_size <= 0 or len(starts) != len(lengths):
        raise ValueError("C8 ND gather requires matching starts and lengths and a positive block size")
    aligned_starts = []
    gather_lengths = []
    row_ranges = []
    total_rows = 0
    for start, length in zip(starts, lengths):
        if start < 0 or length < 0:
            raise ValueError("C8 ND gather starts and lengths must be non-negative")
        aligned_start = start // block_size * block_size
        skip = start - aligned_start if length else 0
        gather_length = skip + length
        aligned_starts.append(aligned_start)
        gather_lengths.append(gather_length)
        row_ranges.append((total_rows + skip, total_rows + gather_length))
        total_rows += gather_length
    return C8NDGatherPlan(tuple(aligned_starts), tuple(gather_lengths), tuple(row_ranges), total_rows)


@dataclass(frozen=True)
class C8NDHistory:
    table: torch.Tensor
    plans: tuple[C8NDGatherPlan, ...]
    starts: tuple[torch.Tensor, ...]
    lengths: tuple[torch.Tensor, ...]

    @classmethod
    def prepare(cls, cache, table, chunk_lengths, chunk_starts):
        if chunk_lengths.device.type != "cpu" or chunk_starts.device.type != "cpu":
            raise RuntimeError("C8 ND history planning requires CPU-resident chunk lengths and starts")
        plans = tuple(
            plan_c8_nd_gather(start.tolist(), length.tolist(), cache.shape[3])
            for length, start in zip(chunk_lengths, chunk_starts)
        )
        starts = tuple(torch.tensor(plan.starts, dtype=torch.int32, device=cache.device) for plan in plans)
        lengths = tuple(torch.tensor(plan.lengths, dtype=torch.int32, device=cache.device) for plan in plans)
        return cls(table, plans, starts, lengths)

    def read(self, cache, chunk, scale, *, dtype):
        from vllm_ascend.device.device_op import DeviceOperator

        plan = self.plans[chunk]
        output = torch.empty((plan.total_rows, 1, K3_MLA_CACHE_WIDTH), dtype=torch.float8_e4m3fn, device=cache.device)
        if not plan.total_rows:
            return output.to(dtype)
        unused = torch.empty_like(output)
        nd_cache = cache.view(torch.float8_e4m3fn).view(cache.shape[0], cache.shape[3], 1, K3_MLA_CACHE_WIDTH)
        DeviceOperator.kv_cache_load(
            nd_cache, nd_cache, self.table, self.lengths[chunk], self.starts[chunk], output, unused
        )
        rows = torch.cat([output[start:end] for start, end in plan.row_ranges], dim=0)
        return rows.float().mul(scale).to(dtype)
