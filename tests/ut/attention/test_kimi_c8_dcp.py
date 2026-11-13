import pytest
import torch

from vllm_ascend.attention.kimi_c8_dcp import build_c8_dcp_query_plan, combine_dcp_lse
from vllm_ascend.attention.kimi_c8_graph import build_c8_graph_query_plan


@pytest.mark.parametrize(
    "interleave,length,queries", [(128, 129, 8), (128, 1025, 16), (1536, 1537, 8), (1536, 12289, 16)]
)
def test_q1_partition_and_lse_match_global_causal_attention(interleave, length, queries):
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(queries, 2, 4, generator=generator)
    k, v = [torch.randn(length, 4, generator=generator) for _ in range(2)]
    owners = (torch.arange(length) // interleave) % 8
    outputs, lses, totals = [], [], torch.zeros(queries, dtype=torch.int32)
    for rank in range(8):
        plan = build_c8_dcp_query_plan(
            [length],
            [0, queries],
            torch.tensor([[7]], dtype=torch.int32),
            world_size=8,
            rank=rank,
            interleave_size=interleave,
        )
        out, lse = torch.full((queries, 2, 4), torch.nan), torch.full((queries, 2), torch.inf)
        local_k, local_v = k[owners == rank], v[owners == rank]
        for token, count in enumerate(plan.cache_seqlens.tolist()):
            assert count == int(((torch.arange(length) < length - queries + token + 1) & (owners == rank)).sum())
            if count:
                scores = q[token] @ local_k[:count].T
                out[token], lse[token] = scores.softmax(-1) @ local_v[:count], scores.logsumexp(-1)
        totals += plan.cache_seqlens
        outputs.append(out)
        lses.append(lse.T)  # Real quant FA returns heads before tokens.
    assert totals.tolist() == list(range(length - queries + 1, length + 1))
    actual, _ = combine_dcp_lse(torch.stack(outputs), torch.stack(lses))
    expected = torch.stack(
        [(q[t] @ k[: length - queries + t + 1].T).softmax(-1) @ v[: length - queries + t + 1] for t in range(queries)]
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("heads", [2, 3])
@pytest.mark.parametrize("empty_lse", [torch.inf, -torch.inf, torch.nan])
def test_nt_tn_layout_and_nonfinite_empty_ranks(heads, empty_lse):
    out = torch.arange(4 * heads * 2, dtype=torch.float32).reshape(2, 2, heads, 2)
    weights = torch.arange(1, 4 * heads + 1, dtype=torch.float32).reshape(2, 2, heads)
    expected = (out * weights[..., None]).sum(0) / weights.sum(0)[..., None]
    for layout, lse in (("TN", weights.log()), ("NT", weights.log().transpose(-1, -2))):
        actual, _ = combine_dcp_lse(out, lse, lse_layout=layout)
        torch.testing.assert_close(actual, expected)
    out[1] = torch.nan
    lse = torch.zeros(2, 2, heads)
    lse[1] = empty_lse
    actual, _ = combine_dcp_lse(out, lse, lse_layout="TN")
    torch.testing.assert_close(actual, out[0])
    empty, empty_lse = combine_dcp_lse(out[1:], lse[1:], lse_layout="TN")
    assert empty.eq(0).all() and empty_lse.isneginf().all()


def test_eager_and_graph_padding_never_exposes_stale_rows():
    table = torch.tensor([[5], [7], [9]], dtype=torch.int32)
    eager = build_c8_dcp_query_plan(
        [0, 1537, 0], [0, 0, 2, 2], table, world_size=8, rank=7, interleave_size=1536, num_tokens=4
    )
    graph = build_c8_graph_query_plan(
        torch.tensor([1535, 1536, 0, 0]),
        torch.tensor([0, 0, 2, 2]),
        torch.tensor([0, 1537, 0]),
        table,
        world_size=8,
        rank=7,
        interleave_size=1536,
        max_model_len=16384,
    )
    for plan in (eager, graph):
        assert plan.cache_seqlens.tolist() == [0] * 4
        assert plan.block_table[:2].tolist() == [[7], [7]]
    for offsets in ([1, 2], [0]):
        with pytest.raises(ValueError):
            build_c8_dcp_query_plan([8], offsets, table[:1], world_size=8, rank=0, interleave_size=1536)
