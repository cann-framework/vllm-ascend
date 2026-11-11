# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device checks for the current BF16 DSL prolog, including DCP cache staging.

Exercise fixed-address ACL graph replay with changing hidden states and slots.
This tests prolog and the DCP cache copy, without emulating distributed attention.
"""

import math

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("ops.mla_prolog_v4")
pytest.importorskip("cann_ops_transformer")

from vllm_ascend.attention.mla_v1 import AscendMLAImpl  # noqa: E402
from vllm_ascend.device.device_config import check_ascend_device_type  # noqa: E402
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile  # noqa: E402
from vllm_ascend.ops.flash_mla_pa_nz import mla_prolog_no_rope, pa_nz_tile_indices  # noqa: E402

BLOCK, HIDDEN, Q_RANK, KV_RANK, HEADS, Q_DIM, AUX_DIM = 128, 1024, 640, 512, 2, 128, 64
WIDTH, TILE = KV_RANK + AUX_DIM, 16
EPS, SENTINEL = 3e-6, -17.0


def reference(x, weights, gamma):
    def norm(value, scale):
        value = value.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS) * scale).bfloat16()

    qa = norm(x @ weights[0], gamma[0])
    q = (qa @ weights[1]).view(-1, HEADS, Q_DIM + AUX_DIM)
    q_nope = torch.bmm(q[..., :Q_DIM].transpose(0, 1), weights[3]).transpose(0, 1)
    q = torch.cat((q_nope, q[..., Q_DIM:]), dim=-1)
    kv = x @ weights[2]
    kv = torch.cat((norm(kv[:, :KV_RANK], gamma[1]), kv[:, KV_RANK:]), dim=-1)
    return q, kv


def check_cache(cache, slots, kv):
    # Build an independent physical PA_NZ reference from the token slots.
    expected = torch.full(cache.shape, SENTINEL, dtype=torch.bfloat16)
    physical = expected.view(cache.shape[0], WIDTH // TILE, BLOCK, TILE)
    rows = kv.reshape(-1, WIDTH // TILE, TILE)
    for token, slot in enumerate(slots.cpu().tolist()):
        physical[slot // BLOCK, :, slot % BLOCK, :] = rows[token]
    actual = cache.cpu()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.04, atol=0.02)
    untouched = expected == SENTINEL
    assert torch.equal(actual[untouched], expected[untouched])


@pytest.mark.parametrize("tokens", [3, 129])
@pytest.mark.parametrize("dcp_current", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
@torch.inference_mode()
def test_bf16_pa_nz_prolog_cache_and_replay(tokens, dcp_current, graph_mode):
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    check_ascend_device_type()
    if not get_current_hardware_profile().supports(HardwareCapability.MLA_FLASH):
        pytest.skip("requires A5")
    torch.manual_seed(17)
    device = torch.device("npu")
    shapes = ((HIDDEN, Q_RANK), (Q_RANK, HEADS * (Q_DIM + AUX_DIM)), (HIDDEN, WIDTH), (HEADS, Q_DIM, KV_RANK))
    weights = [(torch.randn(shape, dtype=torch.float32) / math.sqrt(shape[-2])).bfloat16() for shape in shapes]
    nz_weights = [torch_npu.npu_format_cast(w.to(device), 29) for w in weights[:3]]
    wkb = weights[3].to(device)
    gamma = (torch.ones(Q_RANK), torch.ones(KV_RANK))
    gamma_device = [g.to(device) for g in gamma]
    x = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
    history = torch.empty((4, BLOCK, 1, WIDTH), dtype=torch.bfloat16, device=device)
    current = torch.empty_like(history)
    slots = torch.empty(tokens, dtype=torch.int64, device=device)
    current_slots = torch.arange(tokens, dtype=torch.int64, device=device)
    src_indices = pa_nz_tile_indices(current_slots, BLOCK, WIDTH // TILE)
    dst_indices = torch.empty_like(src_indices)

    def step():
        q, _, _ = mla_prolog_no_rope(
            x,
            nz_weights[0],
            nz_weights[1],
            wkb,
            nz_weights[2],
            *gamma_device,
            current if dcp_current else history,
            current_slots if dcp_current else slots,
            EPS,
        )
        if dcp_current:
            AscendMLAImpl._copy_pa_nz_rows(current, history, src_indices, dst_indices)
        return q

    graph = None
    for iteration in range(3):
        hidden = (torch.randn((tokens, HIDDEN)) * 0.1).bfloat16()
        x.copy_(hidden)
        slots.copy_(torch.randperm(3 * BLOCK)[:tokens] + BLOCK)
        dst_indices.copy_(pa_nz_tile_indices(slots, BLOCK, WIDTH // TILE))
        history.fill_(SENTINEL)
        current.fill_(SENTINEL)
        if graph_mode and iteration == 0:
            step()  # Compile/validate the DSL variant outside capture.
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                query = step()
        if graph_mode:
            graph.replay()
        else:
            query = step()
        torch.npu.synchronize()
        expected_q, expected_kv = reference(hidden, weights, gamma)
        torch.testing.assert_close(query.cpu().float(), expected_q.float(), rtol=0.04, atol=0.02)
        check_cache(history, slots, expected_kv)
        if dcp_current:
            check_cache(current, current_slots, expected_kv)
