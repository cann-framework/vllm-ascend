# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.attention.kimi_mla_dp import DecodePartition, build_compute_decode, dp_to_tp_output


@pytest.mark.parametrize(
    "total,local,lengths,world,rank,message",
    [
        (248, 62, [8] * 30, 4, 0, "aligned"),
        (32, 8, [1, 8, 1, 1], 4, 0, "uniform"),
        (32, 7, [1] * 25, 4, 0, "total_tokens"),
        (1, 0, [1], 4, 0, "positive local_tokens"),
        (32, 8, [1] * 33, 4, 0, "exceed"),
        (32, 8, [0] * 8, 4, 0, "positive integer"),
        (32, 8, [], 4, 0, "empty"),
        (0, 0, [1], 0, 0, "world_size"),
        (32, 8, [1] * 32, 4, 4, "rank"),
    ],
)
def test_rejects_invalid_layout(total, local, lengths, world, rank, message):
    with pytest.raises(ValueError, match=message):
        DecodePartition.from_sp_layout(total, local, lengths, world, rank)


def _gloo_round_trip(rank, world, rendezvous):
    # Spawned workers have no UT conftest; load the pure CPU module directly.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/kimi_mla_dp.py"
    spec = importlib.util.spec_from_file_location("kimi_mla_dp_cpu", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=30))
    try:
        for query_len, batch, total in [(1, 25, 32), (8, 25, 256)]:
            part = module.DecodePartition.from_sp_layout(
                total, total // world, [query_len] * batch, world, rank, allow_empty=True
            )
            channels = 192 // world
            golden = torch.arange(total * 192).reshape(total, 192)
            golden[query_len * batch :] = 0
            local = golden[rank * part.capacity : (rank + 1) * part.capacity].clone()
            output = torch.empty(total, channels, dtype=local.dtype)
            actual = module.dp_to_tp_output(local, part, dist.group.WORLD, output=output)
            packed = local.reshape(part.capacity, world, channels).permute(1, 0, 2).contiguous()
            packed_result = module.dp_to_tp_output(packed, part, dist.group.WORLD)
            torch.testing.assert_close(packed_result, golden[:, rank * channels : (rank + 1) * channels])
            assert actual is output
            torch.testing.assert_close(actual, golden[:, rank * channels : (rank + 1) * channels])
            gathered = [torch.empty_like(actual) for _ in range(world)]
            dist.all_gather(gathered, actual)
            restored = torch.cat(gathered, dim=1)
            torch.testing.assert_close(restored, golden)
            torch.testing.assert_close(restored[rank * part.capacity : (rank + 1) * part.capacity], local)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [4, 8])
def test_gloo_pack_unpack_round_trip(tmp_path, world):
    mp.spawn(_gloo_round_trip, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)


def _compute_metadata_case(q=4, batch=25):
    total, requests = 32 * q, 32
    table = torch.arange(requests * 6).reshape(requests, 6)[:, ::2]
    slots = torch.arange(total, dtype=torch.int64) + 700
    seqs = list(range(101, 101 + batch)) + [0] * (requests - batch)
    positions = torch.arange(total)
    cos = torch.arange(batch * q * 2).reshape(batch * q, 1, 1, 2)
    decode = SimpleNamespace(
        block_table=table,
        seq_lens_list=seqs,
        seq_lens=torch.tensor(seqs),
        input_positions=positions[: batch * q],
        cos=cos,
        sin=-cos,
        actual_seq_lengths_q=[q * (i + 1) for i in range(requests)],
    )
    metadata = SimpleNamespace(
        num_input_tokens=total,
        num_actual_tokens=batch * q,
        num_decode_tokens=batch * q,
        num_decodes=requests,
        num_prefills=0,
        slot_mapping=slots[: batch * q],
        decode=decode,
    )
    common = SimpleNamespace(
        kimi_mla_num_actual_reqs=batch,
        num_reqs=requests,
        query_start_loc_cpu=torch.arange(requests + 1) * q,
        positions=positions,
        slot_mapping=slots,
        block_table_tensor=table,
        kimi_mla_full_graph=True,
    )
    return metadata, common


@pytest.mark.parametrize("q", [1, 8])
@pytest.mark.parametrize("rank", range(4))
def test_compute_metadata_preserves_request_rows_and_padding(q, rank):
    metadata, common = _compute_metadata_case(q)
    compute = build_compute_decode(metadata, common, 4, rank)
    local, part = compute.local_metadata, compute.partition
    start, stop = rank * 8, (rank + 1) * 8
    token_start, token_stop = start * q, stop * q
    assert local is not metadata and local.decode is not metadata.decode
    assert (local.num_input_tokens, local.num_decode_tokens, local.num_decodes) == (8 * q, 8 * q, 8)
    assert local.num_actual_tokens == min(8 * q, 25 * q - token_start)
    assert local.num_prefills == 0 and local.prefill is None
    assert local.query_lens == [q] * 8
    assert local.decode.actual_seq_lengths_q == [q * (i + 1) for i in range(8)]
    assert local.query_start_loc.tolist() == [q * i for i in range(9)]
    assert local.decode.seq_lens_list == (list(range(101, 126)) + [0] * 7)[start:stop]
    torch.testing.assert_close(local.decode.block_table, common.block_table_tensor[start:stop])
    assert local.decode.block_table.data_ptr() == common.block_table_tensor[start:].data_ptr()
    assert local.decode.block_table.stride() == common.block_table_tensor.stride()
    assert metadata.decode.block_table.shape == (32, 3)
    assert metadata.slot_mapping.shape == (32 * q,)
    assert metadata.slot_mapping.data_ptr() == common.slot_mapping.data_ptr()
    assert metadata.slot_mapping[25 * q :].tolist() == [-1] * (7 * q)
    assert local.slot_mapping.data_ptr() == metadata.slot_mapping[token_start:].data_ptr()
    assert local.slot_mapping.tolist() == metadata.slot_mapping[token_start:token_stop].tolist()
    assert local.decode.cos.shape == (8 * q, 1, 1, 2)
    real = part.valid_tokens
    torch.testing.assert_close(local.decode.cos[:real], metadata.decode.cos[token_start : token_start + real])
    assert torch.count_nonzero(local.decode.cos[real:]) == 0
    torch.testing.assert_close(local.decode.sin, -local.decode.cos)


@pytest.mark.parametrize("rank", range(4))
def test_compute_metadata_eager_rejects_real_batch17_not_padded32(rank):
    metadata, common = _compute_metadata_case(1, 17)
    common.kimi_mla_full_graph = False
    with pytest.raises(ValueError, match="empty") as error:
        build_compute_decode(metadata, common, 4, rank)
    assert all(value in str(error.value) for value in ("B=17", "q=1", "TP=4", "T=32", "S=8"))


@pytest.mark.parametrize(
    "q,batch,total,counts",
    [
        (1, 1, 4, (1, 0, 0, 0)),
        (1, 17, 32, (8, 8, 1, 0)),
        (4, 1, 16, (4, 0, 0, 0)),
    ],
)
@pytest.mark.parametrize("rank", range(4))
def test_compute_metadata_graph_accepts_fixed_capacity_empty_ranks(q, batch, total, counts, rank):
    metadata, common = _compute_metadata_case(q, batch)
    metadata.num_input_tokens = total
    compute = build_compute_decode(metadata, common, 4, rank)
    local, part = compute.local_metadata, compute.partition
    capacity, requests = total // 4, total // (4 * q)
    assert part.valid_tokens == counts[rank]
    assert local.num_actual_tokens == counts[rank]
    assert local.num_decode_tokens == capacity and local.num_decodes == requests
    assert local.decode.actual_seq_lengths_q == [q * (i + 1) for i in range(requests)]
    table = common.block_table_tensor[rank * requests : (rank + 1) * requests]
    torch.testing.assert_close(local.decode.block_table, table)
    assert local.decode.block_table.data_ptr() == table.data_ptr()
    if counts[rank] == 0:
        assert local.decode.seq_lens_list == [0] * requests
        assert local.seq_lens.tolist() == [0] * requests
        assert local.slot_mapping.tolist() == [-1] * capacity


def test_compute_metadata_rejects_ragged_real_query_prefix():
    metadata, common = _compute_metadata_case()
    common.query_start_loc_cpu[1] = 3
    with pytest.raises(ValueError, match="uniform"):
        build_compute_decode(metadata, common, 4, 0)


def test_compute_metadata_rejects_temporary_fullgraph_slot_padding():
    metadata, common = _compute_metadata_case()
    common.slot_mapping = metadata.slot_mapping
    with pytest.raises(ValueError, match="stable.*slot"):
        build_compute_decode(metadata, common, 4, 0)


@pytest.mark.parametrize("caller_output,raises", [(False, False), (True, False), (False, True)])
def test_packed_a2a_alias_and_error_drain(monkeypatch, caller_output, raises):
    world = 8
    capacity, channels = 2, 3
    partition = DecodePartition.from_sp_layout(world * capacity, capacity, [1] * world * capacity, world, 0)
    packed = torch.arange(world * capacity * channels).reshape(world, capacity, channels)
    received = []
    events = []

    def submit(receive, send, **kwargs):
        assert send is packed
        receive.copy_(packed + 100)
        received.append(receive)
        events.append("submit")
        return SimpleNamespace(wait=lambda: events.append("wait"))

    monkeypatch.setattr(torch.distributed, "all_to_all_single", submit)
    destination = torch.empty(world * capacity, channels, dtype=packed.dtype) if caller_output else None

    def hook():
        events.append("hook")
        if raises:
            raise RuntimeError("hook failed")

    if raises:
        with pytest.raises(RuntimeError, match="hook failed"):
            dp_to_tp_output(packed, partition, None, destination, hook)
        assert events == ["submit", "hook", "wait"]
        return
    result = dp_to_tp_output(packed, partition, None, destination, hook)
    torch.testing.assert_close(result, (packed + 100).reshape(world * capacity, channels))
    assert events == ["submit", "hook", "wait"]
    if destination is None:
        assert result.data_ptr() == received[0].data_ptr()
    else:
        assert result is destination


def test_noncontiguous_packed_send_is_rejected():
    partition = DecodePartition.from_sp_layout(16, 4, [1] * 16, 4, 0)
    packed = torch.ones(4, 4, 3).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        dp_to_tp_output(packed, partition, None)
