# SPDX-License-Identifier: Apache-2.0
"""Compile Kimi's actual SP imports once, then execute smaller V2 shapes."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch.fx.experimental.proxy_tensor import make_fx

ROOT = Path(__file__).resolve().parents[3]


class TestKimiV2SequenceParallel(unittest.TestCase):
    def test_compiled_shards_masks_and_collectives_match_eager_for_all_ranks(self):
        rank = [0]
        calls = []
        operators = []

        def register(op_name, op_func, fake_impl, **kwargs):
            operator = torch.library.custom_op(f"kimi_v2_sp_test::{op_name}", mutates_args=())(op_func)
            operator.register_fake(fake_impl)
            operators.append(operator)

        class TorchProxy:
            ops = NS(vllm=torch.ops.kimi_v2_sp_test)

            def __getattr__(self, name):
                return getattr(torch, name)

        def collective(name, x):
            calls.append(name)
            if name == "custom_all_gather":
                return x.repeat(8, 1)
            return x.chunk(8)[rank[0]].clone()

        ns = dict(
            torch=TorchProxy(),
            F=torch.nn.functional,
            get_tensor_model_parallel_world_size=lambda: 8,
            get_tensor_model_parallel_rank=lambda: rank[0],
            get_tp_group=lambda: NS(
                device_communicator=NS(
                    custom_all_gather=lambda x: collective("custom_all_gather", x),
                    custom_reduce_scatter=lambda x: collective("custom_reduce_scatter", x),
                )
            ),
            tensor_model_parallel_all_gather=lambda x, dim: collective("fallback_ag", x),
            tensor_model_parallel_reduce_scatter=lambda x, dim: collective("fallback_rs", x),
            cdiv=lambda a, b: (a + b - 1) // b,
            direct_register_custom_op=register,
        )
        path = ROOT / "vllm_ascend/models/common/ops/sequence_parallel.py"
        nodes = [n for n in ast.parse(path.read_text()).body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), ns)

        # Resolve the production Kimi import, so this test fails when the
        # model accidentally switches back to the trace-sensitive upstream SP.
        model_path = ROOT / "vllm_ascend/models/kimi_k3.py"
        imports = [
            n
            for n in ast.parse(model_path.read_text()).body
            if isinstance(n, ast.ImportFrom) and n.module.endswith("ops.sequence_parallel")
        ]
        model_ns = {}
        with patch.dict(sys.modules, {"vllm_ascend.models.common.ops.sequence_parallel": NS(**ns)}):
            exec(compile(ast.Module(body=imports, type_ignores=[]), str(model_path), "exec"), model_ns)

        def forward(x, mask):
            return model_ns["sp_shard"](x), model_ns["sp_padding_mask"](mask, x)

        compiled = make_fx(forward, tracing_mode="symbolic")(
            torch.ones(16, 12 * 128), torch.zeros(16, dtype=torch.bool)
        )
        for current_rank in range(8):
            rank[0] = current_rank
            for tokens in (16, 12, 8, 4, 2, 1):
                with self.subTest(rank=current_rank, tokens=tokens):
                    x = torch.arange(tokens * 12 * 128).reshape(tokens, 12 * 128).float()
                    mask = torch.zeros(tokens, dtype=torch.bool)
                    actual, actual_mask = compiled(x, mask)
                    padded = torch.nn.functional.pad(x, (0, 0, 0, (-tokens) % 8))
                    expected = padded.chunk(8)[current_rank]
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual, forward(x, mask)[0])
                    torch.testing.assert_close(actual_mask, forward(x, mask)[1])
                    self.assertEqual(actual.shape[0], (tokens + 7) // 8)
                    self.assertGreater(actual.numel(), 0)
                    padded_mask = torch.nn.functional.pad(mask, (0, (-tokens) % 8), value=True)
                    torch.testing.assert_close(actual_mask, padded_mask.chunk(8)[current_rank])
                    torch.testing.assert_close(model_ns["sp_reduce_scatter"](x), expected)
                    torch.testing.assert_close(model_ns["sp_all_gather"](actual), actual.repeat(8, 1))
        self.assertNotIn("fallback_ag", calls)
        self.assertNotIn("fallback_rs", calls)


if __name__ == "__main__":
    unittest.main()
