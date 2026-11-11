"""CPU-only regression for the GitHub controller's repository rename.

Run with python3 -m unittest discover -s .github/scripts/lite_actions/tests -v.
No GitHub requests, worker deployment, or accelerator processes are started.
"""

import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
REPO = "cann-framework/vllm-ascend"
OLD_REPO = "torchtitan-npu/vllm-ascend"
SHA = "b" * 40


def load_control(name):
    path = ROOT / ".github/scripts/lite_actions" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_control("prepare_request")
waiter = load_control("wait_result")


class RepositoryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "pull_request": {
                "base": {"ref": "main", "repo": {"full_name": REPO}},
                "head": {"sha": SHA, "repo": {"full_name": "depeng1994/vllm-ascend"}},
                "user": {"login": "depeng1994"},
                "state": "open",
                "draft": False,
                "labels": [],
            }
        }
        self.item = {
            "case_id": "case_000_dispatch_smoke_a5_8p",
            "test_id": "dispatch_smoke_a5_8p",
            "worker": "yellow",
            "target_pool": "a5_shared_yellow",
            "docker": "vllm-kimi-wangjinyin",
            "resources": {"nodes": 1, "devices_per_node": 8},
        }
        self.result = {
            **self.item,
            "schema": 3,
            "repo": REPO,
            "workflow": prepare.WORKFLOW,
            "run_id": 100,
            "attempt": 2,
            "sha": SHA,
            "plan_digest": "d" * 64,
            "status": "PASS",
            "exit_code": 0,
            "cleanup_verified": True,
            "last_20_lines": [],
        }

    def find(self, result, author="depeng1994"):
        comment = {
            "user": {"login": author},
            "body": "Lite-CI-V3 100:2:"
            + self.item["case_id"]
            + "\n"
            + json.dumps(result),
        }
        return waiter.find_result(
            [comment],
            run_id=100,
            attempt=2,
            sha=SHA,
            digest="d" * 64,
            item=self.item,
            workflow=prepare.WORKFLOW,
            author="depeng1994",
        )

    def test_producer_and_waiter_use_canonical_repository(self):
        self.assertEqual(prepare.REPO, REPO)
        self.assertEqual(waiter.REPO, REPO)

    def test_canonical_personal_fork_pr_uses_default_smoke(self):
        plan = prepare.create_plan("pull_request_target", self.payload, "", 100, 2)
        self.assertEqual((plan["schema"], plan["repo"], plan["sha"]), (3, REPO, SHA))
        self.assertEqual(len(plan["cases"]), 4)

    def test_obsolete_target_repository_is_rejected_without_alias(self):
        self.payload["pull_request"]["base"]["repo"]["full_name"] = OLD_REPO
        with self.assertRaisesRegex(ValueError, "target repository"):
            prepare.create_plan("pull_request_target", self.payload, "", 100, 2)

    def test_other_source_fork_is_accepted(self):
        self.payload["pull_request"]["head"]["repo"]["full_name"] = "other/vllm-ascend"
        self.assertEqual(len(prepare.create_plan("pull_request_target", self.payload, "", 100, 2)["cases"]), 4)

    def test_other_actor_is_accepted(self):
        self.payload["pull_request"]["user"]["login"] = "other"
        self.assertEqual(len(prepare.create_plan("pull_request_target", self.payload, "", 100, 2)["cases"]), 4)

    def test_manual_tag_is_rejected(self):
        with patch.dict(
            os.environ,
            {"GITHUB_REF": "refs/tags/untrusted", "GITHUB_SHA": SHA},
            clear=True,
        ), self.assertRaisesRegex(ValueError, "requires a branch"):
            prepare.create_plan("workflow_dispatch", {}, "", 100, 2)

    def test_unknown_selector_is_rejected(self):
        selection = "missing"
        with patch.dict(
            os.environ, {"GITHUB_REF": "refs/heads/main", "GITHUB_SHA": SHA}, clear=True
        ), self.assertRaisesRegex(ValueError, "unknown"):
            prepare.create_plan("workflow_dispatch", {}, selection, 100, 2)

    def test_control_entrypoints_reject_other_repositories_before_io(self):
        for module in (prepare, waiter):
            for repository in (OLD_REPO, "depeng1994/vllm-ascend", "other/repo"):
                with self.subTest(module=module.__name__, repository=repository), patch.dict(
                    os.environ, {"GITHUB_REPOSITORY": repository}, clear=True
                ), self.assertRaisesRegex(ValueError, "untrusted repository"):
                    module.main()

    def test_waiter_accepts_matching_canonical_result(self):
        self.assertEqual(self.find(self.result), self.result)

    def test_waiter_rejects_old_repository_result_without_rewriting_history(self):
        self.assertIsNone(self.find({**self.result, "repo": OLD_REPO}))

    def test_waiter_rejects_other_attempt_sha_and_plan(self):
        for key, value in (
            ("attempt", 1),
            ("sha", "a" * 40),
            ("plan_digest", "e" * 64),
            ("run_id", 101),
        ):
            with self.subTest(key=key):
                self.assertIsNone(self.find({**self.result, key: value}))

    def test_waiter_pass_requires_successful_cleanup(self):
        for patch_value in (
            {"cleanup_verified": False},
            {"exit_code": 1},
            {"exit_code": False},
        ):
            with self.subTest(patch_value=patch_value):
                self.assertIsNone(self.find({**self.result, **patch_value}))

    def test_waiter_rejects_other_worker_container_and_author(self):
        for patch_value in (
            {"worker": "blue"},
            {"docker": "vllm_hccl_smoke"},
            {"repo": "other/repo"},
        ):
            with self.subTest(patch_value=patch_value):
                self.assertIsNone(self.find({**self.result, **patch_value}))
        self.assertIsNone(self.find(copy.deepcopy(self.result), author="other"))


if __name__ == "__main__":
    unittest.main()
