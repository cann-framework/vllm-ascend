"""Contract coverage for vLLM Smoke planning and PR source identity."""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import native_result  # noqa: E402
import native_source  # noqa: E402
import prepare_request  # noqa: E402
import wait_result  # noqa: E402


class BlueSourceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, GITHUB_WORKFLOW_SHA="c" * 40, GITHUB_RUN_ID="123")
        self.env.start()
        self.addCleanup(self.env.stop)
        self.base = "b" * 40
        self.pr = {
            "number": 7,
            "state": "open",
            "draft": True,
            "mergeable": True,
            "merge_commit_sha": "d" * 40,
            "user": {"login": "someone"},
            "head": {"sha": "a" * 40, "repo": {"id": 987, "full_name": "someone/vllm-ascend"}},
            "base": {"ref": "main", "repo": {"id": native_source.REPO_ID, "full_name": native_source.REPO}},
        }
        self.parents = [{"sha": self.base}, {"sha": "a" * 40}]
        self.event = {"pull_request": self.pr}
        self.plan = prepare_request.create_plan("pull_request_target", self.event, "", 123, 1)

    def get(self, path):
        if path == "/pulls/7":
            return self.pr
        if path in {"/branches/main", "/branches/releases%2Fv0.30.0"}:
            return {"commit": {"sha": self.base}}
        if path.startswith("/git/commits/"):
            return {"sha": path.rsplit("/", 1)[-1], "parents": self.parents, "tree": {"sha": "e" * 40}}
        raise AssertionError(path)

    def build(self):
        return native_source.build_source(self.plan, self.event, self.get)

    def test_default_plan_includes_independent_8p_16p_8p_instances(self):
        cases = self.plan["cases"]
        self.assertEqual(
            [c["test_id"] for c in cases],
            ["worker_os_blue", "dispatch_smoke_8p", "dispatch_smoke_16p", "dispatch_smoke_8p"],
        )
        self.assertEqual([c["resources"]["nodes"] for c in cases], [0, 1, 2, 1])
        self.assertEqual(len({c["case_id"] for c in cases}), 4)
        self.assertEqual(cases[1]["test_id"], cases[3]["test_id"])

    def test_single_case_on_pr_keeps_merge_identity_without_full_suite_status(self):
        selection = "dispatch_smoke_8p"
        self.plan = prepare_request.create_plan("pull_request_target", self.event, selection, 123, 1)
        source = self.build()
        self.assertEqual(source["test_ids"], ["dispatch_smoke_8p"])
        self.assertEqual(source["source_sha"], "d" * 40)
        with patch.object(native_result, "post") as post:
            native_result.publish(source, "success")
            post.assert_not_called()

    def test_manual_explicit_commit_is_pinned_separately_from_controller(self):
        with patch.dict(os.environ, GITHUB_REF="refs/heads/main", GITHUB_SHA="c" * 40, CI_COMMIT="a" * 40):
            plan = prepare_request.create_plan("workflow_dispatch", {}, "", 123, 1)
            source = native_source.build_source(plan, {}, self.get)
        self.assertEqual(source["control_sha"], "c" * 40)
        self.assertEqual((plan["sha"], source["source_sha"]), ("a" * 40, "a" * 40))

    def test_manual_entrypoint_rejects_pr_and_commit_together(self):
        with tempfile.TemporaryDirectory() as folder:
            event = Path(folder) / "event.json"
            event.write_text("{}")
            env = {
                "GITHUB_REPOSITORY": prepare_request.REPO,
                "GITHUB_EVENT_PATH": str(event),
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "CI_SOURCE_PR": "7",
                "CI_COMMIT": "a" * 40,
            }
            with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(ValueError, "either a PR or a commit"):
                prepare_request.main()

    def test_arbitrary_fork_source_is_bound_to_true_merge(self):
        source = self.build()
        self.assertEqual(source["head_repo_id"], 987)
        self.assertEqual(
            (source["head_sha"], source["base_sha"], source["source_sha"], source["tree_sha"]),
            ("a" * 40, "b" * 40, "d" * 40, "e" * 40),
        )
        self.assertEqual(source["plan_digest"], self.plan["plan_digest"])
        self.assertEqual(source["test_ids"], [c["test_id"] for c in self.plan["cases"]])

    def test_changed_head_base_or_repo_invalidates_version(self):
        for target, value in (("base", "f" * 40), ("head", "f" * 40), ("repo", 1234)):
            with self.subTest(target=target):
                source = self.build()
                previous_base, previous_pr = self.base, copy.deepcopy(self.pr)
                if target == "base":
                    self.base = value
                elif target == "head":
                    self.pr["head"]["sha"] = value
                else:
                    self.pr["head"]["repo"]["id"] = value
                with self.assertRaisesRegex(ValueError, "STALE"):
                    native_source.assert_current(source, self.get)
                self.base, self.pr = previous_base, previous_pr
                self.event = {"pull_request": self.pr}

    def test_unavailable_or_stale_merge_fails_closed(self):
        self.parents[0]["sha"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "STALE_MERGE_PARENTS"):
            self.build()
        self.pr["mergeable"] = False
        with self.assertRaisesRegex(ValueError, "MERGE_CONFLICT"):
            self.build()
        self.pr["mergeable"] = True
        self.pr["merge_commit_sha"] = None
        with self.assertRaisesRegex(ValueError, "MERGE_NOT_READY"):
            self.build()

    def test_release_base_ref_is_bound_and_retargeting_is_stale(self):
        self.pr["base"]["ref"] = "releases/v0.30.0"
        self.plan = prepare_request.create_plan("pull_request_target", self.event, "", 123, 1)
        source = self.build()
        self.assertEqual(source["base_ref"], "releases/v0.30.0")
        self.pr["base"]["ref"] = "main"
        with self.assertRaisesRegex(ValueError, "STALE"):
            native_source.assert_current(source, self.get)

    def test_manual_source_names_its_actual_branch(self):
        with patch.dict(os.environ, GITHUB_REF="refs/heads/codex/pilot", GITHUB_SHA="c" * 40):
            plan = prepare_request.create_plan("workflow_dispatch", {}, "", 123, 1)
            source = native_source.build_source(plan, {}, self.get)
        self.assertEqual(source["base_ref"], "codex/pilot")

    def test_manual_single_case_and_unknown_selector(self):
        selection = "dispatch_smoke_8p"
        with patch.dict(os.environ, GITHUB_REF="refs/heads/codex/pilot", GITHUB_SHA="c" * 40):
            plan = prepare_request.create_plan("workflow_dispatch", {}, selection, 123, 1)
            self.assertEqual([c["test_id"] for c in plan["cases"]], [selection])
            with self.assertRaises(ValueError):
                prepare_request.create_plan("workflow_dispatch", {}, "missing", 123, 1)

    def test_plan_inherits_model_resource_overrides(self):
        case = replace(
            prepare_request.BLUE_CASES["dispatch_smoke_8p"],
            ci_overrides={"worker": "blue", "target_pool": "another_pool", "docker": "another_container"},
        )
        with patch.dict(prepare_request.BLUE_CASES, {case.test_name: case}):
            plan = prepare_request.create_plan("pull_request_target", self.event, "", 123, 1)
        for item in (plan["cases"][1], plan["cases"][3]):
            self.assertEqual(
                (item["worker"], item["target_pool"], item["docker"]),
                ("blue", "another_pool", "another_container"),
            )
        probe = replace(prepare_request.BLUE_CASES["worker_os_blue"], ci_overrides={"target_pool": "another_pool"})
        with (
            patch.dict(prepare_request.BLUE_CASES, {probe.test_name: probe}),
            self.assertRaisesRegex(ValueError, "probe cannot request resources"),
        ):
            prepare_request.create_plan("pull_request_target", self.event, "", 123, 1)

    def test_new_model_case_expands_without_changing_workflow(self):
        root = SCRIPTS.parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            shutil.copytree(root / "tests", target / "tests")
            shutil.copytree(SCRIPTS, target / "scripts")
            catalog = target / "tests/integration_tests/training_cases/dispatch_smoke.py"
            text = catalog.read_text()
            text = text.replace(
                '        "dispatch_smoke_16p",', '        "dispatch_smoke_16p",\n        "new_smoke_8p",'
            )
            text = text.replace(
                "        npu_smoke(2),",
                '        npu_smoke(2),\n        replace(npu_smoke(1), test_name="new_smoke_8p"),',
            )
            text = text.replace(
                "from tests.integration_tests import OverrideDefinitions",
                "from dataclasses import replace\nfrom tests.integration_tests import OverrideDefinitions",
            )
            catalog.write_text(text)
            script = (
                "import json,os,prepare_request; "
                "p=prepare_request.create_plan('workflow_dispatch',{},'',1,1); "
                "print(json.dumps([c['test_id'] for c in p['cases']]))"
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(target) + os.pathsep + str(target / "scripts"),
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SHA": "c" * 40,
            }
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=target, env=env, check=True, capture_output=True, text=True
            )
            self.assertEqual(
                json.loads(result.stdout),
                ["worker_os_blue", "dispatch_smoke_8p", "dispatch_smoke_16p", "new_smoke_8p", "dispatch_smoke_8p"],
            )

    def test_waiter_requires_matching_source_identity(self):
        source = self.build()
        case = self.plan["cases"][1]
        result = {key: case[key] for key in ("case_id", "test_id", "worker", "target_pool", "docker", "resources")}
        result.update(
            schema=3,
            repo=prepare_request.REPO,
            workflow=prepare_request.WORKFLOW,
            run_id=123,
            attempt=1,
            sha="a" * 40,
            plan_digest=self.plan["plan_digest"],
            status="PASS",
            exit_code=0,
            cleanup_verified=True,
            last_20_lines=[],
            native_source=source,
        )

        def find(*values):
            comments = [
                {"user": {"login": "worker"}, "body": f"Lite-CI-V3 123:1:{case['case_id']}\n" + json.dumps(value)}
                for value in values
            ]
            return wait_result.find_result(
                comments,
                run_id=123,
                attempt=1,
                sha="a" * 40,
                digest=self.plan["plan_digest"],
                item=case,
                workflow=prepare_request.WORKFLOW,
                author="worker",
                native_source=source,
            )

        self.assertEqual(find(result), result)
        wrong = copy.deepcopy(result)
        wrong["native_source"]["source_sha"] = "f" * 40
        self.assertIsNone(find(wrong))
        self.assertIsNone(find({k: v for k, v in result.items() if k != "native_source"}))
        with self.assertRaisesRegex(ValueError, "invalid case result"):
            find(result, wrong)
        with self.assertRaisesRegex(ValueError, "conflicting case results"):
            find(result, {**result, "status": "FAIL", "exit_code": 1})

    def test_waiter_script_accepts_valid_result_without_inherited_pythonpath(self):
        root = SCRIPTS.parents[2]
        source = {
            "mode": "commit",
            "run_id": 123,
            "attempt": 1,
            "head_sha": "a" * 40,
            "control_sha": "c" * 40,
            "plan_digest": self.plan["plan_digest"],
        }
        case = self.plan["cases"][1]
        result = {key: case[key] for key in ("case_id", "test_id", "worker", "target_pool", "docker", "resources")}
        result.update(
            schema=3,
            repo=prepare_request.REPO,
            workflow=prepare_request.WORKFLOW,
            run_id=123,
            attempt=1,
            sha="a" * 40,
            plan_digest=self.plan["plan_digest"],
            status="PASS",
            exit_code=0,
            cleanup_verified=True,
            last_20_lines=[],
            native_source=source,
        )
        comments = [{"user": {"login": "worker"}, "body": f"Lite-CI-V3 123:1:{case['case_id']}\n" + json.dumps(result)}]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "source.json").write_text(json.dumps(source))
            (folder / "sitecustomize.py").write_text(
                "import io,json,os,urllib.request\n"
                "class Response(io.BytesIO):\n    headers = {}\n"
                "def fake(req,timeout):\n"
                "    data = json.loads(os.environ['MOCK_RESPONSES'])\n"
                "    key = 'artifacts' if '/artifacts?' in req.full_url else 'comments'\n"
                "    return Response(json.dumps(data[key]).encode())\n"
                "urllib.request.urlopen = fake\n"
            )
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env.update(
                PYTHONPATH=str(root) + os.pathsep + str(folder),
                GITHUB_REPOSITORY=prepare_request.REPO,
                GITHUB_RUN_ID="123",
                GITHUB_RUN_ATTEMPT="1",
                GITHUB_WORKFLOW_SHA="c" * 40,
                CI_RESULT_AUTHOR="worker",
                CI_CASE=json.dumps(case),
                CI_SOURCE_SHA="a" * 40,
                CI_PLAN_DIGEST=self.plan["plan_digest"],
                CI_WORKFLOW=prepare_request.WORKFLOW,
                CI_NATIVE_SOURCE=str(folder / "source.json"),
                GH_TOKEN="fake",
                MOCK_RESPONSES=json.dumps(
                    {"artifacts": {"artifacts": [{"name": "lite-ci-request-1"}]}, "comments": comments}
                ),
            )
            completed = subprocess.run(
                [sys.executable, ".github/scripts/lite_actions/wait_result.py"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("PASS", completed.stdout)



if __name__ == "__main__":
    unittest.main()
