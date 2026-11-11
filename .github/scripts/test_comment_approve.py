import base64
import json
import os
import sys
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comment_approve import (
    ACTIONS_APP_ID,
    MARKER,
    ApiError,
    Handler,
    initialize,
    parse_policy,
    reconcile,
    signature,
)

POLICY = """# Global owners
* @global-one @global-two
/docs/architecture/ @architect @global-one @global-two
/docs/modules/serving-scheduler/ @scheduler @global-one @global-two
/docs/modules/epd-kv-transfer/ @transfer @global-one @global-two
/docs/modules/kv-cache-pooling/ @pool @global-one @global-two
/docs/modules/standalone/ @standalone @global-one @global-two
"""
SHA = "a" * 40
POLICY_SHA = "c" * 40
KEY = b"test-only-signing-key"
PR = {
    "number": 1,
    "state": "open",
    "draft": False,
    "node_id": "PR_node",
    "user": {"id": 1, "login": "author"},
    "head": {"sha": SHA},
    "base": {"ref": "main", "sha": "b" * 40},
    "changed_files": 1,
}
HEAD = {
    "kind": "head",
    "head_sha": SHA,
    "base_ref": "main",
    "base_sha": "b" * 40,
    "seen_at": "2026-11-05T01:00:00Z",
    "run_id": 10,
}


def human(login="scheduler", user_id=2, comment_id=100, **extra):
    result = {
        "id": comment_id,
        "body": "/approve",
        "updated_at": "2026-11-05T01:00:05Z",
        "user": {"id": user_id, "login": login, "type": "User"},
    }
    result.update(extra)
    return result


def approval(source, **extra):
    result = {
        "kind": "approval",
        "head_sha": SHA,
        "policy_sha": POLICY_SHA,
        "base_ref": "main",
        "base_sha": "b" * 40,
        "comment_id": source["id"],
        "source_updated_at": source["updated_at"],
        "user_id": source["user"]["id"],
        "login": source["user"]["login"],
        "run_id": 20,
    }
    result.update(extra)
    return result


def bot(record, **extra):
    record = dict(record, repository="org/repo", pr_number=1)
    record["signature"] = signature(record, KEY)
    result = {
        "id": 900 + record["run_id"],
        "body": MARKER + json.dumps(record) + " -->\nAcknowledged",
        "user": {"id": 99, "login": "github-actions[bot]", "type": "Bot"},
        "performed_via_github_app": {"id": ACTIONS_APP_ID},
    }
    result.update(extra)
    return result


def event(source=None, name="issue_comment", action="created"):
    result = {"action": action, "repository": {"id": 5, "default_branch": "main"}}
    if name == "issue_comment":
        source = source or human()
        result.update(issue={"number": 1, "pull_request": {}}, comment=source, sender={"id": source["user"]["id"]})
    else:
        result["pull_request"] = deepcopy(PR)
        result["pull_request"]["updated_at"] = HEAD["seen_at"]
    return result


class FakeAPI:
    repo = "org/repo"

    def __init__(self, comments=None, files=None):
        self.pr = deepcopy(PR)
        self.comments = comments if comments is not None else [bot(HEAD)]
        self.files = files if files is not None else [{"filename": "docs/modules/serving-scheduler/design.md"}]
        self.reviews = []
        self.writes = []
        self.policy_text = POLICY
        self.policy_sha = POLICY_SHA
        self.merge_error = None
        self.collaborators = [
            {"login": "global-one", "permissions": {"admin": True, "push": True}},
            {"login": "global-two", "permissions": {"admin": True, "push": True}},
            *[
                {"login": login, "permissions": {"admin": False, "push": True}}
                for login in ("scheduler", "architect", "transfer", "pool", "standalone")
            ],
        ]

    def pages(self, path):
        if path == "pulls?state=open":
            return [deepcopy(self.pr)] if self.pr["state"] == "open" else []
        if path == "issues/1/comments":
            return deepcopy(self.comments)
        if path == "pulls/1/files":
            return deepcopy(self.files)
        if path == "pulls/1/reviews":
            return deepcopy(self.reviews)
        raise AssertionError(path)

    def rest(self, method, path, data=None):
        if method == "GET":
            if path == "pulls/1":
                return deepcopy(self.pr)
            if path == "git/ref/heads/main":
                return {"object": {"sha": "b" * 40}}
            if path.startswith("contents/.github/CODEOWNERS?"):
                return {"sha": self.policy_sha, "content": base64.b64encode(self.policy_text.encode()).decode()}
            if path.startswith("collaborators/") and path.endswith("/permission"):
                login = path.split("/")[1].lower()
                user = next((u for u in self.collaborators if u["login"].lower() == login), None)
                return {
                    "permission": (
                        "admin"
                        if user and user["permissions"]["admin"]
                        else "write"
                        if user and user["permissions"]["push"]
                        else "none"
                    )
                }
            raise AssertionError(path)
        self.writes.append((method, path, data))
        if path == "issues/1/comments" and method == "POST":
            self.comments.append(
                {
                    "id": 1000 + len(self.comments),
                    "body": data["body"],
                    "user": {"login": "github-actions[bot]", "type": "Bot"},
                    "performed_via_github_app": {"id": ACTIONS_APP_ID},
                }
            )
            return deepcopy(self.comments[-1])
        if path == "check-runs":
            return {"id": 200}
        if path == "pulls/1/merge":
            if self.merge_error:
                raise self.merge_error
            return {"merged": True, "sha": "d" * 40}
        return {}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = parse_policy(POLICY)

    def test_renames_require_old_and_new_owners(self):
        scopes = self.policy.scopes(
            [{"filename": "docs/modules/kv-cache-pooling/a.md", "previous_filename": "docs/modules/standalone/a.md"}]
        )
        self.assertEqual(len(scopes), 2)
        self.assertEqual(self.policy.coverage(scopes, {"pool", "standalone"}), ())


class RepositoryPolicyTests(unittest.TestCase):
    def test_current_repository_codeowners_are_supported(self):
        text = (Path(__file__).resolve().parents[1] / "CODEOWNERS").read_text()
        policy = parse_policy(text, admins={"repo-admin"}, writers={"repo-admin"})
        paths = [
            {"filename": "README.md"},
            {"filename": "tests/new/test.py"},
            {"filename": ".github/workflows/new.yml"},
        ]
        self.assertEqual(policy.coverage(policy.scopes(paths), {"repo-admin"}), ())

    def test_writer_owners_and_repository_admin_are_honored(self):
        policy = parse_policy(
            "/docs @writer @former-writer\n/Dockerfile* @writer\n", admins={"Admin"}, writers={"writer"}
        )
        docs = policy.scopes([{"filename": "docs/guide.md"}])
        docker = policy.scopes([{"filename": "Dockerfile.dev"}])
        self.assertEqual(policy.coverage(docs, {"writer"}), ())
        self.assertEqual(policy.coverage(docker, {"writer"}), ())
        self.assertEqual(policy.coverage(docs, {"admin"}), ())
        self.assertTrue(policy.coverage(docs, {"former-writer"}))

    def test_workflows_and_scripts_require_global_authority(self):
        policy = parse_policy("/.github @writer\n", admins={"admin"}, writers={"writer"})
        for path in (".github/CODEOWNERS", ".github/workflows/new.yml", ".github/scripts/json.py"):
            with self.subTest(path=path):
                scopes = policy.scopes([{"filename": path}])
                self.assertTrue(policy.coverage(scopes, {"writer"}))
                self.assertEqual(policy.coverage(scopes, {"admin"}), ())


class HandlerTests(unittest.TestCase):
    def setUp(self):
        smoke = patch.object(Handler, "smoke_passed", return_value=True)
        smoke.start()
        self.addCleanup(smoke.stop)

    def handler(self, api, source=None, name="issue_comment", action="created"):
        return Handler(api, event(source, name, action), name, 50, KEY)

    def record_approvals(self, sources, files=None):
        api = FakeAPI([], files=files)
        api.pr["changed_files"] = len(api.files)
        self.handler(api, name="pull_request_target", action="opened").record()
        for source in sources:
            api.comments.append(source)
            self.handler(api, source).record()
        return api

    def test_recorded_comment_authorization_and_merge(self):
        for login, user_id, expected in (
            ("global-one", 1, True),
            ("scheduler", 1, False),
            ("scheduler", 2, True),
            ("stranger", 3, False),
        ):
            with self.subTest(login=login, user_id=user_id):
                source = human(login=login, user_id=user_id)
                api = self.record_approvals([source])
                self.handler(api, source).evaluate()
                merges = [payload for _, path, payload in api.writes if path == "pulls/1/merge"]
                self.assertEqual(bool(merges), expected)
                if expected:
                    self.assertEqual(merges[-1]["sha"], SHA)

    def test_initialize_existing_pr_ignores_old_comments_and_accepts_new_comment(self):
        old = human(updated_at="2026-11-05T01:00:05Z")
        api = FakeAPI([old])
        initialize(api, {"repository": {"default_branch": "main"}}, 40, KEY, "2026-11-05T01:01:00Z")
        self.handler(api, old).record()
        self.handler(api).evaluate()
        self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))
        current = human(comment_id=101, updated_at="2026-11-05T01:01:10Z")
        api.comments.append(current)
        self.handler(api, current).record()
        self.handler(api).evaluate()
        self.assertTrue(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_reconcile_seeds_missing_and_changed_prs_without_using_old_comments(self):
        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 11, 5, 1, 1, tzinfo=timezone.utc)

        for scenario in ("missing", "changed"):
            with self.subTest(scenario=scenario), patch("comment_approve.datetime", FrozenDateTime):
                old = human(updated_at="2026-11-05T01:00:05Z")
                api = FakeAPI([old])
                if scenario == "changed":
                    api.comments.extend([bot(HEAD), bot(approval(old))])
                    api.pr["base"]["sha"] = "f" * 40
                reconcile(api, api, {"repository": {"default_branch": "main"}}, 51, KEY)
                self.handler(api, old).record()
                self.handler(api).evaluate()
                self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))
                new = human(comment_id=101, updated_at="2026-11-05T01:01:10Z")
                api.comments.append(new)
                self.handler(api, new).record()
                self.handler(api).evaluate()
                self.assertTrue(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_reconcile_preserves_existing_current_approval(self):
        api = self.record_approvals([human()])
        original_comments = deepcopy(api.comments)
        reconcile(api, api, {"repository": {"default_branch": "main"}}, 51, KEY)
        self.assertEqual(api.comments, original_comments)
        self.assertTrue(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_multiple_scopes_require_all_owners_or_admin(self):
        files = [
            {"filename": "docs/modules/serving-scheduler/a.md"},
            {"filename": "docs/modules/epd-kv-transfer/b.md"},
        ]
        for logins, expected in (
            (("scheduler",), False),
            (("scheduler", "transfer"), True),
            (("global-one",), True),
        ):
            with self.subTest(logins=logins):
                sources = [human(login=login, user_id=i + 2, comment_id=i + 100) for i, login in enumerate(logins)]
                api = self.record_approvals(sources, files)
                self.handler(api).evaluate()
                self.assertEqual(any(path.endswith("/merge") for _, path, _ in api.writes), expected)

    def test_comment_version_and_permission_changes_revoke_approval(self):
        for change in (
            "deleted",
            "edited",
            "edited_back",
            "head",
            "base_sha",
            "base_ref",
            "draft",
            "permission",
            "policy",
            "reopened",
            "same_second",
        ):
            with self.subTest(change=change):
                source = human()
                if change == "same_second":
                    source["updated_at"] = HEAD["seen_at"]
                api = self.record_approvals([source])
                if change == "deleted":
                    api.comments.remove(source)
                elif change == "edited":
                    source.update(body="withdrawn", updated_at="2026-11-05T01:01:00Z")
                elif change == "edited_back":
                    source["updated_at"] = "2026-11-05T01:01:00Z"
                elif change == "head":
                    api.pr["head"]["sha"] = "f" * 40
                elif change == "base_sha":
                    api.pr["base"]["sha"] = "f" * 40
                elif change == "base_ref":
                    api.pr["base"]["ref"] = "releases/v0.30.0"
                elif change == "draft":
                    api.pr["draft"] = True
                elif change == "permission":
                    next(user for user in api.collaborators if user["login"] == "scheduler")["permissions"]["push"] = (
                        False
                    )
                elif change == "policy":
                    api.policy_sha = "f" * 40
                elif change == "reopened":
                    reopen_event = event(name="pull_request_target", action="reopened")
                    reopen_event["pull_request"]["updated_at"] = "2026-11-05T01:01:00Z"
                    Handler(api, reopen_event, "pull_request_target", 51, KEY).record()
                if change == "draft":
                    with self.assertRaises(ValueError):
                        self.handler(api).evaluate()
                else:
                    self.handler(api).evaluate()
                self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_forged_approval_records_cannot_merge(self):
        for attack in ("tampered", "human", "other_app", "wrong_key"):
            with self.subTest(attack=attack):
                source = human()
                api = self.record_approvals([source])
                recorded = api.comments[-1]
                if attack == "tampered":
                    recorded["body"] = recorded["body"].replace(SHA, "f" * 40)
                elif attack == "human":
                    recorded["user"]["login"] = "scheduler"
                elif attack == "other_app":
                    recorded["performed_via_github_app"]["id"] = 123
                handler = Handler(api, event(), "issue_comment", 50, b"wrong-key" if attack == "wrong_key" else KEY)
                handler.evaluate()
                self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_invalid_policy_and_incomplete_files_fail_closed(self):
        for failure in ("policy", "files"):
            with self.subTest(failure=failure):
                api = self.record_approvals([human()])
                if failure == "policy":
                    api.policy_text = "* @org/team"
                else:
                    api.pr["changed_files"] = 2
                with self.assertRaises(ValueError):
                    self.handler(api).evaluate()
                self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_ignore_non_command(self):
        api = FakeAPI()
        self.handler(api, human(body="hello")).record()
        self.assertEqual(api.writes, [])

    def test_release_base_uses_default_branch_policy_and_merges(self):
        source = human(login="global-one", user_id=1)
        head = dict(HEAD, base_ref="releases/v0.30.0", base_sha="e" * 40)
        signed = approval(source, base_ref="releases/v0.30.0", base_sha="e" * 40)
        api = FakeAPI([source, bot(head), bot(signed)])
        api.pr["base"] = {"ref": "releases/v0.30.0", "sha": "e" * 40}
        self.handler(api).evaluate()
        self.assertEqual(api.writes[-1][1], "pulls/1/merge")

    def test_signed_approval_cannot_be_replayed_for_another_pr(self):
        class OtherPRAPI(FakeAPI):
            def pages(self, path):
                return super().pages(path.replace("issues/2/", "issues/1/").replace("pulls/2/", "pulls/1/"))

            def rest(self, method, path, data=None):
                if path == "pulls/2":
                    path = "pulls/1"
                return super().rest(method, path.replace("pulls/2/", "pulls/1/"), data)

        source = human()
        api = OtherPRAPI([source, bot(HEAD), bot(approval(source))])
        api.pr["number"] = 2
        own_event = event(source)
        own_event["issue"]["number"] = 2
        Handler(api, own_event, "issue_comment", 50, KEY).evaluate()
        self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_edit_by_another_user_is_not_attributed_to_author(self):
        api = FakeAPI()
        handler = self.handler(api)
        handler.event["sender"]["id"] = 999
        handler.record()
        self.assertEqual(api.writes, [])

    def test_unavailable_permission_check_fails_closed(self):
        source = human()
        api = FakeAPI([source, bot(HEAD), bot(approval(source))])
        original = api.rest

        def fail_permission(method, path, data=None):
            if path.startswith("collaborators/"):
                raise ApiError(503, "Permission service unavailable")
            return original(method, path, data)

        api.rest = fail_permission
        with self.assertRaises(ApiError):
            self.handler(api).evaluate()
        self.assertEqual(api.writes[-1][2]["conclusion"], "failure")

    def test_checks_use_the_separate_app_token(self):
        source = human()
        api = FakeAPI([source, bot(HEAD), bot(approval(source))])
        checks_api = FakeAPI()
        handler = Handler(api, event(), "issue_comment", 50, KEY, checks_api)
        handler.evaluate()
        self.assertFalse(any(path.startswith("check-runs") for _, path, _ in api.writes))
        self.assertTrue(any(path.startswith("check-runs") for _, path, _ in checks_api.writes))

    def test_comment_after_changes_requested_does_not_unblock_merge(self):
        source = human(login="global-one", user_id=2)
        api = FakeAPI([source, bot(HEAD), bot(approval(source))])
        api.reviews = [
            {
                "id": 1,
                "user": {"login": "scheduler"},
                "state": "CHANGES_REQUESTED",
                "submitted_at": "2026-11-05T00:00:00Z",
            },
            {"id": 2, "user": {"login": "scheduler"}, "state": "COMMENTED", "submitted_at": "2026-11-06T00:00:00Z"},
        ]
        self.handler(api).evaluate()
        self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))
        self.assertEqual(api.writes[-1][2]["conclusion"], "failure")

    def test_revoked_permission_before_merge_invalidates_check(self):
        source = human()
        api = FakeAPI([source, bot(HEAD), bot(approval(source))])
        original = api.rest

        def changing_permission(method, path, data=None):
            result = original(method, path, data)
            if method == "PATCH" and path == "check-runs/200" and data["conclusion"] == "success":
                next(user for user in api.collaborators if user["login"] == "scheduler")["permissions"]["push"] = False
            return result

        api.rest = changing_permission
        with self.assertRaises(ValueError):
            self.handler(api).evaluate()
        self.assertEqual(api.writes[-1][2]["conclusion"], "failure")
        self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_record_failure_invalidates_approval_check(self):
        api = FakeAPI()
        self.handler(api).invalidate()
        self.assertEqual(api.writes[-1][2]["conclusion"], "failure")

    def test_changed_base_before_merge_invalidates_check(self):
        source = human()
        api = FakeAPI([source, bot(HEAD), bot(approval(source))])
        original = api.rest

        def changing_base(method, path, data=None):
            result = original(method, path, data)
            if method == "PATCH" and path == "check-runs/200" and data["conclusion"] == "success":
                api.pr["base"]["sha"] = "f" * 40
            return result

        api.rest = changing_base
        with self.assertRaises(ValueError):
            self.handler(api).evaluate()
        self.assertEqual(api.writes[-1][2]["conclusion"], "failure")
        self.assertFalse(any(path.endswith("/merge") for _, path, _ in api.writes))

    def test_pending_checks_retry_fresh_approval_and_merge(self):
        source = human()
        api = FakeAPI([source, bot(HEAD), bot(approval(source))])
        api.merge_error = ApiError(405, "Other checks pending")
        self.handler(api).evaluate()
        self.assertEqual(api.writes[-1][2]["conclusion"], "failure")
        api.merge_error = None
        reconcile(api, api, {"repository": {"default_branch": "main"}}, 51, KEY)
        self.assertEqual(api.writes[-1][1], "pulls/1/merge")


class SmokeContractTests(unittest.TestCase):
    def setUp(self):
        self.control = "b" * 40
        self.ids = ["smoke_8p", "smoke_16p", "smoke_8p"]
        self.repo_id, self.head_repo_id, self.run_id = 5, 6, 100

        def plan(event_name, payload, selection, run_id, attempt):
            return {
                "repo": "org/repo",
                "sha": payload["pull_request"]["head"]["sha"],
                "run_id": run_id,
                "attempt": attempt,
                "plan_digest": "fixed-digest",
                "cases": [{"case_id": f"case_{i:03d}_{name}", "test_id": name} for i, name in enumerate(self.ids)],
            }

        def source(plan, payload, get):
            pr = payload["pull_request"]
            return {
                "pr_number": pr["number"],
                "head_sha": pr["head"]["sha"],
                "base_sha": pr["base"]["sha"],
                "base_ref": pr["base"]["ref"],
                "source_sha": "d" * 40,
                "tree_sha": "e" * 40,
                "control_sha": os.environ["GITHUB_WORKFLOW_SHA"],
                "run_id": plan["run_id"],
                "attempt": plan["attempt"],
                "plan_digest": plan["plan_digest"],
            }

        self.make_plan, self.make_source = plan, source
        self.modules = {
            "blue_catalog": SimpleNamespace(BLUE_SMOKE_SUITE=self.ids),
            "native_result": SimpleNamespace(
                context=lambda number: f"Lite-Actions-Blue-Merge-PR-{number}",
                description=lambda base, ids, ref: json.dumps([base, ids, ref]),
            ),
            "native_source": SimpleNamespace(REPO="org/repo", REPO_ID=self.repo_id, build_source=source),
            "prepare_request": SimpleNamespace(create_plan=plan),
        }
        self.patches = [
            patch.dict(sys.modules, self.modules),
            patch.dict(os.environ, {"GITHUB_WORKFLOW_SHA": self.control}),
        ]
        for active in self.patches:
            active.start()
            self.addCleanup(active.stop)

    def make_api(self):
        outer = self

        class API(FakeAPI):
            def __init__(self):
                source = human()
                super().__init__([source, bot(HEAD), bot(approval(source))])
                self.pr["head"]["repo"] = {"id": outer.head_repo_id}
                self.default_sha = outer.control
                self.ancestor = True
                self.statuses = [
                    {
                        "id": 2,
                        "created_at": "2026-11-10T00:01:00Z",
                        "context": "Lite-Actions-Blue-Merge-PR-1",
                        "state": "success",
                        "creator": {"id": 41898282},
                        "description": json.dumps([self.pr["base"]["sha"], outer.ids, "main"]),
                        "target_url": f"https://github.com/org/repo/actions/runs/{outer.run_id}/attempts/1",
                    }
                ]
                self.run = {
                    "id": outer.run_id,
                    "run_attempt": 1,
                    "workflow_id": 10,
                    "repository": {"id": outer.repo_id},
                    "head_repository": {"id": outer.head_repo_id},
                    "head_sha": SHA,
                    "event": "pull_request_target",
                    "path": ".github/workflows/lite_actions_dispatcher_smoke.yml",
                    "status": "completed",
                    "conclusion": "success",
                    "display_title": f"lite-vllm-pr-1-{SHA}-{outer.control}",
                }
                self.plan = outer.make_plan("pull_request_target", {"pull_request": self.pr}, "", outer.run_id, 1)
                self.source = outer.make_source(self.plan, {"pull_request": self.pr}, None)
                self.artifact_unavailable = False
                self.artifacts = [
                    {
                        "id": i,
                        "name": name,
                        "expired": False,
                        "workflow_run": {"id": outer.run_id, "repository_id": outer.repo_id},
                    }
                    for i, name in ((1, "lite-ci-source-1"), (2, "lite-ci-request-1"))
                ]
                names = ["prepare", "Lite-Actions-Smoke"] + [case["case_id"] for case in self.plan["cases"]]
                self.jobs = [
                    {
                        "name": name,
                        "run_id": outer.run_id,
                        "run_attempt": 1,
                        "status": "completed",
                        "conclusion": "success",
                    }
                    for name in names
                ]

            def pages(self, path):
                if path == f"commits/{SHA}/statuses":
                    return deepcopy(self.statuses)
                return super().pages(path)

            def rest(self, method, path, data=None):
                if method == "GET" and path == "git/ref/heads/main":
                    return {"object": {"sha": self.default_sha}}
                if method == "GET" and path.startswith("compare/"):
                    return {
                        "status": "ahead" if self.ancestor else "diverged",
                        "merge_base_commit": {"sha": outer.control},
                    }
                if method == "GET" and path == f"actions/runs/{outer.run_id}":
                    return deepcopy(self.run)
                if method == "GET" and path == "actions/workflows/lite_actions_dispatcher_smoke.yml":
                    return {"id": 10, "state": "active"}
                return super().rest(method, path, data)

            def object_pages(self, path, key):
                return deepcopy(self.artifacts if key == "artifacts" else self.jobs)

            def artifact_json(self, artifact_id, filename):
                if self.artifact_unavailable:
                    raise ValueError("Artifact unavailable")
                return deepcopy(self.source if filename == "source.json" else self.plan)

        return API()

    def evaluate(self, api):
        try:
            Handler(api, event(), "issue_comment", 50, KEY).evaluate()
        except (ValueError, ApiError):
            self.assertTrue(
                any(
                    path.startswith("check-runs/") and data.get("conclusion") == "failure"
                    for _, path, data in api.writes
                )
            )
        return any(path.endswith("/merge") for _, path, _ in api.writes)

    def test_only_a_complete_current_smoke_allows_merge(self):
        api = self.make_api()
        self.assertTrue(self.evaluate(api))
        manual = self.make_api()
        manual.run.update(
            event="workflow_dispatch",
            head_sha=self.control,
            head_branch="main",
            head_repository={"id": self.repo_id},
            display_title=f"lite-vllm-recheck-1-{self.control}-" + "a" * 32,
        )
        self.assertTrue(self.evaluate(manual))
        self.ids.append("additional_case")
        extended = self.make_api()
        self.assertTrue(self.evaluate(extended))
        extended.jobs.pop()
        extended.writes.clear()
        self.assertFalse(self.evaluate(extended))

    def test_stale_untrusted_or_incomplete_smoke_never_merges(self):
        failures = (
            "no_status",
            "pending",
            "failure",
            "newer_pending",
            "wrong_creator",
            "wrong_description",
            "bad_url",
            "old_attempt",
            "wrong_workflow",
            "wrong_repo",
            "running",
            "wrong_controller",
            "wrong_pr",
            "no_source",
            "expired_source",
            "unreadable_source",
            "duplicate_source",
            "wrong_source_head",
            "wrong_base",
            "wrong_tree",
            "wrong_plan",
            "missing_case",
            "failed_case",
            "old_case_attempt",
            "missing_summary",
        )
        for failure in failures:
            with self.subTest(failure=failure):
                api = self.make_api()
                if failure == "no_status":
                    api.statuses.clear()
                elif failure in ("pending", "failure"):
                    api.statuses[0]["state"] = failure
                elif failure == "newer_pending":
                    api.statuses.append(dict(api.statuses[0], id=3, created_at="2026-11-10T00:02:00Z", state="pending"))
                elif failure == "wrong_creator":
                    api.statuses[0]["creator"]["id"] = 1
                elif failure == "wrong_description":
                    api.statuses[0]["description"] = "older base or smaller suite"
                elif failure == "bad_url":
                    api.statuses[0]["target_url"] = "https://example.com/result"
                elif failure == "old_attempt":
                    api.run["run_attempt"] = 2
                elif failure == "wrong_workflow":
                    api.run["workflow_id"] = 20
                elif failure == "wrong_repo":
                    api.run["repository"]["id"] = 99
                elif failure == "running":
                    api.run["status"] = "in_progress"
                elif failure == "wrong_controller":
                    api.run["display_title"] += "x"
                elif failure == "wrong_pr":
                    api.source["pr_number"] = 2
                elif failure == "no_source":
                    api.artifacts.pop(0)
                elif failure == "expired_source":
                    api.artifacts[0]["expired"] = True
                elif failure == "unreadable_source":
                    api.artifact_unavailable = True
                elif failure == "duplicate_source":
                    api.artifacts.append(deepcopy(api.artifacts[0]))
                elif failure == "wrong_source_head":
                    api.source["head_sha"] = "f" * 40
                elif failure == "wrong_base":
                    api.source["base_sha"] = "f" * 40
                elif failure == "wrong_tree":
                    api.source["tree_sha"] = "f" * 40
                elif failure == "wrong_plan":
                    api.plan["cases"].pop()
                elif failure == "missing_case":
                    api.jobs.pop()
                elif failure == "failed_case":
                    api.jobs[-1]["conclusion"] = "failure"
                elif failure == "old_case_attempt":
                    api.jobs[-1]["run_attempt"] = 2
                elif failure == "missing_summary":
                    api.jobs.pop(1)
                self.assertFalse(self.evaluate(api))

    def test_smoke_result_changes_between_check_and_merge(self):
        api = self.make_api()
        original = api.rest

        def supersede(method, path, data=None):
            result = original(method, path, data)
            if method == "PATCH" and path == "check-runs/200" and data["conclusion"] == "success":
                api.statuses.append(dict(api.statuses[0], id=3, created_at="2026-11-10T00:02:00Z", state="pending"))
            return result

        api.rest = supersede
        self.assertFalse(self.evaluate(api))
        self.assertEqual(api.writes[-1][2]["conclusion"], "failure")

    def test_main_advance_preserves_release_smoke_only_for_ancestor_controller(self):
        for ancestor in (True, False):
            with self.subTest(ancestor=ancestor):
                api = self.make_api()
                api.pr["base"]["ref"] = "releases/v0.30.0"
                api.comments[1] = bot(dict(HEAD, base_ref="releases/v0.30.0"))
                source = api.comments[0]
                api.comments[2] = bot(approval(source, base_ref="releases/v0.30.0"))
                api.statuses[0]["description"] = json.dumps([api.pr["base"]["sha"], self.ids, "releases/v0.30.0"])
                api.source = self.make_source(api.plan, {"pull_request": api.pr}, None)
                api.default_sha, api.ancestor = "f" * 40, ancestor
                with patch.dict(os.environ, {"GITHUB_WORKFLOW_SHA": api.default_sha}):
                    self.assertEqual(self.evaluate(api), ancestor)
                    self.assertEqual(os.environ["GITHUB_WORKFLOW_SHA"], api.default_sha)


if __name__ == "__main__":
    unittest.main()
