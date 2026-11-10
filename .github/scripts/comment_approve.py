"""Trusted /approve handler. Standard library only; never execute PR contents."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from zipfile import ZipFile

CHECK_NAME = "committer-approval"
ACTIONS_APP_ID = 15368
MARKER = "<!-- cann-committer:v1 "


def command(body):
    return isinstance(body, str) and body.lower() == "/approve"


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp has no timezone")
    return parsed


def signature(record, key):
    payload = {k: v for k, v in record.items() if k != "signature"}
    derived = hashlib.sha256(b"cann-committer-record-v1\0" + key).digest()
    return hmac.new(
        derived, json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256
    ).hexdigest()


@dataclass(frozen=True)
class Scope:
    pattern: str
    owners: frozenset


@dataclass(frozen=True)
class Policy:
    globals: frozenset
    rules: tuple
    admins: frozenset = frozenset()

    def scopes(self, files):
        found = {}
        for item in files:
            # A rename needs the owners of both the old and new paths.
            for path in (item["filename"], item.get("previous_filename")):
                if not path:
                    continue
                scope = Scope("*", self.globals)
                # A new workflow or Python module can access the approval
                # environment. Only global reviewers may alter these paths.
                if path == ".github/CODEOWNERS" or path.startswith((".github/workflows/", ".github/scripts/")):
                    found[scope.pattern] = scope
                    continue
                for rule in self.rules:
                    pattern = rule.pattern.lstrip("/").rstrip("/")
                    # A directory rule also applies to descendants; a wildcard
                    # must not cross a path separator.
                    expression = re.escape(pattern).replace(r"\*", "[^/]*")
                    if re.fullmatch(expression + r"(?:/.*)?", path):
                        scope = rule
                found[scope.pattern] = scope
        return tuple(found[key] for key in sorted(found))

    def coverage(self, scopes, logins):
        logins = {login.lower() for login in logins}
        if self.globals.intersection(logins):
            return ()
        return tuple(scope.pattern for scope in scopes if not scope.owners.intersection(logins))


def parse_policy(text, *, admins=(), writers=None, require_globals=True):
    """Support the repository's root *, anchored paths and component wildcards.

    Fail closed for unsupported CODEOWNERS syntax instead of approximating it.
    Team owners require a different membership resolver and are not accepted.
    """
    globals_ = None
    rules = []
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        pattern, *owners = parts
        if not owners or any(not re.fullmatch(r"@[A-Za-z0-9][A-Za-z0-9-]*", owner) for owner in owners):
            raise ValueError("CODEOWNERS must list individual GitHub usernames")
        names = frozenset(owner[1:].lower() for owner in owners)
        if writers is not None:
            names &= {login.lower() for login in writers}
        if pattern == "*":
            if globals_ is not None or rules:
                raise ValueError("CODEOWNERS needs exactly one leading * rule")
            globals_ = names
        else:
            if not pattern.startswith("/") or "**" in pattern or any(c in pattern for c in "?[]!\\"):
                raise ValueError("Unsupported CODEOWNERS pattern: " + pattern)
            if ".." in pattern.split("/"):
                raise ValueError("Invalid CODEOWNERS path")
            rules.append(Scope(pattern, names))
    globals_ = frozenset(set(globals_ or ()) | {login.lower() for login in admins})
    if not globals_ and require_globals:
        raise ValueError("Missing global committers in CODEOWNERS")
    return Policy(globals_, tuple(rules), frozenset(login.lower() for login in admins))


def unpack_record(comment):
    if comment.get("user", {}).get("login") != "github-actions[bot]":
        return None
    if (comment.get("performed_via_github_app") or {}).get("id") != ACTIONS_APP_ID:
        return None
    first = comment.get("body", "").splitlines()
    if not first or not first[0].startswith(MARKER) or not first[0].endswith(" -->"):
        return None
    try:
        data = json.loads(first[0][len(MARKER) : -4])
        if not isinstance(data, dict) or data.get("kind") not in ("head", "approval"):
            return None
        if type(data.get("run_id")) is not int:
            return None
        return data
    except (TypeError, ValueError):
        return None


def latest_head(records):
    heads = [r for r in records if r.get("kind") == "head"]
    if not heads:
        return None
    return max(heads, key=lambda r: (timestamp(r["seen_at"]), r["run_id"]))


def valid_approvers(pr, comments, records, policy_sha, *, admin_logins=()):
    """Only count live, unedited commands recorded for this exact head/policy."""
    head = latest_head(records)
    if not head or (head["head_sha"], head.get("base_ref"), head.get("base_sha")) != (
        pr["head"]["sha"],
        pr["base"]["ref"],
        pr["base"]["sha"],
    ):
        return set()
    live = {c["id"]: c for c in comments}
    logins = set()
    for record in records:
        if record.get("kind") != "approval":
            continue
        source = live.get(record.get("comment_id"))
        if not source or not command(source.get("body")):
            continue
        user = source.get("user", {})
        if user.get("type") != "User" or (
            user.get("id") == pr["user"]["id"] and user.get("login", "").lower() not in admin_logins
        ):
            continue
        if record.get("user_id") != user.get("id"):
            continue
        if record.get("login", "").lower() != user.get("login", "").lower():
            continue
        if record.get("source_updated_at") != source.get("updated_at"):
            continue
        if (
            record.get("head_sha") != pr["head"]["sha"]
            or record.get("base_ref") != pr["base"]["ref"]
            or record.get("base_sha") != pr["base"]["sha"]
            or record.get("policy_sha") != policy_sha
        ):
            continue
        # Same-second ordering is ambiguous; require a fresh comment instead.
        if timestamp(source["updated_at"]) <= timestamp(head["seen_at"]):
            continue
        logins.add(user["login"].lower())
    return logins


def latest_effective_reviews(reviews):
    latest = {}
    for review in reviews:
        login = review.get("user", {}).get("login", "").lower()
        if review.get("state") not in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            continue
        key = (review.get("submitted_at") or "", review["id"])
        if login not in latest or key > latest[login][0]:
            latest[login] = (key, review["state"])
    return latest


def blocking_reviews(reviews, eligible):
    return sorted(
        login
        for login, (_, state) in latest_effective_reviews(reviews).items()
        if login in eligible and state == "CHANGES_REQUESTED"
    )


class ApiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


class GitHub:
    def __init__(self, token, repo):
        self.token = token
        self.repo = repo

    def request(self, method, path, data=None):
        request = Request(
            "https://api.github.com" + path,
            data=None if data is None else json.dumps(data).encode(),
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "cann-committer-approval",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read()
                return json.loads(body) if body else None
        except HTTPError as error:
            try:
                message = json.loads(error.read()).get("message", "Request failed")
            except (ValueError, AttributeError):
                message = "Request failed"
            raise ApiError(error.code, message) from None

    def rest(self, method, path, data=None):
        return self.request(method, f"/repos/{self.repo}/" + path, data)

    def pages(self, path):
        items = []
        for page in range(1, 1001):
            separator = "&" if "?" in path else "?"
            batch = self.rest("GET", f"{path}{separator}per_page=100&page={page}")
            if not isinstance(batch, list):
                raise ValueError("Expected a paginated list")
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise ValueError("Pagination limit exceeded")

    def object_pages(self, path, key):
        items = []
        for page in range(1, 101):
            separator = "&" if "?" in path else "?"
            batch = self.rest("GET", f"{path}{separator}per_page=100&page={page}")
            rows = batch.get(key) if isinstance(batch, dict) else None
            if not isinstance(rows, list):
                raise ValueError("Expected a paginated object")
            items.extend(rows)
            if len(rows) < 100:
                return items
        raise ValueError("Pagination limit exceeded")

    def artifact_json(self, artifact_id, filename):
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, *args):
                return None

        request = Request(
            f"https://api.github.com/repos/{self.repo}/actions/artifacts/{artifact_id}/zip",
            headers={"Authorization": "Bearer " + self.token, "Accept": "application/vnd.github+json"},
        )
        try:
            build_opener(NoRedirect()).open(request, timeout=30)
        except HTTPError as error:
            if error.code != 302:
                raise ValueError("Smoke artifact is unavailable") from None
            location = error.headers.get("Location", "")
        else:
            raise ValueError("Unexpected artifact response")
        parsed = urlparse(location)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not host.endswith((".blob.core.windows.net", ".actions.githubusercontent.com")):
            raise ValueError("Untrusted artifact location")
        try:
            with build_opener(NoRedirect()).open(location, timeout=30) as response:
                compressed = response.read(2 * 1024 * 1024 + 1)
            if len(compressed) > 2 * 1024 * 1024:
                raise ValueError("oversized artifact")
            with ZipFile(io.BytesIO(compressed)) as archive:
                entries = archive.infolist()
                if len(entries) != 1 or entries[0].filename != filename or entries[0].file_size > 1024 * 1024:
                    raise ValueError("unexpected artifact contents")
                data = archive.read(entries[0])
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("invalid artifact")
            return result
        except Exception:
            raise ValueError("Cannot verify Smoke artifact") from None


class Handler:
    def __init__(self, api, event, event_name, run_id, signing_key, checks_api=None):
        if not signing_key:
            raise ValueError("Missing protected approval signing key")
        self.api = api
        self.checks_api = checks_api or api
        self.signing_key = signing_key
        self.event = event
        self.event_name = event_name
        self.run_id = run_id
        self.number = (event.get("issue") or event.get("pull_request") or {})["number"]
        self.default = event["repository"]["default_branch"]

    def pr(self):
        return self.api.rest("GET", f"pulls/{self.number}")

    def policy_source(self):
        # Fetch from the current default branch, never from the PR head.
        ref = self.api.rest("GET", "git/ref/heads/" + quote(self.default, safe=""))
        data = self.api.rest("GET", "contents/.github/CODEOWNERS?" + urlencode({"ref": ref["object"]["sha"]}))
        text = base64.b64decode(data["content"]).decode("utf-8")
        return text, data["sha"]

    def policy(self, candidates=(), *, source=None):
        text, policy_sha = source or self.policy_source()
        admins, writers = set(), set()
        for login in set(candidates):
            if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", login):
                raise ValueError("Invalid approval identity")
            try:
                permission = self.api.rest("GET", f"collaborators/{quote(login, safe='')}/permission").get("permission")
            except ApiError as error:
                if error.status == 404:
                    continue
                raise
            if permission == "admin":
                admins.add(login)
            if permission in {"admin", "maintain", "write"}:
                writers.add(login)
        return parse_policy(text, admins=admins, writers=writers, require_globals=False), policy_sha

    def comments(self):
        return self.api.pages(f"issues/{self.number}/comments")

    def records(self, comments):
        records = []
        for comment in comments:
            record = unpack_record(comment)
            if record is None:
                continue
            if record.get("repository") != self.api.repo or record.get("pr_number") != self.number:
                continue
            supplied = record.get("signature")
            if not isinstance(supplied, str) or not hmac.compare_digest(supplied, signature(record, self.signing_key)):
                continue
            if (
                not re.fullmatch(r"[0-9a-f]{40}", record.get("head_sha", ""))
                or not re.fullmatch(r"[0-9a-f]{40}", record.get("base_sha", ""))
                or not isinstance(record.get("base_ref"), str)
            ):
                continue
            if record["kind"] == "head":
                timestamp(record["seen_at"])
            records.append(record)
        return records

    def post(self, body):
        return self.api.rest("POST", f"issues/{self.number}/comments", {"body": body})

    def post_record(self, record, message):
        record["run_id"] = self.run_id
        record["repository"] = self.api.repo
        record["pr_number"] = self.number
        record["signature"] = signature(record, self.signing_key)
        self.post(MARKER + json.dumps(record, separators=(",", ":")) + " -->\n" + message)

    def post_head(self, pr, seen_at):
        self.post_record(
            {
                "kind": "head",
                "head_sha": pr["head"]["sha"],
                "base_ref": pr["base"]["ref"],
                "base_sha": pr["base"]["sha"],
                "seen_at": seen_at,
            },
            f"当前提交 `{pr['head']['sha'][:12]}` 已开始检查。"
            "请 committer 评论独立一行 `/approve`；新提交或目标变化后需要重新确认。",
        )

    def files(self, pr):
        files = self.api.pages(f"pulls/{self.number}/files")
        if len(files) != pr["changed_files"]:
            raise ValueError("Cannot obtain the complete PR file list; approval is blocked")
        return files

    def record(self):
        pr = self.pr()
        if pr["state"] != "open":
            return
        if self.event_name == "pull_request_target":
            action = self.event["action"]
            if action == "edited" and "base" not in self.event.get("changes", {}):
                return
            if action not in ("opened", "reopened", "synchronize", "converted_to_draft", "ready_for_review", "edited"):
                return
            snapshot = self.event["pull_request"]
            if (snapshot["head"]["sha"], snapshot["base"]["ref"], snapshot["base"]["sha"]) != (
                pr["head"]["sha"],
                pr["base"]["ref"],
                pr["base"]["sha"],
            ):
                return
            self.post_head(snapshot, snapshot["updated_at"])
            return
        if self.event_name != "issue_comment" or self.event["action"] == "deleted":
            return
        source = self.event["comment"]
        if not command(source.get("body")):
            return
        user = source["user"]
        if user.get("type") != "User" or self.event["sender"]["id"] != user["id"]:
            return
        if pr["draft"]:
            self.post("草稿 PR 不接受批准，请先标记为 Ready for review，再评论 `/approve`。")
            return
        policy, policy_sha = self.policy({user["login"]})
        scopes = policy.scopes(self.files(pr))
        login = user["login"].lower()
        if user["id"] == pr["user"]["id"] and login not in policy.admins:
            self.post(f"@{user['login']}，只有仓库管理员可以批准自己的 PR。")
            return
        covered = [scope.pattern for scope in scopes if login in policy.globals or login in scope.owners]
        if not covered:
            self.post(f"@{user['login']} 不在本次改动对应的 committer 列表中，此评论不计入批准。")
            return
        comments = self.comments()
        live = next((c for c in comments if c["id"] == source["id"]), None)
        if not live or live.get("updated_at") != source["updated_at"] or not command(live.get("body")):
            return
        records = self.records(comments)
        head = latest_head(records)
        if not head or (head["head_sha"], head.get("base_ref"), head.get("base_sha")) != (
            pr["head"]["sha"],
            pr["base"]["ref"],
            pr["base"]["sha"],
        ):
            self.post_head(pr, datetime.now(timezone.utc).isoformat())  # noqa: UP017 - Python 3.10
            self.post("PR 目标版本已变化，请确认最新版本后重新评论 `/approve`。")
            return
        if timestamp(source["updated_at"]) <= timestamp(head["seen_at"]):
            self.post("本次评论早于当前提交的检查快照，未计入批准。请等待检查初始化后重新评论 `/approve`。")
            return
        record = {
            "kind": "approval",
            "head_sha": pr["head"]["sha"],
            "policy_sha": policy_sha,
            "base_ref": pr["base"]["ref"],
            "base_sha": pr["base"]["sha"],
            "comment_id": source["id"],
            "source_updated_at": source["updated_at"],
            "user_id": user["id"],
            "login": user["login"],
        }
        if any(all(existing.get(key) == value for key, value in record.items()) for existing in records):
            return
        current = self.pr()
        if (
            (current["head"]["sha"], current["base"]["ref"], current["base"]["sha"])
            != (pr["head"]["sha"], pr["base"]["ref"], pr["base"]["sha"])
            or current["state"] != "open"
            or current["draft"]
        ):
            self.post("批准处理期间 PR 已变化，请针对最新提交重新评论 `/approve`。")
            return
        self.post_record(
            record,
            f"已记录 @{user['login']} 对 `{pr['head']['sha'][:12]}` 的批准，"
            f"覆盖：{', '.join(covered)}。完整结果见 `{CHECK_NAME}` 检查。",
        )

    def approval_snapshot(self, pr):
        if pr["draft"]:
            raise ValueError("PR must be ready for review")
        comments = self.comments()
        records = self.records(comments)
        reviews = self.api.pages(f"pulls/{self.number}/reviews")
        policy_source = self.policy_source()
        live_comments = {comment["id"]: comment for comment in comments}
        candidates = {
            record["login"]
            for record in records
            if record.get("kind") == "approval"
            and isinstance(record.get("login"), str)
            and record.get("head_sha") == pr["head"]["sha"]
            and record.get("base_ref") == pr["base"]["ref"]
            and record.get("base_sha") == pr["base"]["sha"]
            and record.get("policy_sha") == policy_source[1]
            and record.get("comment_id") in live_comments
            and command(live_comments[record["comment_id"]].get("body"))
            and record.get("source_updated_at") == live_comments[record["comment_id"]].get("updated_at")
        }
        candidates.update(
            login
            for login, (_, state) in latest_effective_reviews(reviews).items()
            if state == "CHANGES_REQUESTED" and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", login)
        )
        policy, policy_sha = self.policy(candidates, source=policy_source)
        scopes = policy.scopes(self.files(pr))
        if not scopes:
            raise ValueError("PR contains no changed files")
        logins = valid_approvers(pr, comments, records, policy_sha, admin_logins=policy.admins)
        missing = policy.coverage(scopes, logins)
        eligible = set(policy.globals)
        for scope in scopes:
            eligible.update(scope.owners)
        blocked = blocking_reviews(reviews, eligible)
        summary = "批准人：" + (", ".join(sorted(logins)) or "暂无")
        summary += "\n\n改动范围：" + ", ".join(scope.pattern for scope in scopes)
        if missing:
            summary += "\n\n尚缺批准：" + ", ".join(missing)
        if blocked:
            summary += "\n\n仍有 Changes requested：" + ", ".join(blocked)
        return not missing and not blocked, summary, policy_sha

    def assert_version(self, expected, current):
        if (
            current["state"] != "open"
            or current["draft"]
            or current["head"]["sha"] != expected["head"]["sha"]
            or current["base"]["ref"] != expected["base"]["ref"]
            or current["base"]["sha"] != expected["base"]["sha"]
        ):
            raise ValueError("PR Head/Base changed; a new evaluation is required")

    def smoke_passed(self, pr):
        # These modules are loaded from the checked-out default branch, never the PR.
        try:
            from blue_catalog import BLUE_SMOKE_SUITE
            from native_result import context, description
            from native_source import REPO, REPO_ID, build_source
            from prepare_request import create_plan
        except ImportError:
            return False
        if self.api.repo != REPO:
            return False
        head, base = pr["head"]["sha"], pr["base"]["sha"]
        statuses = self.api.pages(f"commits/{head}/statuses")
        matching = [status for status in statuses if status.get("context") == context(self.number)]
        if not matching:
            return False
        status = max(matching, key=lambda item: (item.get("created_at", ""), item.get("id", 0)))
        if (
            status.get("state") != "success"
            or status.get("creator", {}).get("id") != 41898282
            or status.get("description") != description(base, list(BLUE_SMOKE_SUITE), pr["base"]["ref"])
        ):
            return False
        pattern = rf"https://github\.com/{re.escape(REPO)}/actions/runs/([1-9][0-9]*)/attempts/([1-9][0-9]*)"
        match = re.fullmatch(pattern, status.get("target_url") or "")
        if not match:
            return False
        run_id, attempt = map(int, match.groups())
        run = self.api.rest("GET", f"actions/runs/{run_id}")
        workflow = self.api.rest("GET", "actions/workflows/lite_actions_dispatcher_smoke.yml")
        if (
            run.get("id") != run_id
            or run.get("run_attempt") != attempt
            or run.get("repository", {}).get("id") != REPO_ID
            or run.get("workflow_id") != workflow.get("id")
            or workflow.get("state") != "active"
            or run.get("path") != ".github/workflows/lite_actions_dispatcher_smoke.yml"
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
        ):
            return False
        if run.get("event") == "pull_request_target":
            title = rf"lite-vllm-pr-{self.number}-{head}-([0-9a-f]{{40}})"
            title_match = re.fullmatch(title, run.get("display_title") or "")
            if (
                not title_match
                or run.get("head_sha") != head
                or run.get("head_repository", {}).get("id") != pr["head"]["repo"]["id"]
            ):
                return False
        elif run.get("event") == "workflow_dispatch":
            title = rf"lite-vllm-recheck-{self.number}-([0-9a-f]{{40}})-[0-9a-f]{{32}}"
            title_match = re.fullmatch(title, run.get("display_title") or "")
            if (
                not title_match
                or run.get("head_branch") != self.default
                or run.get("head_repository", {}).get("id") != REPO_ID
            ):
                return False
        else:
            return False
        control = title_match.group(1)
        if run["event"] == "workflow_dispatch" and run.get("head_sha") != control:
            return False
        default_sha = self.api.rest("GET", "git/ref/heads/" + quote(self.default, safe=""))["object"]["sha"]
        if control != default_sha:
            comparison = self.api.rest("GET", f"compare/{control}...{default_sha}")
            if comparison.get("status") != "ahead" or comparison.get("merge_base_commit", {}).get("sha") != control:
                return False

        artifacts = self.api.object_pages(f"actions/runs/{run_id}/artifacts", "artifacts")
        documents = {}
        for name, filename in (
            (f"lite-ci-source-{attempt}", "source.json"),
            (f"lite-ci-request-{attempt}", "ci-request.json"),
        ):
            candidates = [item for item in artifacts if item.get("name") == name and item.get("expired") is False]
            if (
                len(candidates) != 1
                or type(candidates[0].get("id")) is not int
                or candidates[0]["id"] < 1
                or candidates[0].get("workflow_run", {}).get("id") != run_id
                or candidates[0].get("workflow_run", {}).get("repository_id") != REPO_ID
            ):
                return False
            documents[filename] = self.api.artifact_json(candidates[0]["id"], filename)
        expected_plan = create_plan("pull_request_target", {"pull_request": pr}, "", run_id, attempt)
        prior_control = os.environ.get("GITHUB_WORKFLOW_SHA")
        os.environ["GITHUB_WORKFLOW_SHA"] = control
        try:
            expected_source = build_source(
                expected_plan, {"pull_request": pr}, lambda path: self.api.rest("GET", path[1:])
            )
        finally:
            if prior_control is None:
                os.environ.pop("GITHUB_WORKFLOW_SHA", None)
            else:
                os.environ["GITHUB_WORKFLOW_SHA"] = prior_control
        if documents["ci-request.json"] != expected_plan or documents["source.json"] != expected_source:
            return False
        jobs = self.api.object_pages(f"actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")
        expected_names = ["prepare", "Lite-Actions-Smoke"] + [case["case_id"] for case in expected_plan["cases"]]
        for name in expected_names:
            matching_jobs = [job for job in jobs if job.get("name") == name]
            if len(matching_jobs) != 1 or any(
                matching_jobs[0].get(key) != value
                for key, value in (
                    ("run_id", run_id),
                    ("run_attempt", attempt),
                    ("status", "completed"),
                    ("conclusion", "success"),
                )
            ):
                return False
        return True

    def evaluate(self):
        pr = self.pr()
        if pr["state"] != "open":
            return
        check = self.checks_api.rest(
            "POST",
            "check-runs",
            {
                "name": CHECK_NAME,
                "head_sha": pr["head"]["sha"],
                "status": "in_progress",
                "details_url": f"https://github.com/{self.api.repo}/actions/runs/{self.run_id}",
                "external_id": f"{self.number}:{self.run_id}",
            },
        )
        try:
            approved, summary, policy_sha = self.approval_snapshot(pr)
            current = self.pr()
            self.assert_version(pr, current)
            if approved:
                approved, summary, current_policy = self.approval_snapshot(current)
                if not approved or current_policy != policy_sha:
                    raise ValueError("Approval or policy changed during evaluation")
            if approved and not self.smoke_passed(current):
                self.finish_check(check["id"], False, "等待当前版本 Smoke", "可信完整 Smoke 尚未通过")
                return
            if approved:
                self.merge(pr, check["id"], policy_sha, summary)
            else:
                self.finish_check(check["id"], False, "等待 committer 批准", summary)
        except Exception as error:
            self.finish_check(check["id"], False, "检查未能完成，禁止合入", str(error))
            raise

    def invalidate(self):
        pr = self.pr()
        if pr["state"] != "open":
            return
        check = self.checks_api.rest(
            "POST",
            "check-runs",
            {
                "name": CHECK_NAME,
                "head_sha": pr["head"]["sha"],
                "status": "in_progress",
                "external_id": f"{self.number}:{self.run_id}",
            },
        )
        self.finish_check(check["id"], False, "审批记录失败", "等待重新验证")

    def finish_check(self, check_id, success, title, summary):
        self.checks_api.rest(
            "PATCH",
            f"check-runs/{check_id}",
            {
                "status": "completed",
                "conclusion": "success" if success else "failure",
                "output": {"title": title, "summary": summary},
            },
        )

    def merge(self, pr, check_id, policy_sha, summary):
        self.finish_check(check_id, True, "批准已齐全", summary)
        current = self.pr()
        self.assert_version(pr, current)
        approved, _, current_policy = self.approval_snapshot(current)
        if not approved or current_policy != policy_sha:
            raise ValueError("Approval or policy was revoked before merge")
        if not self.smoke_passed(current):
            raise ValueError("Current PR Smoke result changed before merge")
        try:
            result = self.api.rest(
                "PUT",
                f"pulls/{self.number}/merge",
                {
                    "merge_method": "squash",
                    "sha": pr["head"]["sha"],
                },
            )
        except ApiError as error:
            if error.status not in (405, 409, 422):
                raise
            self.finish_check(check_id, False, "等待重新检查", "平台检查未通过或授权已变化")
            print(f"PR #{self.number} remains blocked by GitHub checks; will re-evaluate")
            return
        if not result.get("merged"):
            raise ValueError("GitHub did not merge the PR")
        print(f"Merged PR #{self.number}: {result['sha']}")


def initialize(api, event, run_id, signing_key, seen_at=None):
    seen_at = seen_at or datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.10
    failed = []
    for item in api.pages("pulls?state=open"):
        synthetic = {"pull_request": {"number": item["number"]}, "repository": event["repository"]}
        handler = Handler(api, synthetic, "initialize", run_id, signing_key)
        try:
            pr = handler.pr()
            if pr["state"] == "open":
                handler.post_head(pr, seen_at)
        except Exception as error:
            print(f"PR #{item['number']} initialization failed: {type(error).__name__}")
            failed.append(item["number"])
    if failed:
        raise ValueError("PR initialization failed: " + ", ".join(map(str, failed)))


def reconcile(api, checks_api, event, run_id, signing_key):
    pulls = api.pages("pulls?state=open")
    failed = []
    for item in pulls:
        synthetic = {"pull_request": {"number": item["number"]}, "repository": event["repository"]}
        handler = Handler(api, synthetic, "reconcile", run_id, signing_key, checks_api)
        try:
            pr = handler.pr()
            if pr["state"] != "open":
                continue
            records = handler.records(handler.comments())
            head = latest_head(records)
            if not head or (head["head_sha"], head.get("base_ref"), head.get("base_sha")) != (
                pr["head"]["sha"],
                pr["base"]["ref"],
                pr["base"]["sha"],
            ):
                handler.post_head(pr, datetime.now(timezone.utc).isoformat())  # noqa: UP017 - Python 3.10
                continue
            if pr.get("draft") or not any(
                record.get("kind") == "approval" and record.get("head_sha") == pr["head"]["sha"] for record in records
            ):
                continue
            handler.evaluate()
        except Exception as error:
            # A broken PR must not prevent other PRs from being re-evaluated.
            print(f"PR #{item['number']} reconciliation failed: {type(error).__name__}")
            failed.append(item["number"])
    if failed:
        raise ValueError("Approval reconciliation failed for PRs: " + ", ".join(map(str, failed)))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("record", "evaluate", "invalidate", "reconcile", "initialize"):
        raise ValueError("Usage: comment_approve.py record|evaluate|invalidate|reconcile|initialize")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    api = GitHub(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])
    if sys.argv[1] == "initialize":
        initialize(api, event, int(os.environ["GITHUB_RUN_ID"]), os.environ["APPROVAL_SIGNING_KEY"].encode())
        return
    checks_api = GitHub(os.environ["CHECKS_TOKEN"], api.repo) if sys.argv[1] != "record" else None
    if sys.argv[1] == "reconcile":
        reconcile(
            api,
            checks_api,
            event,
            int(os.environ["GITHUB_RUN_ID"]),
            os.environ["APPROVAL_SIGNING_KEY"].encode(),
        )
        return
    handler = Handler(
        api,
        event,
        os.environ["GITHUB_EVENT_NAME"],
        int(os.environ["GITHUB_RUN_ID"]),
        os.environ["APPROVAL_SIGNING_KEY"].encode(),
        checks_api,
    )
    getattr(handler, sys.argv[1])()


if __name__ == "__main__":
    main()
