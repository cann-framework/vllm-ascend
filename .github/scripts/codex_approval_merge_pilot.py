"""Isolated CPU approval pilot: the simulated merge never runs Smoke."""

import os
import time

from comment_approve import GitHub, Handler, initialize, latest_head

REPO = "cann-framework/vllm-ascend"
REPO_ID = 1438625957
CONTROLLER = "codex/approval-merge-controller-20261110"
TARGET = "codex/approval-merge-target-20261110"
SOURCE = "codex/approval-merge-source-20261110"


def main():
    if (
        os.environ["GITHUB_REPOSITORY"] != REPO
        or os.environ["GITHUB_REF"] != "refs/heads/" + CONTROLLER
    ):
        raise ValueError("Wrong isolated controller branch")
    value = os.environ["PILOT_PR"]
    if not value.isascii() or not value.isdigit() or int(value) < 1:
        raise ValueError("Invalid test PR number")
    number = int(value)
    token = os.environ["GITHUB_TOKEN"]
    api = GitHub(token, REPO)
    checks = GitHub(os.environ["CHECKS_TOKEN"], REPO)
    pr = api.rest("GET", f"pulls/{number}")
    if (
        pr.get("state") != "open"
        or pr.get("user", {}).get("login") != "depeng1994"
        or pr.get("base", {}).get("ref") != TARGET
        or pr.get("head", {}).get("ref") != SOURCE
        or (pr.get("base", {}).get("repo") or {}).get("id") != REPO_ID
        or (pr.get("head", {}).get("repo") or {}).get("id") != REPO_ID
    ):
        raise ValueError("The PR is not the isolated test PR")
    mode = os.environ["PILOT_MODE"]
    event = {
        "repository": {"default_branch": "main"},
        "pull_request": {"number": number},
    }
    run_id = int(os.environ["GITHUB_RUN_ID"])
    key = os.environ["APPROVAL_SIGNING_KEY"].encode()
    if mode == "initialize":

        class ScopedGitHub(GitHub):
            def pages(self, path):
                return [pr] if path == "pulls?state=open" else super().pages(path)

        initialize(ScopedGitHub(token, REPO), event, run_id, key)
        handler = Handler(api, event, "pilot", run_id, key, checks)
        head = latest_head(handler.records(handler.comments()))
        if not head or (head["head_sha"], head["base_sha"], head["base_ref"]) != (
            pr["head"]["sha"],
            pr["base"]["sha"],
            pr["base"]["ref"],
        ):
            raise ValueError("Isolated PR snapshot was not created")
        print("PASS: initialized only the isolated PR")
        return
    if mode == "record":
        comment_id = os.environ["PILOT_COMMENT"]
        if not comment_id.isascii() or not comment_id.isdigit():
            raise ValueError("Invalid test comment ID")
        comment = api.rest("GET", f"issues/comments/{int(comment_id)}")
        if (
            comment.get("issue_url")
            != f"https://api.github.com/repos/{REPO}/issues/{number}"
            or comment.get("user", {}).get("login") != "depeng1994"
            or comment.get("body") != "/approve"
        ):
            raise ValueError("Comment is not the authorized isolated PR command")
        event.update(
            action="created",
            issue={"number": number},
            comment=comment,
            sender=comment["user"],
        )
        handler = Handler(api, event, "issue_comment", run_id, key, checks)
        handler.record()
        if not any(
            record.get("comment_id") == int(comment_id)
            for record in handler.records(handler.comments())
        ):
            raise ValueError("The test approval was not recorded")
        print("PASS: processed the real test PR approval comment")
        return
    handler = Handler(api, event, "pilot", run_id, key, checks)
    if mode in ("withdrawn", "simulated_merge"):
        handler.smoke_passed = (
            lambda pr: True
        )  # Deliberate CPU-only simulation; never real Smoke evidence.
    if mode not in ("no_smoke", "withdrawn", "simulated_merge"):
        raise ValueError("Invalid isolated scenario")
    approved, _, _ = handler.approval_snapshot(pr)
    if approved != (mode != "withdrawn"):
        raise ValueError("Test approval state is not the expected one")
    if mode == "no_smoke" and handler.smoke_passed(pr):
        raise ValueError("The isolated PR unexpectedly has a passing Smoke")
    for _ in range(4 if mode == "simulated_merge" else 1):
        handler.evaluate()
        state = api.rest("GET", f"pulls/{number}")
        if state.get("merged"):
            if mode != "simulated_merge":
                raise ValueError("Unexpected test PR merge")
            print("PASS: isolated PR merged using deliberately simulated Smoke")
            return
        if mode == "simulated_merge":
            time.sleep(5)
    if mode == "simulated_merge":
        raise ValueError("Isolated PR remained blocked by platform rules")
    check_runs = checks.rest(
        "GET", f"commits/{pr['head']['sha']}/check-runs?check_name=committer-approval"
    )
    ours = [
        check
        for check in check_runs.get("check_runs", [])
        if check.get("app", {}).get("id") == 5575236
    ]
    if (
        not ours
        or max(ours, key=lambda check: check["id"]).get("conclusion") != "failure"
    ):
        raise ValueError("Expected a failing approval check")
    print("PASS: isolated PR stayed open and approval check failed")


if __name__ == "__main__":
    main()
