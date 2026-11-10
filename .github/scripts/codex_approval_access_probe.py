"""One-time CPU-only, read-only permission probe; creates one test-branch check."""

import os
import sys

from comment_approve import GitHub, Handler

REPO = "cann-framework/vllm-ascend"
TRIAL_BRANCH = "codex/vllm-smoke-trial-v2-20261110"
TEST_BRANCH = "codex/approval-access-probe-20261110"


def main():
    if (
        os.environ["GITHUB_REPOSITORY"] != REPO
        or os.environ["GITHUB_REF"] != "refs/heads/" + TEST_BRANCH
    ):
        raise ValueError("Probe must run only in the intended test branch")
    api = GitHub(os.environ["GH_TOKEN"], REPO)
    if sys.argv[1:] == ["read"]:
        pr = api.rest("GET", "pulls/23")
        event = {
            "repository": {"default_branch": TRIAL_BRANCH},
            "pull_request": {"number": 23},
        }
        if not Handler(
            api, event, "probe", int(os.environ["GITHUB_RUN_ID"]), b"probe"
        ).smoke_passed(pr):
            raise ValueError(
                "Existing trial result could not be verified with this workflow token"
            )
        print(
            "PASS: read current trial status, run, plan/source artifacts and all jobs"
        )
    elif sys.argv[1:] == ["check"]:
        sha = os.environ["GITHUB_SHA"]
        result = api.rest(
            "POST",
            "check-runs",
            {
                "name": "codex-approval-access-probe",
                "head_sha": sha,
                "status": "completed",
                "conclusion": "success",
                "output": {
                    "title": "CPU access probe",
                    "summary": "Checks App access to this test branch was verified.",
                },
            },
        )
        if result.get("app", {}).get("id") != 5575236 or result.get("head_sha") != sha:
            raise ValueError("Unexpected check identity")
        print(f"PASS: Checks App created test-branch check {result['id']}")
    else:
        raise ValueError("Expected read or check")


if __name__ == "__main__":
    main()
