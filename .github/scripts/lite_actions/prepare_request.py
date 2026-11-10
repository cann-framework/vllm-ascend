#!/usr/bin/env python3
"""Build the versioned plan without importing fork Python in control jobs."""

import hashlib
import json
import os
import re
from pathlib import Path

from blue_catalog import BLUE_CASES, BLUE_DEFAULTS, BLUE_SMOKE_SUITE

REPO = "cann-framework/vllm-ascend"
WORKFLOW = "lite_actions_dispatcher_smoke.yml"
A5_CASES = {"dispatch_smoke_a5_8p": (1, 8, 120), "dispatch_smoke_a5_32p": (4, 8, 240)}
CASES = {**A5_CASES, **BLUE_CASES}
DEFAULTS = dict(worker="yellow", target_pool="a5_shared_yellow", docker="vllm-kimi-wangjinyin")


def selectors(event, payload, value):
    if event == "pull_request_target":
        pr = payload["pull_request"]
        base, head = pr["base"], pr["head"]
        if (
            base["repo"]["full_name"] != REPO
            or not base.get("ref")
            or not head.get("repo", {}).get("full_name")
            or pr["state"] != "open"
        ):
            raise ValueError("invalid PR source or target repository")
        sha = head["sha"]
        ids = list(BLUE_SMOKE_SUITE)
    elif event == "workflow_dispatch":
        if not os.environ.get("GITHUB_REF", "").startswith("refs/heads/"):
            raise ValueError("manual run requires a branch")
        sha = os.environ["GITHUB_SHA"]
        if not value:
            ids = list(BLUE_SMOKE_SUITE)
        else:
            values = json.loads(value)
            if not isinstance(values, list) or not 1 <= len(values) <= 256:
                raise ValueError("invalid smoke test selection")
            if any(not isinstance(x, dict) or set(x) != {"test_id"} for x in values):
                raise ValueError("only fixed test_id selectors supported")
            ids = [x["test_id"] for x in values]
    else:
        raise ValueError("unsupported GitHub event")
    if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError("invalid source SHA")
    if not ids or len(ids) > 256 or any(x not in CASES for x in ids):
        raise ValueError("unknown smoke case")
    if any(ids.count(x) > 1 and (x not in BLUE_CASES or BLUE_CASES[x].ci_kind != "npu_smoke") for x in set(ids)):
        raise ValueError("duplicate non-smoke case")
    if any(x in BLUE_CASES for x in ids) and any(x in A5_CASES for x in ids):
        raise ValueError("Blue and Yellow cases cannot share a native source plan")
    return sha, ids


def create_plan(event, payload, value, run_id, attempt):
    sha, ids = selectors(event, payload, value)
    defaults = BLUE_DEFAULTS if ids[0] in BLUE_CASES else DEFAULTS
    items = []
    for i, test_id in enumerate(ids):
        if test_id in BLUE_CASES:
            test = BLUE_CASES[test_id]
            nodes, ngpu, timeout = test.nnodes, test.ngpu, test.timeout
            label, kind = test.test_descr, test.ci_kind
            overrides = test.ci_overrides or {}
            if not isinstance(overrides, dict) or set(overrides) - {"worker", "target_pool", "docker"}:
                raise ValueError("unknown case override")
            if kind == "worker_probe" and set(overrides) & {"target_pool", "docker"}:
                raise ValueError("probe cannot request resources")
            config = {**defaults, **overrides}
        else:
            nodes, ngpu, timeout = CASES[test_id]
            label, kind = f"A5 HCCL distributed smoke {nodes * ngpu}P", "npu_smoke"
            config = defaults
        items.append(
            {
                "case_id": f"case_{i:03d}_{test_id}",
                "order": i,
                "test_id": test_id,
                "label": label,
                "kind": kind,
                "worker": config["worker"],
                "target_pool": config["target_pool"] if kind != "worker_probe" else None,
                "resources": {"nodes": nodes, "devices_per_node": ngpu},
                "docker": config["docker"] if kind != "worker_probe" else None,
                "execution_timeout_seconds": timeout,
            }
        )
    plan = {
        "schema": 3,
        "repo": REPO,
        "workflow": WORKFLOW,
        "run_id": run_id,
        "attempt": attempt,
        "sha": sha,
        "defaults": defaults,
        "cases": items,
    }
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return plan


def main():
    if os.environ["GITHUB_REPOSITORY"] != REPO:
        raise ValueError("untrusted repository")
    payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    effective_event = os.environ["GITHUB_EVENT_NAME"]
    source_pr = os.environ.get("CI_SOURCE_PR", "")
    if source_pr:
        if effective_event != "workflow_dispatch" or not re.fullmatch(r"[1-9][0-9]{0,8}", source_pr):
            raise ValueError("invalid native PR recheck")
        from native_source import current_pr

        pr, head, base = current_pr(int(source_pr))
        if (head, base) != (
            os.environ.get("CI_EXPECTED_HEAD"),
            os.environ.get("CI_EXPECTED_BASE"),
        ):
            raise ValueError("STALE_RECHECK_INTENT")
        payload = {"pull_request": pr}
        effective_event = "pull_request_target"
    plan = create_plan(
        effective_event,
        payload,
        os.environ.get("CI_TEST_CASES", ""),
        int(os.environ["GITHUB_RUN_ID"]),
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
    )
    out = Path(".ci-request")
    out.mkdir(exist_ok=True)
    native = plan["cases"][0]["test_id"] in BLUE_CASES
    if native:
        from native_source import prepare_source

        source = prepare_source(plan, payload)
        out.joinpath("source.json").write_text(json.dumps(source, sort_keys=True) + "\n")
    (out / "ci-request.json").write_text(json.dumps(plan, separators=(",", ":"), ensure_ascii=False) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("native_source=" + str(native).lower() + "\n")
        output.write("matrix=" + json.dumps({"include": plan["cases"]}, separators=(",", ":")) + "\n")
        output.write("sha=" + plan["sha"] + "\n")
        output.write("plan_digest=" + plan["plan_digest"] + "\n")


if __name__ == "__main__":
    main()
