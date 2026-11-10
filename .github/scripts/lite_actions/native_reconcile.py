"""Compensate Base-only updates using trusted main code and bounded API reads.

A pending status is a durable dispatch-intent claim. Never blindly retry POST
when its response was lost: an operator can explicitly re-run a full PR intent.
This deliberately favors no duplicate hardware execution over guessing success.
"""

import hashlib
import json
import os
import urllib.error

from blue_catalog import BLUE_SMOKE_SUITE
from native_result import context, description, post
from native_source import REPO, api, current_pr

WORKFLOW = "lite_actions_dispatcher_smoke.yml"


def reconcile(get=api, send=post):
    control = os.environ["GITHUB_WORKFLOW_SHA"]
    if get("/branches/main")["commit"]["sha"] != control:
        raise ValueError("reconciler control is no longer main")
    pulls = []
    for page in range(1, 11):
        rows = get(f"/pulls?state=open&per_page=100&page={page}")
        if not isinstance(rows, list):
            raise ValueError("invalid open PR response")
        pulls.extend(rows)
        if len(rows) < 100:
            break
    else:
        raise ValueError("open PR visibility budget incomplete")
    results = []
    for item in pulls:
        number = item["number"]
        head = item.get("head", {}).get("sha")
        if not isinstance(head, str) or len(head) != 40:
            continue
        try:
            statuses = get("/commits/" + head + "/status?per_page=100").get("statuses", [])
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            try:
                current_pr(number, get)
            except ValueError as changed:
                if str(changed) == "unauthorized Blue PR source":
                    continue
                raise
            raise
        prior = next((x for x in statuses if x.get("context") == context(number)), None)
        # Existing versions may need a Base-only recheck; do not start every
        # historical open PR when the workflow is enabled for more forks.
        if not prior:
            continue
        try:
            pr, current_head, base = current_pr(number, get)
        except ValueError as exc:
            if str(exc) != "unauthorized Blue PR source":
                raise
            continue
        if head != current_head:
            continue
        tests = list(BLUE_SMOKE_SUITE)
        desc = description(base, tests, pr["base"]["ref"])
        if prior and prior.get("description") == desc:
            results.append({"pr": number, "state": "EXISTING_VERSION_OR_DISPATCH_UNKNOWN"})
            continue
        nonce = hashlib.sha256(
            json.dumps([number, head, base, control, tests], separators=(",", ":")).encode()
        ).hexdigest()[:32]
        # Invalidate old green before creating the new version's execution.
        send(
            "/statuses/" + head,
            {
                "state": "pending",
                "context": context(number),
                "description": desc,
                "target_url": f"https://github.com/{REPO}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        )
        response = send(
            f"/actions/workflows/{WORKFLOW}/dispatches",
            {
                "ref": "main",
                "return_run_details": True,
                "inputs": {
                    "source_pr": str(number),
                    "expected_head_sha": head,
                    "expected_base_sha": base,
                    "bridge_nonce": nonce,
                },
            },
        )
        results.append(
            {
                "pr": number,
                "state": "DISPATCHED",
                "nonce": nonce,
                "run_id": response.get("workflow_run_id") if isinstance(response, dict) else None,
            }
        )
    return results


if __name__ == "__main__":
    print(json.dumps(reconcile(), sort_keys=True))
