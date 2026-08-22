#!/usr/bin/env python3
"""Build the non-executable MetroGS post-attempt closure receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from m3m_gcp_100k_postattempt_closure import (
    PLAN_RELATIVE,
    build_postattempt_closure_payload,
)
from m3m_gcp_lidar_artifacts import sha256_file


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(
            descriptor,
            (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    plan_path = args.plan.resolve()
    output = args.output.resolve()
    if plan_path != (repo / PLAN_RELATIVE).resolve():
        raise RuntimeError("post-attempt closure plan path mismatch")
    if output.exists():
        raise FileExistsError("post-attempt closure output already exists")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    payload = build_postattempt_closure_payload(repo=repo, plan=plan)
    write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": "PASS_POSTATTEMPT_CLOSURE_CREATED",
                "path": str(output),
                "sha256": sha256_file(output),
                "canonical_sha256": payload["canonical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
