#!/usr/bin/env python3
"""Write a byte-exact deterministic payload for external-probe qualification."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = b"GS_GCP_STAGE0_5_PROBE_SYNTHETIC_V1\x00" + bytes(range(256))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True)
    args.output.write_bytes(payload)
    print(json.dumps({"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
