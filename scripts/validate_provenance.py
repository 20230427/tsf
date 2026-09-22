#!/usr/bin/env python
"""Validate experiment JSON provenance without modifying result files.

Exit codes: 0 when every input verifies; 2 when inputs are well-formed legacy
records whose provenance is unavailable; 1 for malformed or schema-invalid
records. Mixed inputs use the highest-severity code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import validate_provenance_record  # noqa: E402


def validate_path(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "reasons": [str(exc)]}
    return validate_provenance_record(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true",
                        help="emit one machine-readable JSON array")
    args = parser.parse_args()

    reports = []
    for path in args.paths:
        report = validate_path(path)
        reports.append({"path": str(path), **report})
    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print(f"[{report['status']}] {report['path']}")
            for reason in report["reasons"]:
                print(f"  - {reason}")

    statuses = {report["status"] for report in reports}
    if "invalid" in statuses:
        return 1
    if "legacy_unverifiable" in statuses:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
