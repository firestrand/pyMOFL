"""Explicit required-reference runner; ordinary imports/tests never provision data."""

import argparse
import json
from pathlib import Path

from pyMOFL.reference_validation import validate_reference_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate_reference_manifest(args.manifest, capture_root=args.capture_root)
    with args.report.open("x") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Required cases: {report['required_cases']}; "
        f"scalar: {report['scalar_executed']}; batch: {report['batch_executed']}; "
        f"statuses: {report['status_counts']}"
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
