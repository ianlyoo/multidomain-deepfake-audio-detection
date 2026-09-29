#!/usr/bin/env python3
"""Validate a submission archive and, optionally, its generated CSV."""

import argparse
import json

from deepvoicehackathon.submission import validate_archive, validate_prediction_csv

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--predictions")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    args = parser.parse_args()
    report = validate_archive(args.archive)
    result: dict[str, int | str] = {
        "archive": str(report.path),
        "compressed_bytes": report.compressed_bytes,
        "extracted_bytes": report.extracted_bytes,
        "file_count": report.file_count,
    }
    if args.predictions:
        result["prediction_rows"] = validate_prediction_csv(args.predictions, args.sample)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
