#!/usr/bin/env python3
"""Export a model-ready training dataset from an Ibn Sina AI master dataset.

Usage:
    python scripts/export_dataset.py \
        data/master/uzbek_medical_v1.jsonl \
        data/exports/uzbek_medical_v1.jsonl

A master record carries Ibn Sina metadata alongside the conversation:

    {"id": "uz-med-000001", "language": "uz-Latn", "category": "general_medical",
     "source": "manual", "review_status": "draft", "messages": [...]}

The export keeps only the part a trainer consumes:

    {"messages": [...]}

Only records with review_status "approved" are exported. Records still in
draft, merely reviewed, or rejected are counted in the summary but never reach a
training export. Approval is a human decision recorded in the master file; this
script only reads it, it never sets it.

Every record carries two independent human reviews:

    "reviews": {
      "language": {"status": "passed", "reviewer": "language-reviewer-001",
                   "reviewed_at": "2026-09-03T10:30:00Z", "notes": null},
      "medical":  {"status": "passed", "reviewer": "medical-reviewer-001",
                   "reviewed_at": "2026-09-03T11:00:00Z",
                   "notes": "Wording is cautious and refers the reader to a doctor."}
    }

Each review is "pending", "passed" or "failed". A pending review may leave
reviewer, reviewed_at and notes null. A passed review needs a reviewer and a
timezone-aware ISO 8601 timestamp. A failed review needs those plus notes saying
what the problem was.

A record may only be approved when BOTH reviews passed. The reverse does not
hold: two passed reviews never make a record approved by themselves. Nothing in
this script writes to a master file or changes a review_status — approval stays
an explicit human action, and a record with both reviews passed but a status of
"reviewed" is perfectly valid master data that simply does not export.

Nothing here judges whether a review was any good, only that one is recorded.

The whole master file is checked first — required metadata fields, known values,
and duplicate ids across every record regardless of status. If any record fails,
or if no record is approved, nothing is written and an existing export file is
left byte-for-byte unchanged. The export is also written atomically, so a failure
part-way through a write cannot leave a truncated file behind.

Conversation contents are checked separately by validate_dataset.py and
scan_sensitive_data.py, which should be run on the exported file afterwards.

Exit codes:
    0  export succeeded
    1  master dataset invalid, or no approved records
    2  input / output / file error

Standard library only.
"""

import argparse
import sys

import dataset_common
from dataset_common import EXPORTABLE_REVIEW_STATUS, read_master, write_jsonl_atomic

# Order the status counts are reported in.
REPORTED_STATUSES = ("approved", "draft", "reviewed", "rejected")


def count_statuses(records):
    """Count records per review status. Statuses are already known to be valid."""
    counts = dict.fromkeys(dataset_common.ALLOWED_REVIEW_STATUSES, 0)
    for record in records:
        counts[record["review_status"]] += 1
    return counts


def write_export(records, path):
    """Write the model-ready file: one {"messages": [...]} object per line."""
    write_jsonl_atomic([{"messages": r["messages"]} for r in records], path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a model-ready JSONL dataset from an Ibn Sina AI "
        "master dataset."
    )
    parser.add_argument("master", help="path to the master .jsonl dataset")
    parser.add_argument("export", help="path to write the model-ready .jsonl dataset")
    args = parser.parse_args(argv)

    print("Exporting dataset")
    print()
    print("Source:")
    print(args.master)
    print()
    print("Destination:")
    print(args.export)
    print()

    try:
        records, errors, total = read_master(args.master)
    except OSError as exc:
        print("Cannot read master dataset: %s" % exc)
        return 2

    if errors:
        for message in errors:
            print(message)
        print()
        print("Master records: %d" % total)
        print("Exported examples: 0")
        print()
        print("Dataset export FAILED.")
        print("Reason: the master dataset is invalid. Nothing was written.")
        return 1

    if not records:
        print("Master records: 0")
        print("Exported examples: 0")
        print()
        print("Dataset export FAILED.")
        print("Reason: the master dataset contains no records.")
        return 1

    counts = count_statuses(records)
    approved = [
        record
        for record in records
        if record["review_status"] == EXPORTABLE_REVIEW_STATUS
    ]

    print("Master records: %d" % len(records))
    print()
    for status in REPORTED_STATUSES:
        print("%s: %d" % (status.capitalize(), counts[status]))
    print()

    if not approved:
        print("Exported examples: 0")
        print()
        print("Dataset export FAILED.")
        print("Reason: no approved records are available for training.")
        return 1

    try:
        write_export(approved, args.export)
    except OSError as exc:
        print("Cannot write export: %s" % exc)
        return 2

    print("Exported examples: %d" % len(approved))
    print("Skipped examples: %d" % (len(records) - len(approved)))
    print()
    print("Dataset export PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
