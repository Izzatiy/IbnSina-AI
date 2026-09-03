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

Master records are checked for their required metadata fields and for duplicate
ids before anything is written; the export file is left untouched if any record
fails. Conversation contents are checked separately by validate_dataset.py and
scan_sensitive_data.py, which should be run on the exported file afterwards.

All records are exported regardless of review_status. Approval filtering is not
implemented yet.

Exit codes:
    0  export succeeded
    1  invalid master dataset
    2  input / output / file error

Standard library only.
"""

import argparse
import json
import os
import sys

ALLOWED_LANGUAGES = ("uz-Latn", "uz-Cyrl", "ru", "en", "kaa")
ALLOWED_REVIEW_STATUSES = ("draft", "reviewed", "approved", "rejected")
REQUIRED_FIELDS = ("id", "language", "category", "source", "review_status", "messages")


def check_record(record, seen_ids):
    """Return the metadata errors for one master record.

    Only the master-specific metadata is checked here; the shape of the
    conversation itself is validate_dataset.py's job, so this stays limited to
    "messages is a non-empty list".
    """
    errors = []

    if not isinstance(record, dict):
        return ["record must be a JSON object"]

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append('missing "%s"' % field)

    record_id = record.get("id")
    if "id" in record:
        if not isinstance(record_id, str) or not record_id.strip():
            errors.append('"id" must be a non-empty string')
        elif record_id in seen_ids:
            errors.append(
                'duplicate id "%s" (first seen on line %d)'
                % (record_id, seen_ids[record_id])
            )

    if "language" in record and record["language"] not in ALLOWED_LANGUAGES:
        errors.append(
            'invalid language "%s" (allowed: %s)'
            % (record["language"], ", ".join(ALLOWED_LANGUAGES))
        )

    if "review_status" in record and record["review_status"] not in ALLOWED_REVIEW_STATUSES:
        errors.append(
            'invalid review_status "%s" (allowed: %s)'
            % (record["review_status"], ", ".join(ALLOWED_REVIEW_STATUSES))
        )

    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list):
            errors.append('"messages" must be an array')
        elif not messages:
            errors.append('"messages" cannot be empty')

    return errors


def read_master(path):
    """Read and check every master record. Returns (records, errors, total).

    Every line is checked; the whole file is reported at once rather than
    stopping at the first bad record. `total` counts non-empty lines, so it
    stays correct even when one record produces several errors.
    """
    records = []
    errors = []
    seen_ids = {}
    total = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append("Line %d: invalid JSON (%s)" % (line_number, exc.msg))
                continue

            record_errors = check_record(record, seen_ids)
            if record_errors:
                errors.extend(
                    "Line %d: %s" % (line_number, message) for message in record_errors
                )
                continue

            seen_ids[record["id"]] = line_number
            records.append(record)

    return records, errors, total


def write_export(records, path):
    """Write one {"messages": [...]} object per line."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps({"messages": record["messages"]}, ensure_ascii=False) + "\n"
            )


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
        print("Dataset export FAILED: master dataset is invalid, nothing was written.")
        return 1

    if not records:
        print("Master records: 0")
        print("Exported examples: 0")
        print()
        print("Dataset export FAILED: master dataset contains no records.")
        return 1

    try:
        write_export(records, args.export)
    except OSError as exc:
        print("Cannot write export: %s" % exc)
        return 2

    print("Master records: %d" % len(records))
    print("Exported examples: %d" % len(records))
    print()
    print("Dataset export PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
