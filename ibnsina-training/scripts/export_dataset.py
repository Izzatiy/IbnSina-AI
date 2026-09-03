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

An approved record must also carry the trace of that human review:

    "review": {"reviewer": "reviewer-001",
               "reviewed_at": "2026-09-03T10:30:00Z",
               "notes": "Language and medical wording reviewed."}

reviewer must be a non-empty string, reviewed_at a valid ISO 8601 datetime, and
notes either null or a string. Records that are not approved may leave all three
null. Nothing here judges whether the review was any good — it only records that
a person did one.

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
import json
import os
import sys
import tempfile
from datetime import datetime

ALLOWED_LANGUAGES = ("uz-Latn", "uz-Cyrl", "ru", "en", "kaa")
ALLOWED_REVIEW_STATUSES = ("draft", "reviewed", "approved", "rejected")
REQUIRED_FIELDS = (
    "id",
    "language",
    "category",
    "source",
    "review_status",
    "review",
    "messages",
)

# Keys every "review" object carries; they may be null unless the record is approved.
REVIEW_FIELDS = ("reviewer", "reviewed_at", "notes")

# Only this status may enter a model-ready training export.
EXPORTABLE_REVIEW_STATUS = "approved"

# Order the status counts are reported in.
REPORTED_STATUSES = ("approved", "draft", "reviewed", "rejected")


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

    if "review" in record:
        errors.extend(check_review(record))

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


def parse_iso_datetime(value):
    """Parse an ISO 8601 datetime string, or return None if it is not one.

    A trailing "Z" is normalised to "+00:00" so the check behaves the same on
    Python versions older than 3.11. A bare date such as "2026-09-03" is
    rejected: reviewed_at records when a review happened, not just the day.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if "T" not in value.upper():
        return None

    return parsed


def check_review(record):
    """Return the review-metadata errors for one master record.

    Every record must carry a "review" object with the three review keys. The
    values may all be null unless the record is approved, in which case a
    reviewer and a reviewed_at timestamp are required.
    """
    errors = []
    review = record.get("review")

    if not isinstance(review, dict):
        return ['"review" must be a JSON object']

    for field in REVIEW_FIELDS:
        if field not in review:
            errors.append('"review" is missing "%s"' % field)

    approved = record.get("review_status") == EXPORTABLE_REVIEW_STATUS
    record_id = record.get("id")
    label = 'approved record "%s"' % record_id if isinstance(record_id, str) else "approved record"

    reviewer = review.get("reviewer")
    if approved:
        if reviewer is None or (isinstance(reviewer, str) and not reviewer.strip()):
            errors.append("%s is missing reviewer" % label)
        elif not isinstance(reviewer, str):
            errors.append("%s has a non-string reviewer" % label)
    elif reviewer is not None and not isinstance(reviewer, str):
        errors.append('"review.reviewer" must be a string or null')

    reviewed_at = review.get("reviewed_at")
    if approved:
        if reviewed_at is None:
            errors.append("%s is missing reviewed_at" % label)
        elif parse_iso_datetime(reviewed_at) is None:
            errors.append(
                "%s has an invalid reviewed_at %s (expected an ISO 8601 datetime, "
                "e.g. 2026-09-03T10:30:00Z)" % (label, json.dumps(reviewed_at))
            )
    elif reviewed_at is not None and parse_iso_datetime(reviewed_at) is None:
        errors.append('"review.reviewed_at" must be an ISO 8601 datetime or null')

    notes = review.get("notes")
    if notes is not None and not isinstance(notes, str):
        errors.append('"review.notes" must be a string or null')

    return errors


def count_statuses(records):
    """Count records per review status. Statuses are already known to be valid."""
    counts = dict.fromkeys(ALLOWED_REVIEW_STATUSES, 0)
    for record in records:
        counts[record["review_status"]] += 1
    return counts


def write_export(records, path):
    """Write one {"messages": [...]} object per line, atomically.

    The content goes to a temporary file in the destination directory and is
    only then moved into place, so an existing export is never truncated and a
    failed write cannot leave a partial file behind.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            for record in records:
                handle.write(
                    json.dumps({"messages": record["messages"]}, ensure_ascii=False)
                    + "\n"
                )
        os.replace(handle.name, path)
    except BaseException:
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise


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
