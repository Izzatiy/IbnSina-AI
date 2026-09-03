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
    "reviews",
    "messages",
)

# The two independent human reviews every record carries.
REVIEW_SECTIONS = ("language", "medical")

# Keys every review section carries.
REVIEW_FIELDS = ("status", "reviewer", "reviewed_at", "notes")

# Allowed per-review outcomes.
REVIEW_SECTION_STATUSES = ("pending", "passed", "failed")

# The per-review outcome both sections need before a record may be approved.
PASSING_SECTION_STATUS = "passed"

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

    if "reviews" in record:
        errors.extend(check_reviews(record))

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
    """Parse a timezone-aware ISO 8601 datetime, or return None if it is not one.

    A trailing "Z" is normalised to "+00:00" so the check behaves the same on
    Python versions older than 3.11. A bare date such as "2026-09-03" is
    rejected: reviewed_at records when a review happened, not just the day. So
    is a naive timestamp — reviewers are in different places, and a time with no
    offset does not identify a moment.
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

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None

    return parsed


def describe_timestamp_problem(value):
    """Return a message describing why a timestamp is unusable, or None if it is fine."""
    if not isinstance(value, str):
        return "must be an ISO 8601 datetime string or null"

    if parse_iso_datetime(value) is not None:
        return None

    return (
        "%s is not a timezone-aware ISO 8601 datetime "
        "(expected e.g. 2026-09-03T10:30:00Z or 2026-09-03T13:30:00+03:00)"
        % json.dumps(value)
    )


def check_review_section(name, section, record_label):
    """Return the errors for one review section ("language" or "medical").

    A pending review may leave reviewer, reviewed_at and notes null. A passed
    review needs a reviewer and a timezone-aware ISO 8601 timestamp. A failed
    review needs those plus notes saying what the problem was — a rejection
    without a reason is not useful to anyone reading the master file later.
    """
    where = "reviews.%s" % name
    errors = []

    if not isinstance(section, dict):
        return ["%s must be a JSON object" % where]

    for field in REVIEW_FIELDS:
        if field not in section:
            errors.append('%s is missing "%s"' % (where, field))

    unexpected = sorted(set(section) - set(REVIEW_FIELDS))
    for field in unexpected:
        errors.append('%s has an unexpected field "%s"' % (where, field))

    status = section.get("status")
    if "status" in section and status not in REVIEW_SECTION_STATUSES:
        errors.append(
            '%s has an invalid status %s (allowed: %s)'
            % (where, json.dumps(status), ", ".join(REVIEW_SECTION_STATUSES))
        )
        # The remaining rules depend on the status, so stop here.
        return errors

    completed = status in ("passed", "failed")

    reviewer = section.get("reviewer")
    if reviewer is not None and not isinstance(reviewer, str):
        errors.append("%s.reviewer must be a string or null" % where)
    elif completed and (reviewer is None or not reviewer.strip()):
        errors.append(
            '%s is "%s" but has no reviewer%s' % (where, status, record_label)
        )

    reviewed_at = section.get("reviewed_at")
    if reviewed_at is not None:
        problem = describe_timestamp_problem(reviewed_at)
        if problem:
            errors.append("%s.reviewed_at %s" % (where, problem))
    elif completed:
        errors.append(
            '%s is "%s" but has no reviewed_at%s' % (where, status, record_label)
        )

    notes = section.get("notes")
    if notes is not None and not isinstance(notes, str):
        errors.append("%s.notes must be a string or null" % where)
    elif status == "failed" and (notes is None or not notes.strip()):
        errors.append(
            '%s is "failed" but has no notes explaining why%s' % (where, record_label)
        )

    return errors


def check_reviews(record):
    """Return the review errors for one master record.

    Every record carries both review sections. A record may only be approved
    when both of them passed; nothing here ever changes review_status, and a
    record whose reviews both passed stays unapproved until a person says so.
    """
    reviews = record.get("reviews")
    if not isinstance(reviews, dict):
        return ['"reviews" must be a JSON object']

    record_id = record.get("id")
    record_label = ' in record "%s"' % record_id if isinstance(record_id, str) else ""

    errors = []

    for name in sorted(set(reviews) - set(REVIEW_SECTIONS)):
        errors.append('"reviews" has an unexpected section "%s"' % name)

    for name in REVIEW_SECTIONS:
        if name not in reviews:
            errors.append('"reviews" is missing the "%s" review' % name)
        else:
            errors.extend(check_review_section(name, reviews[name], record_label))

    if record.get("review_status") != EXPORTABLE_REVIEW_STATUS:
        return errors

    # Approval gate: both reviews must have passed.
    label = (
        'approved record "%s"' % record_id
        if isinstance(record_id, str)
        else "approved record"
    )
    for name in REVIEW_SECTIONS:
        section = reviews.get(name)
        if not isinstance(section, dict):
            continue
        status = section.get("status")
        if status in REVIEW_SECTION_STATUSES and status != PASSING_SECTION_STATUS:
            errors.append(
                '%s cannot be approved: the %s review is "%s" (both reviews must '
                'be "%s")' % (label, name, status, PASSING_SECTION_STATUS)
            )

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
