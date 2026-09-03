"""Shared master-dataset rules for the Ibn Sina AI training scripts.

The master format and its validation live here so that the exporter and the
review CLI agree on what a valid record is, rather than each carrying its own
copy of the rules.

Standard library only.
"""

import json
import os
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
    "approval",
    "messages",
)

# The two independent human reviews every record carries.
REVIEW_SECTIONS = ("language", "medical")

# Keys every review section carries.
REVIEW_FIELDS = ("status", "reviewer", "reviewed_at", "notes")

# Allowed per-review outcomes.
REVIEW_SECTION_STATUSES = ("pending", "passed", "failed")

# Keys the "approval" object carries; they may be null unless the record is approved.
APPROVAL_FIELDS = ("approver", "approved_at", "notes")

# The record statuses approve_record.py may move a record away from.
APPROVABLE_FROM_STATUSES = ("draft", "reviewed")

# The per-review outcome both sections need before a record may be approved.
PASSING_SECTION_STATUS = "passed"

# Only this status may enter a model-ready training export.
EXPORTABLE_REVIEW_STATUS = "approved"

# Order the status counts are reported in.
REPORTED_STATUSES = ("approved", "draft", "reviewed", "rejected")

# Optional master-only audit trail of replaced completed reviews. Never exported.
REVIEW_HISTORY_FIELD = "review_history"

# Keys every review_history entry carries.
REVIEW_HISTORY_ENTRY_FIELDS = ("type", "previous", "replaced_at", "replaced_by", "reason")

# A review is "completed" once a human has passed or failed it.
COMPLETED_SECTION_STATUSES = ("passed", "failed")


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

    if "approval" in record:
        errors.extend(check_approval(record))

    if REVIEW_HISTORY_FIELD in record:
        errors.extend(check_review_history(record))

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


def check_approval(record):
    """Return the approval-metadata errors for one master record.

    A record that is not approved may leave every approval field null. An
    approved one must name who authorised it and when, because approval is the
    step that lets a record reach a training export.
    """
    approval = record.get("approval")
    if not isinstance(approval, dict):
        return ['"approval" must be a JSON object']

    errors = []

    for field in APPROVAL_FIELDS:
        if field not in approval:
            errors.append('"approval" is missing "%s"' % field)

    for field in sorted(set(approval) - set(APPROVAL_FIELDS)):
        errors.append('"approval" has an unexpected field "%s"' % field)

    approved = record.get("review_status") == EXPORTABLE_REVIEW_STATUS
    record_id = record.get("id")
    label = (
        'approved record "%s"' % record_id
        if isinstance(record_id, str)
        else "approved record"
    )

    approver = approval.get("approver")
    if approver is not None and not isinstance(approver, str):
        errors.append('"approval.approver" must be a string or null')
    elif approved and (approver is None or not approver.strip()):
        errors.append("%s has no approval.approver" % label)

    approved_at = approval.get("approved_at")
    if approved_at is not None:
        problem = describe_timestamp_problem(approved_at)
        if problem:
            errors.append("approval.approved_at %s" % problem)
    elif approved:
        errors.append("%s has no approval.approved_at" % label)

    notes = approval.get("notes")
    if notes is not None and not isinstance(notes, str):
        errors.append('"approval.notes" must be a string or null')

    return errors


def check_review_history(record):
    """Return the review_history errors for one master record.

    review_history is optional. When present it must be a list of append-only
    audit entries, each recording a completed review that a re-review replaced.
    An entry keeps the whole previous review verbatim, plus when it was
    replaced, by whom, and the reason. Nothing here is ever exported.
    """
    history = record.get(REVIEW_HISTORY_FIELD)
    if not isinstance(history, list):
        return ['"%s" must be an array' % REVIEW_HISTORY_FIELD]

    errors = []

    for index, entry in enumerate(history):
        where = "%s[%d]" % (REVIEW_HISTORY_FIELD, index)

        if not isinstance(entry, dict):
            errors.append("%s must be a JSON object" % where)
            continue

        for field in REVIEW_HISTORY_ENTRY_FIELDS:
            if field not in entry:
                errors.append('%s is missing "%s"' % (where, field))

        for field in sorted(set(entry) - set(REVIEW_HISTORY_ENTRY_FIELDS)):
            errors.append('%s has an unexpected field "%s"' % (where, field))

        review_type = entry.get("type")
        if "type" in entry and review_type not in REVIEW_SECTIONS:
            errors.append(
                "%s.type %s is not one of %s"
                % (where, json.dumps(review_type), ", ".join(REVIEW_SECTIONS))
            )

        if "previous" in entry:
            previous = entry["previous"]
            # Reuse the review-section shape rules for the archived review.
            section_errors = check_review_section("<previous>", previous, "")
            errors.extend(
                "%s.previous%s" % (where, message.split("<previous>", 1)[-1])
                for message in section_errors
            )
            if isinstance(previous, dict):
                status = previous.get("status")
                if status not in COMPLETED_SECTION_STATUSES:
                    errors.append(
                        '%s.previous.status must be a completed review '
                        "(%s), not %s"
                        % (
                            where,
                            " or ".join(COMPLETED_SECTION_STATUSES),
                            json.dumps(status),
                        )
                    )

        replaced_at = entry.get("replaced_at")
        if "replaced_at" in entry:
            problem = describe_timestamp_problem(replaced_at)
            if problem:
                errors.append("%s.replaced_at %s" % (where, problem))

        replaced_by = entry.get("replaced_by")
        if "replaced_by" in entry and (
            not isinstance(replaced_by, str) or not replaced_by.strip()
        ):
            errors.append("%s.replaced_by must be a non-empty string" % where)

        reason = entry.get("reason")
        if "reason" in entry and (not isinstance(reason, str) or not reason.strip()):
            errors.append("%s.reason must be a non-empty string" % where)

    return errors


def write_jsonl_atomic(objects, path):
    """Write one JSON object per line, atomically.

    The content goes to a temporary file in the destination directory, is
    flushed and fsynced, and only then moved into place. An existing file at
    `path` is therefore never truncated, and a failure at any point before the
    replace leaves it byte-for-byte unchanged.
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
            for obj in objects:
                handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise
