#!/usr/bin/env python3
"""Authorise one Ibn Sina AI master record for training exports.

Usage:
    python scripts/approve_record.py \
        data/master/uzbek_medical_v1.jsonl \
        uz-med-000001 \
        --approver training-approver-001

    python scripts/approve_record.py \
        data/master/uzbek_medical_v1.jsonl \
        uz-med-000001 \
        --approver training-approver-001 \
        --notes "Both language and medical reviews verified."

This is the last human step before a record can be exported, and it exists so
that review_status is not edited by hand in the JSONL file.

Reviews and approval answer different questions. The reviews say whether the
language and medical checks passed; approval says a person takes responsibility
for putting this example into training data. This command never makes that
decision on its own:

  * Both reviews must already be "passed". Anything else is refused — this
    command does not review, and it cannot stand in for one.
  * Only a draft or reviewed record may be approved. An already approved record
    is not re-approved, and a rejected one is not revived here.
  * Nothing is ever approved implicitly. Two passed reviews do not approve a
    record; a person running this command does.

The whole master file is validated before and after the change, and the write is
atomic: if anything fails, the original file is left byte-for-byte unchanged.

Exit codes:
    0  approval recorded
    1  invalid dataset, invalid request, unmet review requirements, record not
       found, duplicate id, already approved, or rejected
    2  file / read / write / system error

Standard library only.
"""

import argparse
import sys
from datetime import datetime, timezone

import dataset_common
from dataset_common import (
    APPROVABLE_FROM_STATUSES,
    EXPORTABLE_REVIEW_STATUS,
    PASSING_SECTION_STATUS,
    REVIEW_SECTIONS,
    read_master,
    write_jsonl_atomic,
)


class ApprovalArgumentParser(argparse.ArgumentParser):
    """Argument parser that exits 1 on a bad request rather than argparse's 2.

    A malformed request is an invalid approval request, reported as exit code 1;
    2 is reserved for file and system errors.
    """

    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write("%s: error: %s\n" % (self.prog, message))
        raise SystemExit(1)


def utc_now_iso():
    """Current UTC time as a timezone-aware ISO 8601 string, e.g. 2026-09-03T09:45:00Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def find_record(records, record_id):
    """Return (record, error). Duplicate ids are refused rather than guessed at."""
    matches = [r for r in records if r.get("id") == record_id]

    if not matches:
        return None, "Record not found: %s" % record_id
    if len(matches) > 1:
        return None, (
            "Record id %s appears %d times in the master dataset"
            % (record_id, len(matches))
        )
    return matches[0], None


def check_preconditions(record):
    """Return why this record may not be approved, or None if it may."""
    status = record["review_status"]

    if status == EXPORTABLE_REVIEW_STATUS:
        return "record is already approved."
    if status not in APPROVABLE_FROM_STATUSES:
        return "%s records cannot be approved with this command." % status

    unmet = [
        name
        for name in REVIEW_SECTIONS
        if record["reviews"][name]["status"] != PASSING_SECTION_STATUS
    ]
    if unmet:
        details = ", ".join(
            '%s review is "%s"' % (name, record["reviews"][name]["status"])
            for name in unmet
        )
        return (
            "both reviews must be \"%s\" before approval (%s)."
            % (PASSING_SECTION_STATUS, details)
        )

    return None


def main(argv=None):
    parser = ApprovalArgumentParser(
        description="Authorise one Ibn Sina AI master record for training "
        "exports. Refuses to approve anything whose language and medical "
        "reviews have not both already passed.",
    )
    parser.add_argument("master", help="path to the master .jsonl dataset")
    parser.add_argument("record_id", help="id of the record to approve, e.g. uz-med-000001")
    parser.add_argument(
        "--approver",
        required=True,
        help="internal approver identifier, e.g. training-approver-001. Do not "
        "use a personal name, email address or phone number.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="optional approval notes. Whitespace-only notes are stored as null.",
    )
    args = parser.parse_args(argv)

    print("Record approval")
    print()
    print("Dataset:")
    print(args.master)
    print()
    print("Record:")
    print(args.record_id)
    print()

    def fail(reason):
        print("Record approval FAILED.")
        print("Reason: %s" % reason)
        return 1

    approver = args.approver.strip()
    if not approver:
        return fail("approver must not be empty.")

    # Whitespace-only notes carry no information, so they are stored as null
    # rather than as a blank string.
    notes = args.notes
    if notes is not None and not notes.strip():
        notes = None

    # 1-2. Read and validate the whole master dataset before touching anything.
    try:
        records, errors, _total = read_master(args.master)
    except OSError as exc:
        print("Cannot read master dataset: %s" % exc)
        return 2

    if errors:
        for message in errors:
            print(message)
        print()
        return fail("master dataset is invalid. No changes were made.")

    # 3. Find the target record.
    record, lookup_error = find_record(records, args.record_id)
    if lookup_error:
        return fail(lookup_error)

    # 4. Check the approval preconditions.
    blocked = check_preconditions(record)
    if blocked:
        return fail(blocked)

    # 5. Apply the change in memory: status and approval only.
    approved_at = utc_now_iso()
    record["review_status"] = EXPORTABLE_REVIEW_STATUS
    record["approval"] = {
        "approver": approver,
        "approved_at": approved_at,
        "notes": notes,
    }

    # 6. Re-validate the result before it reaches disk.
    seen_ids = {}
    for index, candidate in enumerate(records, start=1):
        result_errors = dataset_common.check_record(candidate, seen_ids)
        if result_errors:
            for message in result_errors:
                print("Record %d: %s" % (index, message))
            print()
            return fail("the updated dataset would be invalid. No changes were made.")
        seen_ids[candidate["id"]] = index

    # 7. Write atomically; the original survives any failure before the replace.
    try:
        write_jsonl_atomic(records, args.master)
    except OSError as exc:
        print("Cannot write master dataset: %s" % exc)
        print("The original file was left unchanged.")
        return 2

    for name in REVIEW_SECTIONS:
        print("%s review: %s" % (name.capitalize(), record["reviews"][name]["status"]))
    print("Approver: %s" % approver)
    print("Approved at: %s" % approved_at)
    if notes is not None:
        print("Notes recorded: yes")
    print()
    print("Review status: %s" % record["review_status"])
    print()
    print("Record approval PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
