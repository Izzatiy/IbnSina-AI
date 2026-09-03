#!/usr/bin/env python3
"""Record one human review decision against one Ibn Sina AI master record.

Usage:
    python scripts/review_record.py \
        data/master/uzbek_medical_v1.jsonl \
        uz-med-000001 \
        --type language \
        --status passed \
        --reviewer language-reviewer-001

    python scripts/review_record.py \
        data/master/uzbek_medical_v1.jsonl \
        uz-med-000001 \
        --type medical \
        --status failed \
        --reviewer medical-reviewer-001 \
        --notes "The answer needs safer wording around fever."

This exists so that review fields are not edited by hand in the JSONL file. It
updates exactly one review section of exactly one record and nothing else.

What it deliberately does NOT do:

  * It does not approve anything. review_status is never touched, so passing
    both reviews still leaves a record at whatever status a person gave it.
    Authorising a record for training stays a separate, explicit human action.
  * It does not overwrite a review that is already passed or failed. That would
    erase the audit trail; corrections will be designed as their own workflow.
  * It does not set a review back to pending.

The whole master file is validated before and after the change, and the write is
atomic: if anything fails, the original file is left byte-for-byte unchanged.

Exit codes:
    0  review recorded
    1  invalid dataset, invalid request, record not found, duplicate id, or a
       review that is already completed
    2  file / read / write / system error

Standard library only.
"""

import argparse
import sys
from datetime import datetime, timezone

import dataset_common
from dataset_common import (
    PASSING_SECTION_STATUS,
    REVIEW_SECTIONS,
    read_master,
    write_jsonl_atomic,
)

# A human records a completed decision; reopening a review is not implemented.
SETTABLE_STATUSES = ("passed", "failed")


class ReviewArgumentParser(argparse.ArgumentParser):
    """Argument parser that exits 1 on a bad request rather than argparse's 2.

    An unknown --type or --status is an invalid review request, which this
    script reports as exit code 1; 2 is reserved for file and system errors.
    """

    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write("%s: error: %s\n" % (self.prog, message))
        raise SystemExit(1)


def utc_now_iso():
    """Current UTC time as a timezone-aware ISO 8601 string, e.g. 2026-09-03T08:30:00Z."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


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


def build_review(status, reviewer, notes, reviewed_at):
    """The review section to store. Field order matches the master format."""
    return {
        "status": status,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "notes": notes,
    }


def main(argv=None):
    parser = ReviewArgumentParser(
        description="Record a human language or medical review decision on one "
        "Ibn Sina AI master record. Records the review only — it never approves "
        "a record for training.",
    )
    parser.add_argument("master", help="path to the master .jsonl dataset")
    parser.add_argument("record_id", help="id of the record to review, e.g. uz-med-000001")
    parser.add_argument(
        "--type",
        dest="review_type",
        required=True,
        choices=list(REVIEW_SECTIONS),
        help="which review to record",
    )
    parser.add_argument(
        "--status",
        required=True,
        choices=list(SETTABLE_STATUSES),
        help='the reviewer\'s decision ("pending" cannot be set with this tool)',
    )
    parser.add_argument(
        "--reviewer",
        required=True,
        help="internal reviewer identifier, e.g. language-reviewer-001. Do not "
        "use a personal name, email address or phone number.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="review notes. Optional for a passed review, required for a failed one.",
    )
    args = parser.parse_args(argv)

    print("Review update")
    print()
    print("Dataset:")
    print(args.master)
    print()
    print("Record:")
    print(args.record_id)
    print()

    def fail(reason):
        print("Review update FAILED.")
        print("Reason: %s" % reason)
        return 1

    reviewer = args.reviewer.strip()
    if not reviewer:
        return fail("reviewer must not be empty.")

    notes = args.notes
    if notes is not None and not notes.strip():
        notes = None
    if args.status == "failed" and notes is None:
        return fail(
            "a failed review must include --notes explaining what the problem is."
        )

    # 1. Read and validate the whole master dataset before touching anything.
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

    record, lookup_error = find_record(records, args.record_id)
    if lookup_error:
        return fail(lookup_error)

    section = record["reviews"][args.review_type]
    if section["status"] != "pending":
        return fail(
            "%s review for %s is already completed (%s). Completed reviews are "
            "not overwritten by this tool."
            % (args.review_type, args.record_id, section["status"])
        )

    # 2. Apply the change in memory. Only this one review section is touched.
    reviewed_at = utc_now_iso()
    record["reviews"][args.review_type] = build_review(
        args.status, reviewer, notes, reviewed_at
    )

    # 3. Re-validate the result before it reaches disk.
    seen_ids = {}
    for index, candidate in enumerate(records, start=1):
        result_errors = dataset_common.check_record(candidate, seen_ids)
        if result_errors:
            for message in result_errors:
                print("Record %d: %s" % (index, message))
            print()
            return fail("the updated dataset would be invalid. No changes were made.")
        seen_ids[candidate["id"]] = index

    # 4. Write atomically; the original survives any failure before the replace.
    try:
        write_jsonl_atomic(records, args.master)
    except OSError as exc:
        print("Cannot write master dataset: %s" % exc)
        print("The original file was left unchanged.")
        return 2

    print("Review type: %s" % args.review_type)
    print("Decision: %s" % args.status)
    print("Reviewer: %s" % reviewer)
    print("Reviewed at: %s" % reviewed_at)
    if notes is not None:
        print("Notes recorded: yes")
    print()
    print(
        "review_status is unchanged (%s). This tool records a review; it does "
        "not approve a record for training." % record["review_status"]
    )
    print()
    print("Review update PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
