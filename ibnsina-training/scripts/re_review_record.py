#!/usr/bin/env python3
"""Replace one completed human review on one Ibn Sina AI master record.

Usage:
    python scripts/re_review_record.py \
        data/master/uzbek_medical_v1.jsonl \
        uz-med-000001 \
        --type language \
        --status passed \
        --reviewer language-reviewer-001 \
        --reason "Previous decision used medical-quality criteria during language review."

review_record.py deliberately refuses to overwrite a completed review, because
silently replacing a decision destroys the audit trail. This command is the
narrow, explicit exception: it replaces one completed review and files the old
one in review_history rather than discarding it.

It only ever touches a review that is already "passed" or "failed". A review
still "pending" belongs to review_record.py — this tool is for correcting a
decision, not making a first one.

What it deliberately does NOT do:

  * It does not approve anything. review_status is never touched, so correcting
    a review to "passed" still leaves the record at whatever status a person
    gave it. Authorising a record for training stays a separate action.
  * It does not delete the previous review. Every replacement appends an entry
    to review_history recording the whole previous review, when it was
    replaced, by whom, and why.
  * It does not set a review back to "pending".

review_history is master-only metadata. Exports are built by picking the
conversation out of each record, so nothing here can reach a training file.

The whole master file is validated before and after the change, and the write is
atomic: if anything fails, the original file is left byte-for-byte unchanged.

Exit codes:
    0  re-review recorded
    1  invalid dataset, invalid request, record not found, duplicate id, or the
       target review is not a completed review
    2  file / read / write / system error

Standard library only.
"""

import argparse
import copy
import sys
from datetime import datetime, timezone

import dataset_common
from dataset_common import (
    COMPLETED_SECTION_STATUSES,
    REVIEW_HISTORY_FIELD,
    REVIEW_SECTIONS,
    read_master,
    write_jsonl_atomic,
)

# A re-review records a completed decision, exactly like the first review did.
SETTABLE_STATUSES = ("passed", "failed")


class ReReviewArgumentParser(argparse.ArgumentParser):
    """Argument parser that exits 1 on a bad request rather than argparse's 2.

    An unknown --type or --status is an invalid re-review request, reported as
    exit code 1; 2 is reserved for file and system errors.
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


def build_history_entry(review_type, previous, replaced_at, replaced_by, reason):
    """The audit entry preserving a review that is being replaced."""
    return {
        "type": review_type,
        "previous": copy.deepcopy(previous),
        "replaced_at": replaced_at,
        "replaced_by": replaced_by,
        "reason": reason,
    }


def main(argv=None):
    parser = ReReviewArgumentParser(
        description="Replace one completed language or medical review on an Ibn "
        "Sina AI master record, preserving the previous review in "
        "review_history. Records the review only — it never approves a record "
        "for training.",
    )
    parser.add_argument("master", help="path to the master .jsonl dataset")
    parser.add_argument(
        "record_id", help="id of the record to re-review, e.g. uz-med-000001"
    )
    parser.add_argument(
        "--type",
        dest="review_type",
        required=True,
        choices=list(REVIEW_SECTIONS),
        help="which review to replace",
    )
    parser.add_argument(
        "--status",
        required=True,
        choices=list(SETTABLE_STATUSES),
        help='the reviewer\'s new decision ("pending" cannot be set with this tool)',
    )
    parser.add_argument(
        "--reviewer",
        required=True,
        help="internal reviewer identifier, e.g. language-reviewer-001. Do not "
        "use a personal name, email address or phone number.",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="why the previous completed review is being replaced. Required, and "
        "must not be blank — it is the audit record of the correction.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="notes for the new review. Optional for a passed review, required "
        "for a failed one.",
    )
    args = parser.parse_args(argv)

    print("Review correction")
    print()
    print("Dataset:")
    print(args.master)
    print()
    print("Record:")
    print(args.record_id)
    print()

    def fail(reason):
        print("Review correction FAILED.")
        print("Reason: %s" % reason)
        return 1

    reviewer = args.reviewer.strip()
    if not reviewer:
        return fail("reviewer must not be empty.")

    reason = args.reason.strip()
    if not reason:
        return fail(
            "--reason must not be empty. It records why a completed review is "
            "being replaced."
        )

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
    previous_status = section["status"]
    if previous_status not in COMPLETED_SECTION_STATUSES:
        return fail(
            '%s review for %s is "%s", not a completed review. Use '
            "review_record.py to record a first decision; this tool only "
            "replaces a review that is already %s."
            % (
                args.review_type,
                args.record_id,
                previous_status,
                " or ".join(COMPLETED_SECTION_STATUSES),
            )
        )

    # 2. Apply the change in memory. The previous review is filed, not dropped.
    replaced_at = utc_now_iso()
    history = record.setdefault(REVIEW_HISTORY_FIELD, [])
    history.append(
        build_history_entry(
            args.review_type, section, replaced_at, reviewer, reason
        )
    )
    record["reviews"][args.review_type] = build_review(
        args.status, reviewer, notes, replaced_at
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
    print("Previous decision: %s (by %s)" % (previous_status, section["reviewer"]))
    print("New decision: %s" % args.status)
    print("Reviewer: %s" % reviewer)
    print("Reviewed at: %s" % replaced_at)
    if notes is not None:
        print("Notes recorded: yes")
    print("Reason: %s" % reason)
    print()
    print(
        "Previous review preserved in %s (now %d entr%s)."
        % (
            REVIEW_HISTORY_FIELD,
            len(history),
            "y" if len(history) == 1 else "ies",
        )
    )
    print(
        "review_status is unchanged (%s). This tool records a review; it does "
        "not approve a record for training." % record["review_status"]
    )
    print()
    print("Review correction PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
