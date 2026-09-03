#!/usr/bin/env python3
"""Validate an Ibn Sina AI training dataset in JSONL chat/messages format.

Usage:
    python scripts/validate_dataset.py datasets/uzbek_medical_v1.jsonl

Every non-empty line must be one JSON object of the form:

    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

The whole file is always checked; every problem is reported with its line
number. Exit code is 0 when the dataset passes and non-zero when it fails.

Standard library only.
"""

import argparse
import json
import sys

ALLOWED_ROLES = ("system", "user", "assistant")
REQUIRED_ROLES = ("user", "assistant")


def validate_example(obj):
    """Return the list of error messages for one parsed example."""
    errors = []

    if not isinstance(obj, dict):
        return ["root value must be a JSON object"]

    if "messages" not in obj:
        return ['missing "messages" field']

    messages = obj["messages"]
    if not isinstance(messages, list):
        return ['"messages" must be an array']
    if not messages:
        return ['"messages" cannot be empty']

    seen_roles = set()

    for i, message in enumerate(messages):
        where = "messages[%d]" % i

        if not isinstance(message, dict):
            errors.append("%s must be a JSON object" % where)
            continue

        if "role" not in message:
            errors.append('%s is missing "role"' % where)
        else:
            role = message["role"]
            if not isinstance(role, str):
                errors.append("%s.role must be a string" % where)
            elif role not in ALLOWED_ROLES:
                errors.append('invalid role "%s"' % role)
            else:
                seen_roles.add(role)

        if "content" not in message:
            errors.append('%s is missing "content"' % where)
        else:
            content = message["content"]
            if not isinstance(content, str):
                errors.append("%s.content must be a string" % where)
            elif not content.strip():
                errors.append("%s.content cannot be empty" % where)

    for role in REQUIRED_ROLES:
        if role not in seen_roles:
            errors.append('missing at least one "%s" message' % role)

    return errors


def validate_file(path):
    """Validate every example in the file. Returns (checked, valid, errors)."""
    checked = 0
    valid = 0
    errors = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            checked += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append("Line %d: invalid JSON (%s)" % (line_number, exc.msg))
                continue

            line_errors = validate_example(obj)
            if line_errors:
                errors.extend(
                    "Line %d: %s" % (line_number, message) for message in line_errors
                )
            else:
                valid += 1

    return checked, valid, errors


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate an Ibn Sina AI JSONL training dataset."
    )
    parser.add_argument("dataset", help="path to the .jsonl dataset file")
    args = parser.parse_args(argv)

    print("Validating: %s" % args.dataset)
    print()

    try:
        checked, valid, errors = validate_file(args.dataset)
    except OSError as exc:
        print("Cannot read dataset: %s" % exc)
        return 2

    for message in errors:
        print(message)
    if errors:
        print()

    print("Examples checked: %d" % checked)
    print("Valid examples: %d" % valid)
    print("Invalid examples: %d" % (checked - valid))
    print()

    if checked == 0:
        print("Dataset validation FAILED: no examples found.")
        return 1

    if valid == checked:
        print("Dataset validation PASSED.")
        return 0

    print("Dataset validation FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
