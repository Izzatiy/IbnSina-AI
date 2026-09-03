#!/usr/bin/env python3
"""Scan an Ibn Sina AI training dataset for obvious personally identifiable or
sensitive data.

Usage:
    python scripts/scan_sensitive_data.py datasets/uzbek_medical_v1.jsonl

Every textual "content" field of every message is checked against a small set of
deterministic patterns: email addresses, phone-like numbers, URLs carrying
credential-looking query parameters, secret assignments, and IPv4 addresses.

This is a conservative first-pass filter, not a complete PII detector. It does
not look at names and it does not treat medical conditions as sensitive matches.

Exit codes (kept consistent with validate_dataset.py):
    0  scan passed, no findings
    1  sensitive or suspicious data found
    2  file / input / read error

Standard library only.
"""

import argparse
import json
import re
import sys

# --- Patterns ---------------------------------------------------------------

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)

# Query parameter names that suggest a URL is carrying a credential.
SENSITIVE_PARAMS = (
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "auth",
    "api_key",
    "apikey",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "pwd",
    "session_id",
    "sessionid",
)

URL_TOKEN_RE = re.compile(
    r"\bhttps?://[^\s<>\"]*[?&](" + "|".join(SENSITIVE_PARAMS) + r")=([^\s&#\"'<>]+)",
    re.IGNORECASE,
)

# Names that suggest an assignment holds a secret.
SECRET_KEYS = (
    "api[_-]?key",
    "access[_-]?token",
    "refresh[_-]?token",
    "auth[_-]?token",
    "bearer[_-]?token",
    "client[_-]?secret",
    "secret[_-]?key",
    "private[_-]?key",
    "secret",
    "password",
    "passwd",
    "token",
)

# Conservative: the value must be at least 8 characters with no whitespace, so
# ordinary prose such as "password: unutdim" is not flagged.
SECRET_RE = re.compile(
    r"\b(" + "|".join(SECRET_KEYS) + r")\b\s*[:=]\s*[\"']?([^\s\"',;]{8,})",
    re.IGNORECASE,
)

# International form: a leading + followed by 9-15 digits, optionally grouped.
PHONE_INTL_RE = re.compile(r"\+\d[\d\s().-]{7,18}\d")

# Local grouped form, e.g. "90 123 45 67" — separators are required so that
# plain numbers in medical text (120/80, 38, 39) are not matched.
PHONE_LOCAL_RE = re.compile(r"\b\d{2}[\s.-]\d{3}[\s.-]\d{2}[\s.-]\d{2}\b")

IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


# --- Helpers ----------------------------------------------------------------


def redact(value):
    """Return a value safe to print: first 4 characters, then a fixed mask."""
    if len(value) <= 4:
        return "*" * 8
    return value[:4] + "*" * 8


def is_phone_like(text):
    """A phone candidate must carry 9-15 digits once separators are removed."""
    digits = re.sub(r"\D", "", text)
    return 9 <= len(digits) <= 15


def is_ipv4(text):
    """Reject dotted numbers whose octets are out of range (e.g. 300.1.2.3)."""
    return all(part.isdigit() and int(part) <= 255 for part in text.split("."))


def scan_text(text):
    """Return findings for one content string as (start, end, label) tuples.

    Patterns are applied most-specific first; a later match overlapping an
    already-accepted span is dropped, so a token inside a URL is reported once.
    """
    candidates = []

    for match in EMAIL_RE.finditer(text):
        candidates.append((match.start(), match.end(), "EMAIL: %s" % match.group(0)))

    for match in URL_TOKEN_RE.finditer(text):
        label = "URL_TOKEN: %s parameter in %s" % (
            match.group(1),
            match.group(0).split("?")[0],
        )
        candidates.append((match.start(), match.end(), label))

    for match in SECRET_RE.finditer(text):
        label = "SECRET: %s=%s" % (match.group(1), redact(match.group(2)))
        candidates.append((match.start(), match.end(), label))

    for pattern in (PHONE_INTL_RE, PHONE_LOCAL_RE):
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if is_phone_like(value):
                candidates.append((match.start(), match.end(), "PHONE: %s" % value))

    for match in IPV4_RE.finditer(text):
        if is_ipv4(match.group(0)):
            label = "IP: %s (potentially sensitive)" % match.group(0)
            candidates.append((match.start(), match.end(), label))

    findings = []
    taken = []
    for start, end, label in candidates:
        if any(start < other_end and end > other_start for other_start, other_end in taken):
            continue
        taken.append((start, end))
        findings.append((start, end, label))

    findings.sort()
    return findings


def scan_file(path):
    """Scan every example. Returns (scanned, reports, parse_errors)."""
    scanned = 0
    reports = []
    parse_errors = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            scanned += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(
                    "Line %d: could not be scanned, invalid JSON (%s)"
                    % (line_number, exc.msg)
                )
                continue

            if not isinstance(obj, dict):
                continue
            messages = obj.get("messages")
            if not isinstance(messages, list):
                continue

            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    continue

                findings = scan_text(content)
                if findings:
                    reports.append(
                        (line_number, index, [label for _, _, label in findings])
                    )

    return scanned, reports, parse_errors


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan an Ibn Sina AI JSONL training dataset for obvious "
        "personally identifiable or sensitive data."
    )
    parser.add_argument("dataset", help="path to the .jsonl dataset file")
    args = parser.parse_args(argv)

    print("Scanning: %s" % args.dataset)
    print()

    try:
        scanned, reports, parse_errors = scan_file(args.dataset)
    except OSError as exc:
        print("Cannot read dataset: %s" % exc)
        return 2

    for message in parse_errors:
        print(message)
    if parse_errors:
        print()

    total = 0
    for line_number, index, labels in reports:
        print("Line %d, messages[%d].content:" % (line_number, index))
        for label in labels:
            print("  %s" % label)
            total += 1
        print()

    print("Examples scanned: %d" % scanned)
    print("Sensitive findings: %d" % total)
    print()

    if total:
        print("Sensitive-data scan FAILED.")
        return 1

    if parse_errors:
        print("Sensitive-data scan INCOMPLETE: some lines could not be parsed.")
        return 2

    print("Sensitive-data scan PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
