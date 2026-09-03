# Ibn Sina AI — Training

This project holds the datasets and (later) the tooling used to **train and evaluate
Ibn Sina AI** models.

Ibn Sina AI is intended to become a multilingual medical AI assistant. This repository
covers only the training/evaluation side of that work — it is not the runtime platform.

## Structure

```
ibnsina-training/
├── data/
│   ├── master/
│   │   └── uzbek_medical_v1.jsonl
│   └── exports/          (empty — nothing is approved yet)
├── scripts/
│   ├── validate_dataset.py
│   ├── scan_sensitive_data.py
│   └── export_dataset.py
└── README.md
```

### `data/master/`

The **master datasets** — the source of truth. Each record is a conversation plus
the Ibn Sina metadata that says where it came from and whether it has been
reviewed. Master files are edited by hand; they are never fed to a trainer
directly.

### `data/exports/`

The **model-ready datasets**, produced from the master files by
`scripts/export_dataset.py`. Each line contains only `{"messages": [...]}` — the
metadata is stripped. This is what a trainer would eventually consume.

Export files are generated output: edit the master and re-export rather than
editing an export by hand.

**This directory is currently empty, and that is correct.** No master record is
approved yet, so there is nothing that may legitimately be exported. An empty
`data/exports/` means "nothing has been authorized for training" — it is not a
missing file or a broken setup, and it must not be filled in by hand or with
sample data. It stays empty until someone approves a record in the master file and
re-runs the exporter. (`.gitkeep` is there only so Git tracks the directory.)

A previous export generated before approval gating existed was deleted for this
reason: it had been produced from `draft` records and could no longer be
reproduced under the current rules.

### `scripts/`

Standalone helper scripts. Standard library only — no third-party dependencies.

### `uzbek_medical_v1`

The first experimental Uzbek medical instruction dataset.

A master line looks like this:

```json
{
  "id": "uz-med-000001",
  "language": "uz-Latn",
  "category": "general_medical",
  "source": "manual",
  "review_status": "draft",
  "reviews": {
    "language": {"status": "pending", "reviewer": null, "reviewed_at": null, "notes": null},
    "medical":  {"status": "pending", "reviewer": null, "reviewed_at": null, "notes": null}
  },
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

and the corresponding export line is just:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

#### Metadata fields

| Field | Meaning |
| --- | --- |
| `id` | Stable unique identifier, e.g. `uz-med-000001`. Must be unique within a file. |
| `language` | Language of the conversation (see below). |
| `category` | Topic bucket, e.g. `general_medical`, `symptom_guidance`, `care_seeking`. |
| `source` | Where the example came from, e.g. `manual`. |
| `review_status` | Where the example is in review (see below). |
| `reviews` | The two independent human reviews (see below). |
| `messages` | The conversation itself, in chat/messages format. |

#### Review metadata

Ibn Sina training data needs two **independent** human reviews, because "reads
well" and "is medically acceptable" are different questions answered by different
people:

```json
"reviews": {
  "language": {
    "status": "passed",
    "reviewer": "language-reviewer-001",
    "reviewed_at": "2026-09-03T10:30:00Z",
    "notes": null
  },
  "medical": {
    "status": "failed",
    "reviewer": "medical-reviewer-001",
    "reviewed_at": "2026-09-03T11:00:00Z",
    "notes": "The response gives an overly broad medical claim."
  }
}
```

**Language review** asks whether the text works as language: natural phrasing,
grammar, clarity, terminology, and whether the response sounds appropriate in the
target language.

**Medical review** asks whether the content is acceptable as medical
communication: medical correctness, unsafe statements, misleading claims, and
whether the guidance is appropriate.

**Passing a language review does not imply medical correctness.** They are
separate judgements by separate reviewers, and neither substitutes for the other.

##### Per-review status

Each review section carries its own `status`:

| Value | Meaning |
| --- | --- |
| `pending` | This review has not been completed. |
| `passed` | The reviewer accepted this aspect of the example. |
| `failed` | The reviewer found a problem. |

What each status requires:

| | `pending` | `passed` | `failed` |
| --- | --- | --- | --- |
| `reviewer` | may be null | required | required |
| `reviewed_at` | may be null | required | required |
| `notes` | may be null | optional | **required** |

A `failed` review must say why. A rejection with no reason is not useful to
whoever reads the master file later.

`reviewer` is an internal identifier (see below). `reviewed_at` must be a
**timezone-aware** ISO 8601 datetime — `2026-09-03T10:30:00Z` or
`2026-09-03T13:30:00+03:00`. A bare date (`2026-09-03`) and a naive timestamp with
no offset (`2026-09-03T10:30:00`) are both rejected: reviewers work in different
places, and a time without an offset does not identify a moment. Parsing uses the
standard library's `datetime.fromisoformat` — no date libraries are added.

##### The approval rule

```
Language review passed
          +
Medical review passed
          +
review_status set to "approved" by a person
          |
          v
Eligible for a training export
```

A record with `review_status: "approved"` is valid **only** when both
`reviews.language.status` and `reviews.medical.status` are `passed`, each with
valid reviewer metadata. Any other combination fails the export:

```
Line 1: approved record "uz-med-000001" cannot be approved: the medical review is "pending" (both reviews must be "passed")
```

**The reverse does not hold.** Two passed reviews never make a record approved by
themselves. Nothing in these scripts writes to a master file or changes a
`review_status` — they validate states, they do not make workflow decisions. This
is valid master data that simply does not export:

```
review_status = reviewed
language      = passed
medical       = passed
```

Likewise, a `rejected` record never exports, and no automatic link is imposed
between a failed review and `review_status`. A record may be rejected for a human
reason even when both reviews passed.

##### Reviewer identifiers and privacy

`reviewer` is an **internal identifier or stable reviewer label**, e.g.
`reviewer-001`, `medical-reviewer-002`, `language-reviewer-001`.

Do not put a person's full legal name, email address, phone number, or any other
personal information here. The field exists to make a review traceable, not to
identify an individual in the dataset. Mapping a label back to a person, if that
is ever needed, belongs outside the training data.

**Review metadata records that a human review occurred. It does not itself prove
that the medical content is correct.** Nothing in this repository checks the
quality of a review — only that one is recorded. And **passing a medical review
does not automatically authorize the example for training**: only records
explicitly marked `approved`, with both reviews passed, may enter a training
export.

#### Language codes

These codes are reserved for future use. Only `uz-Latn` currently appears in the
data — no translations exist yet.

| Code | Language |
| --- | --- |
| `uz-Latn` | Uzbek (Latin script) |
| `uz-Cyrl` | Uzbek (Cyrillic script) |
| `ru` | Russian |
| `en` | English |
| `kaa` | Karakalpak |

#### Record review status

This is the record-level `review_status`, distinct from the per-review `status`
values above:

| Value | Meaning |
| --- | --- |
| `draft` | Work in progress. Not reviewed. All current examples are `draft`. |
| `reviewed` | Human review activity has occurred, but the record is **not** authorized for training. |
| `approved` | Explicitly authorized to enter a training export. |
| `rejected` | Must not be used for training. Never exports, whatever its reviews say. |

Only `approved` records are exported. `draft`, `reviewed`, and `rejected` records
are counted in the export summary but never written to a model-ready file. An
`approved` record must also have both reviews `passed` with valid reviewer
metadata (see above), so a record cannot become approved without leaving a trace
of who reviewed it. None of these statuses is ever assigned automatically.

**Changing a record to `approved` is a human decision. The scripts do not determine
medical correctness.** No medical review system is implemented and none is planned
for automation — nothing in this repository reads a conversation and decides whether
it is safe. The exporter only reads the value a person wrote.

Its purpose is to teach:

- Uzbek (Latin script) language behaviour
- Medical communication style
- Safety behaviour — Ibn Sina AI never claims to be a doctor and never gives
  dangerous or overly specific treatment instructions
- The general Ibn Sina AI response style

## Pipeline

```
Master dataset            data/master/uzbek_medical_v1.jsonl
      |
      v
Language review           reviews.language.status = passed
      +
Medical review            reviews.medical.status = passed
      |                   each with its reviewer and timestamp
      v
review_status = approved  set by hand, as a separate human decision
      |
      v
export_dataset.py         exports approved records only, strips metadata
      |
      v
Model-ready dataset       data/exports/uzbek_medical_v1.jsonl
      |
      v
validate_dataset.py       checks the chat/messages structure
      |
      v
scan_sensitive_data.py    checks for obvious personal or sensitive data
```

Nothing reaches a training export without passing through the approval step, and
nothing sets that approval automatically.

Run it end to end:

```bash
python scripts/export_dataset.py \
  data/master/uzbek_medical_v1.jsonl \
  data/exports/uzbek_medical_v1.jsonl

python scripts/validate_dataset.py data/exports/uzbek_medical_v1.jsonl

python scripts/scan_sensitive_data.py data/exports/uzbek_medical_v1.jsonl
```

Right now the first command stops the pipeline: every example in
`uzbek_medical_v1` is still `draft`, so there is nothing approved to export. Approve
records in the master file first.

**A dataset that comes out the far end of this pipeline is still NOT automatically
safe or medically approved for production model training.** These three scripts
check structure and obvious pattern matches — nothing more. They do not judge
whether the medical content is correct, safe, or appropriate. Human medical review
is still required and is not implemented yet.

## Dataset export

`export_dataset.py` reads a master dataset and writes the model-ready export:

```bash
python scripts/export_dataset.py \
  data/master/uzbek_medical_v1.jsonl \
  data/exports/uzbek_medical_v1.jsonl
```

**Only records with `review_status: "approved"` are exported.** The summary reports
how many records sit at each status:

```
Master records: 5

Approved: 2
Draft: 1
Reviewed: 1
Rejected: 1

Exported examples: 2
Skipped examples: 3

Dataset export PASSED.
```

If no record is approved, the export **fails** rather than producing an empty
training file:

```
Approved: 0
Draft: 3

Exported examples: 0

Dataset export FAILED.
Reason: no approved records are available for training.
```

This is the current state of `uzbek_medical_v1` — all three examples are still
`draft`, so it does not export. That is intended: nothing has been medically
reviewed yet.

Before writing anything, the exporter checks every master record for the seven
required fields, a non-empty string `id`, a known `language`, a known
`review_status`, a well-formed `reviews` object — both the language and medical
review present and internally consistent, and both `passed` with valid reviewer
metadata on every approved record — and a non-empty `messages` list — and it rejects duplicate `id`
values, reporting the line where the id was first seen. Duplicate ids fail the
whole run even when one of the duplicates would have been skipped as unapproved,
because ids identify master records rather than exported ones.

The operation is all-or-nothing. If any record fails validation, or no record is
approved, nothing is written and an existing export file stays byte-for-byte
unchanged — it is not truncated first. A failed export never deletes or overwrites
a previously valid approved export; the last good training file survives a bad run
untouched. Equally, a failed export never creates one, so `data/exports/` simply
stays empty while nothing is approved. The write itself goes to a temporary file
that is moved into place only once complete, so an interrupted write cannot leave
a partial export behind.

Exit codes: `0` on success, `1` for an invalid master dataset or no approved
records, `2` for a read or write error.

## Dataset validation

Every dataset should be validated before it is used for anything else:

```bash
python scripts/validate_dataset.py data/exports/uzbek_medical_v1.jsonl
```

The validator checks each line of the file and reports the line number of every
problem it finds — it does not stop at the first error. For each example it checks
that:

- the line is valid JSON and the root value is an object;
- the object has a non-empty `messages` array;
- every message has a `role` and a `content`;
- `role` is one of `system`, `user`, or `assistant`;
- `content` is a non-empty string;
- the example has at least one `user` message and at least one `assistant` message.

It exits with code `0` when the dataset passes and a non-zero code when it fails,
so it can be used in a shell pipeline or a CI check.

## Sensitive-data scanning

A second, separate check looks for obvious personally identifiable or sensitive
data in the text of a dataset:

```bash
python scripts/scan_sensitive_data.py data/exports/uzbek_medical_v1.jsonl
```

It inspects every `content` field of every message and reports the line number,
the message index, and the category of each match:

- `EMAIL` — email addresses;
- `PHONE` — international (`+998 90 123 45 67`, `+998901234567`) and grouped local
  (`90 123 45 67`) phone-like numbers;
- `URL_TOKEN` — URLs carrying a credential-looking query parameter such as
  `token`, `access_token`, `api_key`, `secret`, or `password`;
- `SECRET` — assignments such as `API_KEY=...` or `password: ...`;
- `IP` — IPv4 addresses, reported as *potentially* sensitive rather than assumed
  to be private.

Secret values are never printed in full — they are redacted, e.g.
`SECRET: API_KEY=abcd********`.

Exit codes match the validator: `0` when nothing is found, `1` when something is
found, `2` on a read or input error.

**Passing this scanner does not guarantee that a dataset contains no personal or
sensitive information. It is only a first deterministic safety check.** It matches
fixed patterns only: it does not detect people's names, does not understand context,
and deliberately does not treat medical conditions as sensitive matches. Human
review remains required before any dataset is accepted.

## The master dataset is the source of truth

`review_status` lives only in `data/master/`. Exported datasets do not carry it,
so it can never be changed there.

To approve an example: edit its record in the master file, then re-run the export.
Never edit a file in `data/exports/` by hand — it is generated output and the next
export overwrites it.

## Privacy rule

**Private patient information must never be added directly to training datasets.**

No real patient names, phone numbers, medical records, ID numbers, addresses, or any
other personally identifiable or private health information belongs in these files.
Training data contains only generic, synthetic examples.

Private user information will be handled separately, through the Ibn Sina platform's
permission-controlled memory/data layer.

## Status

**Step 7.** Project structure, the first dataset, dataset validation,
sensitive-data scanning, the master/export split, approval gating, and separate
language and medical review metadata.

No model training, fine-tuning, inference, evaluation, or API integration is
implemented yet.
