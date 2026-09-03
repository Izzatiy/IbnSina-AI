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
│   ├── dataset_common.py
│   ├── validate_dataset.py
│   ├── scan_sensitive_data.py
│   ├── export_dataset.py
│   ├── review_record.py
│   ├── re_review_record.py
│   └── approve_record.py
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

Helper scripts. Standard library only — no third-party dependencies.
`dataset_common.py` holds the master-record rules the other scripts share, so the
exporter and the review CLI agree on what a valid record is.

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
  "approval": {"approver": null, "approved_at": null, "notes": null},
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
| `approval` | Who authorised the record for training, and when (see below). |
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

#### Approval metadata

Passing both reviews is not the same as authorising a record for training. That
authorisation is recorded separately:

```json
"approval": {
  "approver": "training-approver-001",
  "approved_at": "2026-09-03T09:45:00Z",
  "notes": "Both language and medical reviews verified."
}
```

| Field | Meaning |
| --- | --- |
| `approver` | Internal identifier of the person who authorised the record. |
| `approved_at` | When the approval happened, as a timezone-aware ISO 8601 datetime. |
| `notes` | Optional approval notes. May be `null`. |

A record that is not approved may leave all three `null` — that is how the three
current examples look. An `approved` record must have a non-blank `approver` and a
valid timezone-aware `approved_at`, under the same timestamp rules as
`reviewed_at`: a bare date and a naive timestamp are both rejected.

`approver` follows the same privacy rule as `reviewer`: an internal, stable
identifier such as `training-approver-001`, `medical-lead-001` or
`dataset-owner-001` — never a personal name, email address or phone number.

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
Draft master record        data/master/uzbek_medical_v1.jsonl
      |
      v
Language human review      review_record.py --type language
      |
      v
Medical human review       review_record.py --type medical
      |
      v
Both passed                reviews.language.status = reviews.medical.status = passed
      |
      v
Explicit human approval    approve_record.py --approver ...
      |
      v
review_status = approved   with approval.approver and approval.approved_at
      |
      v
export_dataset.py          exports approved records only, strips metadata
      |
      v
Model-ready JSONL          data/exports/uzbek_medical_v1.jsonl
      |
      v
validate_dataset.py        checks the chat/messages structure
      |
      v
scan_sensitive_data.py     checks for obvious personal or sensitive data
```

**No step happens automatically.** Each arrow above is a person deciding to take
it. Passing a review does not schedule an approval, and approving a record does
not run an export.

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

## Human review CLI

Review fields are not edited by hand. `review_record.py` records one review
decision against one master record:

```bash
# a language reviewer accepts the wording
python scripts/review_record.py \
  data/master/uzbek_medical_v1.jsonl \
  uz-med-000001 \
  --type language \
  --status passed \
  --reviewer language-reviewer-001

# a medical reviewer accepts the content, with a note
python scripts/review_record.py \
  data/master/uzbek_medical_v1.jsonl \
  uz-med-000001 \
  --type medical \
  --status passed \
  --reviewer medical-reviewer-001 \
  --notes "Cautious wording, refers the reader to a doctor."

# a medical reviewer rejects it — notes are required for a failed review
python scripts/review_record.py \
  data/master/uzbek_medical_v1.jsonl \
  uz-med-000002 \
  --type medical \
  --status failed \
  --reviewer medical-reviewer-001 \
  --notes "The answer needs safer wording around fever."
```

Output:

```
Review update

Dataset:
data/master/uzbek_medical_v1.jsonl

Record:
uz-med-000001

Review type: language
Decision: passed
Reviewer: language-reviewer-001
Reviewed at: 2026-09-03T08:30:00Z

Review update PASSED.
```

`--type` is `language` or `medical`; `--status` is `passed` or `failed`;
`--reviewer` is required and must not be blank. `reviewed_at` is generated by the
tool from the current UTC time — there is no `--reviewed-at` argument, so the
timestamp cannot be typed in wrong or backdated by accident. `--notes` is optional
on a pass and required on a fail.

**`review_record.py` records a human review result. It does not approve a record
for training.** `review_status` is never touched: pass both reviews and the record
still sits at whatever status a person gave it, exporting nothing until someone
deliberately sets it to `approved`. Nor does a failed review set a record to
`rejected`.

**Completed reviews cannot be overwritten with this tool.** A review that is
already `passed` or `failed` is refused:

```
Review update FAILED.
Reason: language review for uz-med-000001 is already completed (passed).
Completed reviews are not overwritten by this tool.
```

This protects the audit trail. There is no `--force`; a corrections and
re-review workflow will be designed separately. Setting a review back to
`pending` is likewise not supported.

Everything else is protected too. The whole master file is validated before the
change and again afterwards, so a broken dataset is never updated; a missing or
duplicated record id is refused rather than guessed at; and the write is atomic —
content goes to a temporary file that replaces the original only once complete, so
any failure leaves the master byte-for-byte unchanged. Only the selected review
section changes: the other review, the messages, and all other metadata are left
exactly as they were, one JSON object per line, in the original order.

Exit codes: `0` recorded, `1` invalid dataset or request (record not found,
duplicate id, already-completed review, bad argument), `2` file or system error.

## Approval CLI

`review_status` is not edited by hand either. `approve_record.py` performs the
final human authorisation:

```bash
python scripts/approve_record.py \
  data/master/uzbek_medical_v1.jsonl \
  uz-med-000001 \
  --approver training-approver-001

# with a note
python scripts/approve_record.py \
  data/master/uzbek_medical_v1.jsonl \
  uz-med-000001 \
  --approver training-approver-001 \
  --notes "Both language and medical reviews verified."
```

Output:

```
Record approval

Dataset:
data/master/uzbek_medical_v1.jsonl

Record:
uz-med-000001

Language review: passed
Medical review: passed
Approver: training-approver-001
Approved at: 2026-09-03T09:45:00Z

Review status: approved

Record approval PASSED.
```

**Reviews and approval are different things.** The reviews determine whether the
language and medical checks passed. Approval is the separate human authorisation
that lets a record enter a training export — someone taking responsibility for
putting this example into training data.

**`approve_record.py` never approves a record unless both required reviews have
already passed.** Anything else is refused:

```
Record approval FAILED.
Reason: both reviews must be "passed" before approval (medical review is "pending").
```

Two passed reviews still do not approve a record. Only a person running this
command does.

**Approved records cannot be re-approved or edited by this command.** An already
approved record is refused (`record is already approved.`), and so is a rejected
one (`rejected records cannot be approved with this command.`). Only a `draft` or
`reviewed` record may be approved. Reopening and restoring are not implemented.

`--approver` is required and must not be blank; it is an internal identifier, not
a personal name. `approved_at` is generated by the tool from the current UTC time —
there is no timestamp argument. `--notes` is optional, and whitespace-only notes
are stored as `null` rather than as a blank string.

A successful approval changes only `review_status` and the `approval` object. The
reviews, the messages, and all other metadata are left exactly as they were. As
with the review CLI, the whole master file is validated before and after the
change, a missing or duplicated id is refused rather than guessed at, and the
write is atomic, so any failure leaves the master byte-for-byte unchanged.

Exit codes: `0` approved, `1` invalid dataset or request (unmet reviews, record
not found, duplicate id, already approved, rejected, bad argument), `2` file or
system error.

## First smoke-test dataset

`uzbek_medical_v1` intentionally contains only **10 Uzbek Latin records**. That is
not an oversight and it is not a production medical dataset — it is the smallest
set that can prove the whole pipeline works before any training is attempted.

- **10 records, `uz-Latn` only.** No Russian, English, Cyrillic Uzbek or
  Karakalpak yet, and no specialised diagnosis or treatment cases. The topics are
  simple and educational: hydration during mild illness, common-cold self-care,
  headache and chest-pain warning signs, when a persistent cough needs assessment,
  why antibiotics are not for viral illness, and sleep hygiene.
- **Only genuinely human-reviewed and approved records enter the model-ready
  export.** A record needs a real language review, a real medical review, and an
  explicit approval — recorded through the CLIs — before it can be exported.
  Records still in draft are simply not exported.
- **This dataset is for pipeline and training smoke testing, not production
  clinical quality.** Ten short examples cannot teach safe medical communication;
  they can only demonstrate that master records, reviews, approval, export,
  validation and privacy scanning fit together.
- **More data will be added only after the first training pipeline works end to
  end.** Growing the dataset before that would mean reviewing material that the
  pipeline might not even be able to consume.

### Current status of this dataset

All 10 records are `draft` with both reviews `pending`. **No medical review has
been performed**, so nothing is approved and `data/exports/` is empty. The dataset
is waiting on a qualified human medical reviewer; the tooling itself has been
verified end to end against temporary synthetic copies.

## Re-review CLI

`review_record.py` refuses to overwrite a completed review, because silently
replacing a decision destroys the audit trail. `re_review_record.py` is the
narrow, explicit exception — for correcting a review that was decided on the
wrong criteria, or revisited:

```bash
python scripts/re_review_record.py \
  data/master/uzbek_medical_v1.jsonl \
  uz-med-000001 \
  --type language \
  --status passed \
  --reviewer language-reviewer-001 \
  --reason "Previous decision applied medical-completeness criteria that belong to the medical review."
```

`--reason` is required and must not be blank: it is the record of *why* a
completed decision was replaced. `--notes` stays optional on a pass and required
on a fail, and `reviewed_at` is generated from the current UTC time.

**The previous review is filed, never deleted.** Each correction appends an entry
to the record's `review_history`:

```json
"review_history": [
  {
    "type": "language",
    "previous": {
      "status": "failed",
      "reviewer": "language-reviewer-001",
      "reviewed_at": "2026-09-03T10:14:33Z",
      "notes": "..."
    },
    "replaced_at": "2026-09-03T19:38:35Z",
    "replaced_by": "language-reviewer-001",
    "reason": "Previous decision applied medical-completeness criteria..."
  }
]
```

| Field | Meaning |
| --- | --- |
| `type` | Which review was replaced: `language` or `medical`. |
| `previous` | The whole replaced review, kept verbatim. Must be a completed (`passed`/`failed`) review. |
| `replaced_at` | When the correction happened, timezone-aware ISO 8601. |
| `replaced_by` | Internal identifier of whoever made the correction. |
| `reason` | Why the previous decision was replaced. |

`review_history` is **optional** — a record that was never corrected does not
carry it — and it is **master-only metadata that never reaches a training
export**. Exports are built by picking the conversation out of each record, so a
record with any amount of review history still exports as `{"messages": [...]}`
and nothing else.

The tool only replaces a review that is already `passed` or `failed`. A `pending`
review belongs to `review_record.py`: this is for correcting a decision, not
making a first one. It cannot set a review back to `pending`, and like every
other command here it never touches `review_status` — correcting a review to
`passed` still leaves approval to a separate human action.

Same protections as the rest: the whole master file is validated before and after
the change, a missing or duplicated id is refused rather than guessed at, and the
write is atomic, so any failure leaves the master byte-for-byte unchanged.

Exit codes: `0` recorded, `1` invalid dataset or request (record not found,
duplicate id, target review not completed, blank reason, bad argument), `2` file
or system error.

### What each review is for

A **language review** judges the text as language: grammar, naturalness, clarity,
readability, terminology, and whether the meaning comes across. It should **not**
fail a record for lacking clinical guidance, thresholds, age-specific detail or
other medical completeness — those are the **medical review's** business. Mixing
the two is precisely the kind of mistake this command exists to correct.

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

**Step 10.** The pipeline is complete: master records, dataset validation,
sensitive-data scanning, the master/export split, approval gating, separate
language and medical review metadata, the review CLI, and the approval CLI.

The first smoke-test dataset has been written — 10 Uzbek Latin records, all
`draft`. **The training export is blocked pending real human medical review.**
No model training, fine-tuning, inference or API integration is implemented.

