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
| `messages` | The conversation itself, in chat/messages format. |

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

#### Review status

| Value | Meaning |
| --- | --- |
| `draft` | Work in progress. Not reviewed. All current examples are `draft`. |
| `reviewed` | Someone has reviewed it, but it is **not** yet authorized for training. |
| `approved` | Explicitly authorized to enter a training export. |
| `rejected` | Must not be used for training. |

Only `approved` records are exported. `draft`, `reviewed`, and `rejected` records
are counted in the export summary but never written to a model-ready file.

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
Human review              a person reads the example
      |
      v
review_status = approved  set by hand in the master file
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

Before writing anything, the exporter checks every master record for the six
required fields, a non-empty string `id`, a known `language`, a known
`review_status`, and a non-empty `messages` list — and it rejects duplicate `id`
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

**Step 5.** Project structure, the first dataset, dataset validation,
sensitive-data scanning, the master/export split, and approval gating.

No model training, fine-tuning, inference, evaluation, or API integration is
implemented yet.
