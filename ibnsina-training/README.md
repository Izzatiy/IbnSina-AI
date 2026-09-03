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
│   └── exports/
│       └── uzbek_medical_v1.jsonl
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
| `draft` | Written but not reviewed. All current examples are `draft`. |
| `reviewed` | Looked at by a reviewer. |
| `approved` | Accepted for training. |
| `rejected` | Not to be used. |

This is metadata only. **No medical review system is implemented** — nothing
currently checks or enforces these values beyond their spelling, and the exporter
does not yet filter on them.

Its purpose is to teach:

- Uzbek (Latin script) language behaviour
- Medical communication style
- Safety behaviour — Ibn Sina AI never claims to be a doctor and never gives
  dangerous or overly specific treatment instructions
- The general Ibn Sina AI response style

## Pipeline

```
Master dataset          data/master/uzbek_medical_v1.jsonl
      |
      v
export_dataset.py       strips metadata, checks ids and metadata fields
      |
      v
Model-ready dataset     data/exports/uzbek_medical_v1.jsonl
      |
      v
validate_dataset.py     checks the chat/messages structure
      |
      v
scan_sensitive_data.py  checks for obvious personal or sensitive data
```

Run it end to end:

```bash
python scripts/export_dataset.py \
  data/master/uzbek_medical_v1.jsonl \
  data/exports/uzbek_medical_v1.jsonl

python scripts/validate_dataset.py data/exports/uzbek_medical_v1.jsonl

python scripts/scan_sensitive_data.py data/exports/uzbek_medical_v1.jsonl
```

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

Before writing anything, it checks every master record for the six required
fields, a non-empty string `id`, a known `language`, a known `review_status`, and
a non-empty `messages` list — and it rejects duplicate `id` values, reporting the
line where the id was first seen. All errors in the file are reported at once, and
the export file is left untouched if any record fails.

All records are exported regardless of `review_status`. Filtering on approval is
deliberately not implemented yet.

Exit codes: `0` on success, `1` for an invalid master dataset, `2` for a read or
write error.

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

## Privacy rule

**Private patient information must never be added directly to training datasets.**

No real patient names, phone numbers, medical records, ID numbers, addresses, or any
other personally identifiable or private health information belongs in these files.
Training data contains only generic, synthetic examples.

Private user information will be handled separately, through the Ibn Sina platform's
permission-controlled memory/data layer.

## Status

**Step 4.** Project structure, the first dataset, dataset validation,
sensitive-data scanning, and the master/export split.

No model training, fine-tuning, inference, evaluation, or API integration is
implemented yet.
