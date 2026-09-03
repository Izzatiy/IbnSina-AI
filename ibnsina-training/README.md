# Ibn Sina AI — Training

This project holds the datasets and (later) the tooling used to **train and evaluate
Ibn Sina AI** models.

Ibn Sina AI is intended to become a multilingual medical AI assistant. This repository
covers only the training/evaluation side of that work — it is not the runtime platform.

## Structure

```
ibnsina-training/
├── datasets/
│   └── uzbek_medical_v1.jsonl
├── scripts/
│   ├── validate_dataset.py
│   └── scan_sensitive_data.py
└── README.md
```

### `datasets/`

Contains the training datasets, one file per dataset version.

### `scripts/`

Standalone helper scripts. Standard library only — no third-party dependencies.

### `datasets/uzbek_medical_v1.jsonl`

The first experimental Uzbek medical instruction dataset.

Each line is one valid JSON object using the standard chat/messages format:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Its purpose is to teach:

- Uzbek (Latin script) language behaviour
- Medical communication style
- Safety behaviour — Ibn Sina AI never claims to be a doctor and never gives
  dangerous or overly specific treatment instructions
- The general Ibn Sina AI response style

## Dataset validation

Every dataset should be validated before it is used for anything else:

```bash
python scripts/validate_dataset.py datasets/uzbek_medical_v1.jsonl
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
python scripts/scan_sensitive_data.py datasets/uzbek_medical_v1.jsonl
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

**Step 3.** Project structure, the first dataset, dataset validation, and
sensitive-data scanning.

No model training, fine-tuning, inference, evaluation, or API integration is
implemented yet.
