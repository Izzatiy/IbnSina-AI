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
└── README.md
```

### `datasets/`

Contains the training datasets, one file per dataset version.

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

## Privacy rule

**Private patient information must never be added directly to training datasets.**

No real patient names, phone numbers, medical records, ID numbers, addresses, or any
other personally identifiable or private health information belongs in these files.
Training data contains only generic, synthetic examples.

Private user information will be handled separately, through the Ibn Sina platform's
permission-controlled memory/data layer.

## Status

**Step 1.** Project structure and the first dataset format only.

No model training, fine-tuning, inference, evaluation, or API integration is
implemented yet.
