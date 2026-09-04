# Model checkpoint selection — first smoke-test QLoRA run

Status: **Step 11D — selection only.** Nothing has been downloaded, installed or
trained. This records a decision so it survives the ephemeral development
container.

## Selected checkpoint

```
Qwen/Qwen2.5-7B-Instruct
```

7.61B parameters, text-only, Apache-2.0, ungated.

> **Supersedes `Qwen/Qwen3.5-9B`,** which was selected in Step 11B and replaced
> in Step 11D after a text-only comparison. See
> [Superseded selection](#superseded-selection-qwenqwen35-9b) below for why.
> `Qwen/Qwen3.5-9B` is **no longer the active checkpoint** and must not be used
> for this run.

## Rationale

**The first run is a text-only pipeline smoke test.** Every reason below follows
from that, and from the fact that model quality is explicitly not what this run
measures.

- **`Qwen/Qwen2.5-7B-Instruct` is genuinely text-only.** Its `config.json` is
  flat — no `vision_config`, no `image_token_id`, no nested `text_config`.
- **It uses the standard causal-LM loading path**: `AutoTokenizer` plus
  `AutoModelForCausalLM` against `Qwen2ForCausalLM`. This is the most ordinary,
  most documented load in the ecosystem.
- **Our dataset already uses `system`/`user`/`assistant` chat structure**, which
  this checkpoint's chat template consumes directly with no restructuring.
- **Its LoRA targets are straightforward and language-only** — the usual
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`,
  with nothing that needs excluding.
- **It avoids vision-tower handling** entirely. There is no vision tower to load,
  to skip when attaching adapters, or to carry in memory.
- **It avoids Qwen3.5 multimodal and linear-attention tooling complexity.** The
  Qwen3.5 family is multimodal throughout and uses hybrid linear/full attention,
  so quantisation and adapter support for it would have to be verified rather
  than assumed.
- **It avoids Qwen3 thinking-block handling.** The Qwen3 chat template carries
  `<think>` and `enable_thinking` machinery. Our nine examples contain no
  reasoning traces, so that template can inject empty think blocks into assistant
  turns during formatting — a known sharp edge, avoidable entirely by not using
  a thinking-mode template.
- **7.61B is comfortably appropriate for the planned single 24–48 GB GPU QLoRA
  run.** Roughly 4.5 GB of 4-bit weights, and about 8–10 GB total at sequence
  2048 — well inside the 24 GB minimum recorded in the environment spec.
- **Model quality is not the objective of this run.** The older, smaller, plainer
  model is the correct engineering choice precisely because nothing about it is
  novel.

## Verified repository facts

Checked read-only against the Hugging Face API. No weights were downloaded —
only repository metadata, `config.json` and `tokenizer_config.json`.

| | |
| --- | --- |
| Repository | `Qwen/Qwen2.5-7B-Instruct` — exists, public |
| Parameters | 7,615,616,512 (BF16) |
| Gated | No |
| License | Apache-2.0 |
| Pipeline tag | `text-generation` |
| Conversational | Yes |
| Architecture | `Qwen2ForCausalLM`, `model_type: qwen2` |
| Config shape | Flat — no vision or image keys, no nested configs |
| Hidden size / layers | 3584 / 28 (28 attention heads, 4 KV heads) |
| Max positions | 32768 |
| Chat template | Present, 2,507 chars — `system`/`user`/`assistant`, no `<think>`, no vision handling |

## Do not use the Base checkpoint

**Do not use a base (non-instruct) checkpoint for this first smoke test.**

A base model has no instruction tuning and no established chat behaviour, so it
would add exactly the alignment mismatch this run is designed to avoid.

Base-vs-Instruct may be revisited later, **once there is a genuinely large and
reviewed training corpus.** With enough reviewed data, starting from a base model
becomes a defensible option rather than a handicap. That is not the situation
today.

## Superseded selection: `Qwen/Qwen3.5-9B`

**Historical record only. This checkpoint is not in use.**

Step 11B selected `Qwen/Qwen3.5-9B` as the conversational checkpoint. Read-only
metadata then showed it is a **vision-language model**: pipeline tag
`image-text-to-text`, architecture `Qwen3_5ForConditionalGeneration`, an
`image_token_id`, a nested `text_config`, a `preprocessor_config.json` and a
video preprocessor, plus hybrid `linear_attention`/`full_attention` layers and a
7,756-character chat template carrying both thinking and vision handling.

A Step 11C comparison confirmed there is **no text-only checkpoint anywhere in
the Qwen3.5 family** — all 21 official Qwen3.5 repositories, every size and every
`-Base` variant, are `image-text-to-text`. Staying in that family while going
text-only was therefore not an option.

Three concrete failure modes it introduced, none of which our text-only data
could ever benefit from:

1. the wrong loader class at load time (`AutoProcessor` and a
   conditional-generation class rather than the standard causal-LM path),
2. adapters silently attached to a vision encoder if LoRA targeted all linear
   layers,
3. unverified 4-bit kernel support for the `qwen3_5` hybrid attention
   architecture.

`Qwen/Qwen3-8B` was also considered and rejected: genuinely text-only and
otherwise suitable, but its chat template carries thinking-mode machinery, which
is one more thing to get right in a run whose entire purpose is having nothing go
wrong.

## Smoke-test success criterion

**Success means:**

> approved dataset → model load → QLoRA training → adapter save → adapter reload

That is the whole bar for the first run. Model quality is **not** a success
criterion and must not be reported as one.

### Limitation — read this before interpreting any result

**The 9-example dataset is expected to overfit, and must not be considered a
usable or safe medical model.**

Nine short conversations cannot teach safe medical communication. Any model this
run produces is a throwaway pipeline artifact — not a medical assistant — and
must not be evaluated, demonstrated or deployed as one.

## Dataset unchanged

The Step 10 dataset and export are **not** modified by this decision.

```
sha256  30d7b74b6bb34611b8c54538b6e7748980f9ecc458e28e4ccd900ff83b6c2d27
file    ibnsina-training/data/exports/uzbek_medical_v1.jsonl
size    12,976 bytes, 9 lines
```

## Software stack

The stack that serves this checkpoint was pinned in Step 11E: Python 3.11,
PyTorch 2.13.0 (CUDA 13 wheel), Transformers 5.15.1, PEFT 0.20.0, TRL 1.10.0,
bitsandbytes 0.50.1, Accelerate 1.14.0, datasets 5.0.1, and **no Unsloth**.

`Qwen2ForCausalLM` support, NF4 4-bit loading, PEFT's Qwen2 LoRA targets and
TRL's conversational SFT were each verified against those exact versions. See
`training-stack.md` for the evidence and the caveats.

See `runpod-environment-spec.md` for the environment these will run in.
