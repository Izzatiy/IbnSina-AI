# Model checkpoint selection — first smoke-test QLoRA run

Status: **Step 11B — selection only.** Nothing has been downloaded, installed or
trained. This records a decision so it survives the ephemeral development
container.

## Selected checkpoint

```
Qwen/Qwen3.5-9B
```

## Rationale

- It is the **conversational / instruction-oriented** checkpoint.
- Our approved training export is already structured as `system`, `user` and
  `assistant` messages, so it maps onto a chat-tuned model directly.
- For a first end-to-end smoke test we want **minimal chat-template and
  alignment mismatch**. Starting from a base model would mean teaching
  conversational structure from nine examples, which is the wrong problem to be
  solving at this stage.
- The 9-example dataset is **not** intended to create a medically useful model.
- The purpose of the run is only to prove:

  > approved export → model load → QLoRA training → adapter artifact → adapter reload

- **Success is a loadable adapter artifact, not output quality.**

## Do not use the Base checkpoint

**Do not use `Qwen/Qwen3.5-9B-Base` for this first smoke test.**

The base model has no instruction tuning and no established chat behaviour, so
it would add exactly the alignment mismatch this run is designed to avoid.

Base-vs-Instruct may be revisited later, **once there is a genuinely large and
reviewed training corpus.** With enough reviewed data, starting from Base becomes
a defensible option rather than a handicap. That is not the situation today.

## Verified repository facts

Checked read-only against the Hugging Face API on 2026-09-04. No weights were
downloaded — only repository metadata and `config.json`.

| | |
| --- | --- |
| Repository | `Qwen/Qwen3.5-9B` — exists, public |
| Gated | No (no access request or token gate for download) |
| License | Apache-2.0 |
| Conversational | Yes (`conversational` tag) — confirms the instruct-oriented rationale |
| Base model | `Qwen/Qwen3.5-9B-Base`, i.e. this checkpoint is a finetune of it — confirms the Base-vs-Instruct relationship |
| Architecture | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` |
| Precision | bfloat16 |
| Last modified | 2026-03-02 |

The existence check was validated against a control: deliberately fake model IDs
return HTTP 401 while this one returns 200, so the result is meaningful rather
than an artifact of the network proxy.

## Important: this is a multimodal checkpoint

**`Qwen/Qwen3.5-9B` is a vision-language model, not a text-only one.** Its
pipeline tag is `image-text-to-text`, its config carries an `image_token_id` and
a nested `text_config`, and its architecture class is
`Qwen3_5ForConditionalGeneration`.

This does **not** invalidate the selection. Our data is text-only, and
fine-tuning the language side of a VLM on text-only conversations is normal. But
it has consequences that the next steps must account for:

1. **Loading path differs.** It is likely to need `AutoProcessor` and the
   conditional-generation class rather than a plain `AutoTokenizer` plus
   `AutoModelForCausalLM`. Assuming the text-only path will fail at load time.
2. **LoRA target modules need care.** Adapters should target the language
   model's projections, **not** the vision tower. Naively targeting "all linear
   layers" would adapt a vision encoder that our text-only data can never train
   meaningfully.
3. **Footprint is larger than a text-only 9B**, because the vision tower sits on
   top of the language model. This does not break the 24 GB minimum for a 4-bit
   QLoRA run, but it does reinforce the 48 GB preference recorded in the
   environment spec.
4. **The architecture is new and hybrid.** `config.json` shows alternating
   `linear_attention` and `full_attention` layers. Support for it across
   quantisation and adapter tooling should be **verified**, not assumed, when
   versions are pinned.
5. **The chat template should be checked against our 3-message structure**
   (`system`, `user`, `assistant`) before training, since the template is
   multimodal-aware.

None of these are decided here. They are flagged so the version-pinning and
training-code steps address them deliberately instead of discovering them at
runtime.

## Smoke-test success criterion, restated

**Success means: training completes and produces an adapter artifact that can be
loaded successfully.**

Model quality is **not** a success criterion and must not be reported as one.
The 9-example dataset is expected to overfit and must not be treated as
producing a usable or safe medical model. Any model this run produces is a
throwaway pipeline artifact — not a medical assistant — and must not be
evaluated, demonstrated or deployed as one.

## Dataset unchanged

The Step 10 dataset and export are **not** modified by this decision.

```
sha256  30d7b74b6bb34611b8c54538b6e7748980f9ecc458e28e4ccd900ff83b6c2d27
file    ibnsina-training/data/exports/uzbek_medical_v1.jsonl
size    12,976 bytes, 9 lines
```

## Still intentionally undecided

Selecting the checkpoint does not settle anything else. These remain open:

- PyTorch version
- Transformers version
- PEFT version
- TRL version
- bitsandbytes version
- Accelerate version
- whether Unsloth is used at all, and if so which version

They are settled after the pod exists and its driver, CUDA runtime and Python
version are known, since those facts constrain the choices — and now also after
confirming which versions actually support the `qwen3_5` architecture.

See `runpod-environment-spec.md` for the environment these will run in.
