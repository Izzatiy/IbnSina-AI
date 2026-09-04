# Training software stack — pinned versions

Status: **Step 11E — version pinning only.** Nothing has been installed,
downloaded or trained, and no `requirements.txt` has been created. This records
decisions so they survive the ephemeral development container.

Target: RunPod, Linux, single NVIDIA GPU (24 GB min / 48 GB preferred), 4-bit
QLoRA, text-only `system`/`user`/`assistant` data, 9-example smoke test, on
`Qwen/Qwen2.5-7B-Instruct`.

Selection priorities, in order: **compatibility, predictability, standard
Hugging Face APIs, easy debugging, reproducibility.** Not novelty — several
newer releases were deliberately passed over.

## The pinned stack

| Component | Version |
| --- | --- |
| Python | **3.11** |
| PyTorch | **2.13.0** |
| CUDA runtime (from the wheel) | **13.x** |
| Transformers | **5.15.1** |
| PEFT | **0.20.0** |
| TRL | **1.10.0** |
| bitsandbytes | **0.50.1** |
| Accelerate | **1.14.0** |
| datasets | **5.0.1** |
| Unsloth | **DO NOT USE** |

All version and dependency facts below were read from PyPI release metadata and
from the libraries' own source at the exact pinned tags. Nothing was installed.

---

## Python 3.11

**Why:** every pinned package declares `requires_python >= 3.10`, so 3.11 is
inside all of their supported ranges. PyTorch 2.13.0 publishes
`cp311 manylinux_2_28_x86_64` wheels, so there is no source build. It also
matches the local development environment (3.11.15), which keeps the
standard-library-only dataset scripts behaving identically in both places.

**Evidence:** torch 2.13.0 ships wheels for cp310–cp314; transformers 5.15.1,
PEFT 0.20.0, Accelerate 1.14.0 and datasets 5.0.1 all declare `>=3.10.0`; TRL
1.10.0 and bitsandbytes 0.50.1 declare `>=3.10`.

**Caveat:** 3.12+ is avoided for now; quantisation and adapter libraries are the
usual laggards on the newest minor.

## PyTorch 2.13.0

**Why:** released 2026-07-08, so roughly two months settled. The newest release,
2.14.0, landed 2026-09-02 — two days before this decision — and is exactly the
kind of choice the "not merely because it is newest" rule exists to prevent.

**Evidence:** `cp311-cp311-manylinux_2_28_x86_64` wheel published. Satisfies
bitsandbytes' `torch<3,>=2.4`, PEFT's `torch>=1.13.0`, and Accelerate's
`torch>=2.0.0`.

**Caveat:** see the CUDA section — this wheel is CUDA 13, which constrains the
host driver.

## CUDA runtime 13.x

**Why:** not a free choice. The default PyPI wheel for torch 2.13.0 declares
`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-cusparselt-cu13` and
`nvidia-nvshmem-cu13` — it *is* a CUDA 13 build. torch 2.12.1 is the same
family, so dropping one minor version does not avoid this.

**This is the single largest RunPod compatibility risk in the stack.** CUDA 13
requires a recent NVIDIA driver on the host. RunPod images and hosts vary, and a
pod whose driver predates CUDA 13 will fail at `torch.cuda.is_available()`
regardless of the GPU being perfectly capable.

**Decision rule, to apply at pod start — not now:**

1. Run `nvidia-smi` and read the driver version *before* installing anything.
2. If the driver supports CUDA 13, install torch 2.13.0 from the default PyPI
   index.
3. If it does not, install a CUDA 12.x build of the same torch version from the
   PyTorch index instead. `https://download.pytorch.org/whl/cu128/` was
   confirmed reachable (HTTP 200). The rest of the stack is unaffected: nothing
   else here pins a CUDA family.

Keep the three layers distinct when diagnosing: host driver, the CUDA runtime
inside the torch wheel, and a separately installed toolkit/`nvcc` (not needed).

## Transformers 5.15.1

**Why:** the 5.x line is the current generation, and TRL 1.10.0 requires
`transformers>=4.56.2`, so 5.x is expected. 5.15.1 is a patch release
(2026-08-19) of 5.15.0 (2026-08-10), so its first-week bugs are already fixed,
while the 5.16.x line is barely a week old.

**Qwen2 support — verified, not assumed:** at tag `v5.15.1`,
`src/transformers/models/qwen2/modeling_qwen2.py` defines `Qwen2ForCausalLM`
(confirmed present; at the adjacent `v5.16.1` tag it is at line 406, alongside
`Qwen2Model`, `Qwen2Attention`, `Qwen2MLP`, `Qwen2DecoderLayer`).

**4-bit NF4 — verified:** at tag `v5.15.1`,
`src/transformers/utils/quantization_config.py` exposes `load_in_4bit` and
`bnb_4bit_quant_type`, documented as `fp4` or `nf4`.

**Caveat — a real trap:** `bnb_4bit_quant_type` **defaults to `"fp4"`, not
`"nf4"`.** QLoRA wants NF4, so it must be set explicitly. Omitting it produces a
run that works but is not the intended quantisation.

**Caveat:** transformers 5.x is a major version. Any tutorial or snippet written
against 4.x may use moved or removed APIs. Prefer the 5.x documentation.

## PEFT 0.20.0

**Why:** released 2026-07-28, about five weeks settled, and the current line.
Its dependency declarations are permissive (`torch>=1.13.0`, `transformers`
unpinned, `accelerate>=0.21.0`), so it does not fight any other pin.

**Qwen2 LoRA support — verified:** `src/peft/utils/constants.py` at tag `v0.20.0`
contains an explicit built-in mapping entry, `"qwen2": ["q_proj", "v_proj"]`,
plus a QLoRA-oriented entry `"qwen2": ["q_proj", "v_proj", "down_proj"]`. PEFT
knows this architecture by name.

## TRL 1.10.0

**Why:** released 2026-08-13, about three weeks settled. 1.11.0 and 1.12.0 both
landed on 2026-08-26 and are too fresh for a run whose priority is nothing going
wrong. 1.10.0 carries identical core dependency requirements to 1.12.0.

**Conversational SFT — verified:** at tag `v1.10.0`,
`trl/trainer/sft_trainer.py` imports `is_conversational`,
`is_conversational_from_value` and `apply_chat_template`, and its documented
dataset contract names *"either a `messages` key for conversational inputs or a
`text` field"*. Our export uses exactly a `messages` key.

**Compatibility with our pins:** TRL 1.10.0 requires `accelerate>=1.4.0`
(we pin 1.14.0 ✓), `datasets>=4.7.0` (we pin 5.0.1 ✓),
`transformers>=4.56.2` (we pin 5.15.1 ✓), plus `jinja2` and `packaging>20.0`,
both pulled automatically.

## bitsandbytes 0.50.1

**Why:** released 2026-08-13 — crucially, **after** torch 2.13.0 (2026-07-08),
so it was built and tested in a world where that torch exists. An older, more
"settled" release such as 0.49.2 (2026-02-16) predates this torch entirely,
which is the wrong kind of conservative for a CUDA-kernel library.

**Evidence:** declares `torch<3,>=2.4`, satisfied by 2.13.0. Ships a single
universal `py3-none-manylinux_2_24_x86_64` wheel rather than per-CUDA variants.

**Caveat — the highest-risk item in this stack.** bitsandbytes carries its own
CUDA kernels, and PyPI metadata does not state which CUDA versions those kernels
target. Its compatibility with a **CUDA 13** torch build could not be verified
from metadata alone. Treat it as unproven until checked in the pod. It is also
the component most likely to fail loudly and confusingly if the driver/CUDA
combination is wrong.

## Accelerate 1.14.0

**Why:** released 2026-06-11, about three months settled — the rare case where
the newest release is also the most seasoned. Satisfies TRL's `accelerate>=1.4.0`
and PEFT's `accelerate>=0.21.0` comfortably.

**Evidence:** requires only `torch>=2.0.0`, `numpy>=1.17`,
`huggingface_hub>=0.21.0`. No conflicts with any other pin.

## datasets 5.0.1

**Why:** released 2026-07-28, about five weeks settled, and the current major
line matching the transformers 5.x / TRL 1.x generation. Satisfies TRL's
`datasets>=4.7.0`.

**Caveat:** 5.0 is a major version. Our usage is trivial — loading a 9-line local
JSONL file — so the blast radius of any 4.x→5.x behaviour change is small, but a
4.x-era snippet may still need adjusting.

## Unsloth: DO NOT USE

**Decision: do not use Unsloth for this smoke test.**

Reasons, weighed against the stated priorities:

- **Its benefit does not apply here.** Unsloth's value is throughput and memory
  savings on substantial training runs. Ours is nine short examples on a 7.6B
  model — a run measured in minutes, comfortably inside 24 GB. There is nothing
  material to optimise.
- **It is another dependency layer** with its own coupling to specific torch,
  transformers and triton versions, in a stack that already spans two fresh major
  versions (transformers 5.x, TRL 1.x) and a CUDA-13 torch.
- **It patches model classes at import time.** When something breaks — and on a
  first run something usually does — patched internals make the traceback harder
  to read and the failure harder to attribute.
- **The priority list says standard Hugging Face APIs and easy debugging.**
  Plain `transformers` + `peft` + `trl` produces the error messages that the
  official documentation and every search result describe.

Revisit Unsloth when there is a real corpus and training time actually costs
something.

## LoRA target modules — likely, not final

Hyperparameters are **not** finalised here. But the module names are a fact of
the architecture, and they were read directly from
`modeling_qwen2.py` at tag `v5.16.1`:

| Module | Defined in | Line |
| --- | --- | --- |
| `q_proj` | `Qwen2Attention` | 189 |
| `k_proj` | `Qwen2Attention` | 190 |
| `v_proj` | `Qwen2Attention` | 191 |
| `o_proj` | `Qwen2Attention` | 192 |
| `gate_proj` | `Qwen2MLP` | 41 |
| `up_proj` | `Qwen2MLP` | 42 |
| `down_proj` | `Qwen2MLP` | 43 |

The likely target set is those seven — the standard QLoRA choice — or the
attention-only subset `q_proj, k_proj, v_proj, o_proj`.

**Confirmed language-only, no vision modules.** A case-insensitive search of the
entire Qwen2 modeling file for `vision`, `image`, `visual` and `patch_embed`
returned **zero matches**. There is no vision tower in this architecture, so
there is nothing for LoRA to accidentally adapt — which is precisely why
`Qwen/Qwen2.5-7B-Instruct` replaced the multimodal `Qwen/Qwen3.5-9B`.

## Verification summary

| Requirement | Status | How it was established |
| --- | --- | --- |
| `Qwen2ForCausalLM` supported by Transformers | **Verified** | Class present in `modeling_qwen2.py` at tag `v5.15.1` |
| 4-bit NF4 via bitsandbytes | **Verified in the API** | `load_in_4bit` + `bnb_4bit_quant_type` (`fp4`/`nf4`) in `quantization_config.py` at `v5.15.1` |
| PEFT LoRA on Qwen2 projections | **Verified** | Explicit `"qwen2"` entries in PEFT `constants.py` at `v0.20.0` |
| TRL conversational SFT | **Verified** | `is_conversational` / `apply_chat_template` imports and a documented `messages` contract in `sft_trainer.py` at `v1.10.0` |
| Accelerate compatible with Transformers/TRL | **Verified** | 1.14.0 satisfies TRL's `>=1.4.0` and PEFT's `>=0.21.0`; requires only `torch>=2.0.0` |
| PyTorch CUDA wheel vs RunPod | **Conditional** | Wheel is CUDA 13; host driver must support it, else use the cu128 index. Verify at pod start |
| Python supported by every package | **Verified** | All declare `>=3.10`; torch ships cp311 linux wheels |
| bitsandbytes kernels on CUDA 13 | **Unverified** | Not determinable from metadata; must be checked in the pod |

## Known caveats

1. **CUDA 13 driver requirement.** The default torch wheel is CUDA 13. Check
   `nvidia-smi` before installing; fall back to the cu128 index if the driver is
   older.
2. **bitsandbytes on CUDA 13 is unproven here.** Highest-risk component. Import
   it and do a tiny 4-bit load *before* attempting a training run.
3. **`bnb_4bit_quant_type` defaults to `fp4`.** Set `nf4` explicitly or QLoRA
   silently uses the wrong quantisation.
4. **Transformers 5.x and datasets 5.x are major versions**, and TRL is on 1.x.
   Snippets written for transformers 4.x / TRL 0.x may not apply. Use current
   docs.
5. **Versions are pinned but not yet resolved together.** No dependency
   resolution has been run, because nothing has been installed. Transitive pins
   (`tokenizers`, `huggingface-hub`) will be resolved by pip at install time —
   transformers 5.15.1 requires `tokenizers>=0.22.0,<=0.23.0` and
   `huggingface-hub>=1.5.0,<2.0`.
6. **The stack is chosen for a 7.6B text-only model.** It is not validated for
   the superseded multimodal checkpoint.

## Not yet done

- No `requirements.txt` — deliberately deferred.
- Nothing installed, no model downloaded, no pod started, no training code, no
  training run.
- LoRA hyperparameters (rank, alpha, dropout, learning rate, epochs) are not set.

See `runpod-environment-spec.md` for the environment and `model-selection.md`
for the checkpoint these versions serve.
