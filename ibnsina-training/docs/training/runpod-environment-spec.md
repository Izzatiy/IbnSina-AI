# RunPod training-environment specification

Status: **Step 11A — specification only.** Nothing has been rented, installed,
downloaded or trained. This document records decisions already made so they
survive the ephemeral development container.

## Training target

**Training runs on RunPod, on a single NVIDIA GPU. The local development
machine is not the training target.**

The local environment was inspected in Step 11 and has **no GPU at all** — no
NVIDIA kernel driver, no `/dev/nvidia*`, no `/dev/dri`. It is a 4-vCPU, 15 GiB
KVM guest. It stays what it is: the place where the dataset and its review and
approval tooling are maintained. It is not where a model is fine-tuned.

## GPU

| | |
| --- | --- |
| **Preferred** | RTX A6000 (48 GB) or L40S (48 GB). RTX 6000 Ada (48 GB) is an equivalent alternative. |
| **Minimum fallback** | RTX A5000 (24 GB) or RTX 4090 (24 GB). |
| **Count** | 1 |
| **Required VRAM** | 24 GB minimum, 48 GB preferred |

**Hard requirement: compute capability ≥ 8.0 (Ampere or newer),** for native
bf16 support. This matters more than the name on the card.

- **Avoid V100** — Volta, no native bf16.
- **Avoid T4** — Turing, 16 GB, no bf16.

Both still appear in cheap listings and both cause avoidable trouble.

**Do not use A100/H100 (80 GB) for the smoke test.** It is far more expensive
for a run that fits comfortably in 24 GB.

### Why 24 GB is enough

For QLoRA on a 7–8B model: 4-bit base weights land around 4–5 GB, adapters and
optimiser state stay under 1 GB, and the remainder is activations driven by
sequence length and batch size — roughly 8–12 GB total at sequence 2048,
batch 1–2. The 48 GB preference buys longer sequences, larger batches, and the
option of a ~32B model in 4-bit later without re-speccing the pod.

## System RAM

**≥ 32 GB, ideally 48–64 GB.**

RunPod bundles RAM with the GPU class and the 24/48 GB tiers typically ship
50–200 GB, so this usually needs no separate action. It matters because weight
download, 4-bit quantisation and tokenisation all happen in CPU RAM before
anything reaches the GPU.

## Persistent disk

**100 GB Network Volume.**

A 7–8B checkpoint in safetensors is roughly 15 GB, and the Hugging Face cache
keeps the original alongside anything derived from it. Add checkpoints, a second
model if the first disappoints, and logs. 50 GB fits exactly one small model and
nothing else; the extra is cheap insurance against a mid-run "no space left on
device".

RunPod's two storage types are **not** interchangeable:

- **Network Volume** — persists across pod stop/start. Everything that matters
  goes here.
- **Container disk** — does not persist. Anything left here is lost.

## Container / base image

**Use RunPod's official PyTorch template. Do not build a custom image for the
smoke test.**

The official images already have the NVIDIA driver plumbed through, a CUDA
runtime matched to their PyTorch build, and SSH/Jupyter working. Building a
custom image means debugging driver/CUDA/toolkit alignment before anything has
trained once.

Selection rule, to apply when versions are chosen: pick the template whose CUDA
runtime matches the host driver, with Python 3.10 or 3.11.

A CUDA **toolkit** (`nvcc`) in the image is **not** required. A pip-installed
PyTorch ships its own CUDA runtime and needs only the host driver. `nvcc` is
only necessary if something must compile CUDA kernels from source. Keep these
three layers distinct when diagnosing problems:

1. **Host NVIDIA driver** — proven by `nvidia-smi` working.
2. **CUDA runtime used by PyTorch** — reported by `torch.version.cuda`.
3. **Separately installed CUDA toolkit / `nvcc`** — usually absent, usually fine.

## Python version

**3.11 preferred, 3.10 acceptable.**

3.11 matches the local development environment (3.11.15), which keeps the
standard-library-only dataset scripts behaving identically in both places. Take
whatever the chosen image ships if it is 3.10 or 3.11 rather than fighting it.
Avoid 3.13+ for now — quantisation and adapter libraries are the usual laggards
on the newest minor version.

## Virtual environment

**Use a virtual environment, placed on persistent storage at
`/workspace/venv`.**

Putting it on the Network Volume means it survives pod stop/start and multi-GB
wheels are not reinstalled every session.

Two honest caveats:

- A venv on a network volume imports more slowly than one on container disk.
  Acceptable here; noticeable with very large stacks.
- A venv holds symlinks to a specific Python. Switching later to an image with a
  different Python **minor** version breaks the venv and it must be recreated.

Decision rule, to apply when versions are chosen: if the image's preinstalled
PyTorch is suitable, create the venv with `--system-site-packages` and inherit it
— no multi-GB re-download, CUDA already matched. If a different PyTorch is
needed, use a clean venv instead.

## Directory layout

```
/workspace/                          <- Network Volume mount (persistent)
├── IbnSina-AI/                      <- git clone of this repository
│   └── ibnsina-training/
│       ├── data/master/             <- master dataset (source of truth)
│       ├── data/exports/            <- uzbek_medical_v1.jsonl  <- TRAINING INPUT
│       ├── docs/
│       └── scripts/
├── hf-cache/                        <- HF_HOME: model + dataset cache
│   ├── hub/
│   └── datasets/
├── runs/
│   └── <run-id>/                    <- e.g. 2026-09-05-smoke-01
│       ├── checkpoints/             <- intermediate training checkpoints
│       ├── adapter/                 <- FINAL LoRA ADAPTER (the deliverable)
│       └── logs/                    <- training logs, metrics, config snapshot
└── venv/                            <- Python environment
```

Two conventions fixed now:

- **`HF_HOME=/workspace/hf-cache`** — one variable covers hub and datasets in
  current versions. `TRANSFORMERS_CACHE` is deprecated; do not use it.
- **Runs are namespaced by run-id**, so a second attempt never overwrites the
  first. Each run directory also holds a copy of its exact configuration and the
  SHA-256 of the dataset it trained on.

## What must survive pod termination

| Directory | Persistence | Why |
| --- | --- | --- |
| `runs/<id>/adapter/` | **Must** | The deliverable. Losing it loses the run. |
| `runs/<id>/checkpoints/` | **Must** | Needed to resume after interruption. |
| `runs/<id>/logs/` | **Must** | The only evidence of what happened. |
| `hf-cache/` | **Must** | Avoids re-downloading ~15 GB every session. |
| `IbnSina-AI/` | Should | Cheap to re-clone, but convenient. |
| `venv/` | Should | Avoids reinstalling the stack each session. |
| `/tmp`, container system dirs | Ephemeral | Fine to lose. |

**One rule above all: a pod can be reclaimed at any time, and a Network Volume
is not a backup.** The final adapter and logs must be pushed to git (or the
Hugging Face Hub) as soon as a run completes — never left sitting only on the
volume.

## Post-launch environment checks

Run these before installing anything. Stop if any fails.

**GPU**

1. `nvidia-smi` — exact model, total VRAM, driver version, and that VRAM is
   nearly all free (no other tenant on the card).
2. GPU count is 1, and it is the class that was paid for.
3. Compute capability ≥ 8.0, confirming bf16 is usable (once torch is present).

**CUDA layers, kept distinct**

4. Host **driver** present — `nvidia-smi` working proves it.
5. **torch CUDA runtime** — `torch.version.cuda`, `torch.cuda.is_available()`,
   `torch.cuda.get_device_name(0)`.
6. **`nvcc`** — record whether present; its absence is not a blocker.

**Platform**

7. `python --version` matches the targeted 3.10/3.11.
8. RAM via `free -h`.
9. Disk free on **both** the container disk and `/workspace` — and confirm
   `/workspace` is genuinely the mounted Network Volume and writable, not a
   container-local directory of the same name.

**Connectivity and tooling**

10. Egress reaches `huggingface.co` and `github.com`.
11. `git` **and `git-lfs`** are both present. Git LFS was absent in the local
    environment and is needed before any weights or adapters are committed.

**Data integrity**

12. After cloning, verify the exported dataset is byte-identical to the approved
    Step 10 artifact:

    ```
    sha256  30d7b74b6bb34611b8c54538b6e7748980f9ecc458e28e4ccd900ff83b6c2d27
    file    ibnsina-training/data/exports/uzbek_medical_v1.jsonl
    size    12,976 bytes, 9 lines
    ```

    Then re-run `scripts/validate_dataset.py` and
    `scripts/scan_sensitive_data.py` against it inside the pod.

    **If the hash does not match, the pod is not training on the approved data
    and the run must stop.**

## Smoke-test success criterion

**Success means: training completes and produces an adapter artifact that can be
loaded successfully.**

That is the whole bar for the first run. Model quality is **not** a success
criterion and must not be reported as one.

### Limitation — read this before interpreting any result

**The current 9-example dataset exists only for an end-to-end pipeline smoke
test. It is expected to overfit, and it must not be treated as producing a
usable or safe medical model.**

Nine short conversations cannot teach safe medical communication. The purpose of
this run is to prove the path works end to end — approved data → pod →
training → adapter → retrievable artifact. Any model it produces is a
throwaway pipeline artifact, not a medical assistant, and must not be evaluated,
demonstrated, or deployed as one.

## Intentionally undecided

The following are **deliberately not yet chosen** and must not be inferred from
this document:

- PyTorch version
- Transformers version
- PEFT version
- TRL version
- bitsandbytes version
- Accelerate version
- whether Unsloth is used at all, and if so which version

These are settled in a later step, after the pod exists and its driver, CUDA
runtime and Python version are known — because those facts constrain the
choices.

The **model checkpoint is no longer undecided**: the active selection is
`Qwen/Qwen2.5-7B-Instruct` (Step 11D), which supersedes the `Qwen/Qwen3.5-9B`
chosen in Step 11B. It is text-only and loads by the standard causal-LM path, so
no vision-tower or processor handling is needed. See `model-selection.md` for the
decision and its rationale.
