# Training environment

Environment specification for the **first Ibn Sina AI smoke-test QLoRA run**.

Nothing here has been installed or executed. These files describe an environment;
they do not create one.

## Target

| | |
| --- | --- |
| Platform | RunPod, Linux, single NVIDIA GPU |
| VRAM | 24 GB minimum, 48 GB preferred |
| Compute capability | ≥ 8.0 (Ampere or newer), for native bf16 |
| Python | **3.11** |
| Checkpoint | **`Qwen/Qwen2.5-7B-Instruct`** — text-only, Apache-2.0 |
| Method | 4-bit QLoRA, standard Hugging Face stack |
| Unsloth | **Intentionally not used** |

## Pinned versions

Set in Step 11E. Do not change them here — change
[`../docs/training/training-stack.md`](../docs/training/training-stack.md)
first, with the reasoning, then mirror it into `requirements.txt` and
`verify_environment.py`.

```
torch==2.13.0
transformers==5.15.1
peft==0.20.0
trl==1.10.0
bitsandbytes==0.50.1
accelerate==1.14.0
datasets==5.0.1
```

**Expected PyTorch CUDA runtime: 13.x.** This is not a separate choice — the
default PyPI wheel for `torch==2.13.0` *is* a CUDA 13 build, pulling
`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-cusparselt-cu13` and
`nvidia-nvshmem-cu13`.

### How the CUDA build is installed

`requirements.txt` carries a plain `torch==2.13.0` specifier and **no index
directives**, because the default index already provides the CUDA 13 build.

If the pod's driver is too old for CUDA 13, install torch from the PyTorch
CUDA 12.8 index **first**, then apply the requirements file:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

pip then sees torch already satisfied and leaves it alone. The CUDA index is
deliberately **not** written into `requirements.txt`: an `--index-url` line there
would route *every* package through that index, not just torch.

## Intended workflow

Conceptual sequence for the pod. **Not executed here, and no training commands
are included yet.**

1. **Check the driver before anything else.** Run `nvidia-smi` and read the
   driver version. This determines which torch build to install.
2. **Create a clean Python 3.11 environment**, on persistent storage
   (`/workspace/venv`) so it survives pod stop/start.
3. **Install the pinned PyTorch/CUDA build correctly** — default PyPI for
   CUDA 13, or the cu128 index if the driver requires it (see above).
4. **Install the remaining pinned requirements** with
   `pip install -r requirements.txt`.
5. **Verify imports and CUDA visibility** with `verify_environment.py` (below).
   Do not continue until it exits 0 in strict mode.
6. **Only then proceed to model loading.**

Step 6 onwards is out of scope for Step 11F. No model has been downloaded and no
training code exists.

## Verifying the environment

```bash
# On the training target — CUDA required. This is the gate before training.
python training/verify_environment.py

# Off-target inspection only (laptop, CPU-only container).
python training/verify_environment.py --allow-cpu-only
```

The script is read-only. It imports the pinned packages, reports Python, torch,
the torch CUDA runtime, `torch.cuda.is_available()`, detected GPU names and
compute capabilities, and the version of every pinned package. It then checks
each installed version against the pin. It downloads nothing and changes nothing.

**CUDA policy — the distinction matters.** The script runs fine on a machine with
no GPU, so it can be inspected anywhere. But on the actual training target CUDA
availability is *required*: an environment that cannot see the GPU is not a
usable training environment. `--allow-cpu-only` downgrades that failure to a
warning for off-target inspection, and it must **not** be used as the check
before a training run. In cpu-only mode the script says so explicitly in its own
output.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Every pinned package installed at its exact pinned version, environment coherent. In strict mode this includes CUDA being usable. |
| `1` | Version mismatch, or an otherwise unusable training environment — in strict mode, CUDA unavailable. Also raised for a GPU below compute capability 8.0, or if Unsloth is found installed. |
| `2` | A required package is missing, an import failed, or system-level inspection raised. |

A `torch` version reported as `2.13.0+cu130` matches the `2.13.0` pin — the build
suffix identifies the CUDA build, not the release, and is stripped before
comparison. A `.dev` suffix is **not** accepted as a match.

## Scope and limits

**This environment exists for the 9-example pipeline smoke test.** Its success
criterion is:

> approved dataset → model load → QLoRA training → adapter save → adapter reload

**This must not be interpreted as defining a production medical-model stack.**
The versions were chosen for compatibility, predictability and easy debugging on
a tiny run — not for training quality, throughput or scale. The dataset it serves
is nine examples and is expected to overfit; nothing produced by this environment
is a usable or safe medical model.

A production stack would need its own selection pass, against a real corpus, real
throughput requirements and a real evaluation plan.

## Known caveats

Full detail in [`../docs/training/training-stack.md`](../docs/training/training-stack.md).
The two that will bite first:

1. **CUDA 13 driver requirement.** Check `nvidia-smi` before installing; fall
   back to the cu128 index if the driver is older.
2. **bitsandbytes on CUDA 13 is unverified.** It carries its own CUDA kernels and
   PyPI metadata does not state which CUDA they target. Import it and do a small
   4-bit load *before* attempting a training run — it is the component most
   likely to fail confusingly.

Also worth remembering: `bnb_4bit_quant_type` defaults to `"fp4"`, not `"nf4"`.
QLoRA wants NF4, so it must be set explicitly.

## Files

| File | Purpose |
| --- | --- |
| `requirements.txt` | The pinned stack, with the CUDA installation note |
| `verify_environment.py` | Read-only environment verification |
| `README.md` | This document |
