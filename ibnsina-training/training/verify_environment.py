#!/usr/bin/env python3
"""Verify the Ibn Sina AI training environment against the Step 11E pins.

Usage:
    python training/verify_environment.py                  # strict: CUDA required
    python training/verify_environment.py --allow-cpu-only # off-target inspection

Read-only. It imports the pinned packages and reports what it finds. It never
downloads anything, never touches the model cache, and never modifies the
environment.

CUDA policy — the important distinction:

  * The script PARSES and RUNS on a machine with no GPU. You can inspect a
    laptop or a CPU-only container with --allow-cpu-only and get a report.
  * On the ACTUAL RunPod training target, run it WITHOUT that flag. There, CUDA
    availability is REQUIRED: a training environment that cannot see the GPU is
    not a usable training environment, and strict mode fails accordingly.

  --allow-cpu-only is for looking around off-target. It must NOT be used as the
  gate before a training run.

Exit codes:
    0  every pinned package is installed at its exact pinned version and the
       environment is coherent (in strict mode this includes CUDA being usable)
    1  version mismatch, or an otherwise unusable training environment
       (in strict mode: CUDA not available)
    2  a required package is missing, an import failed, or system-level
       inspection raised

Standard library only, plus imports of the pinned packages themselves.
"""

import argparse
import importlib
import importlib.util
import platform
import sys

# The Step 11E pins. Keep in lockstep with training/requirements.txt and
# ../docs/training/training-stack.md.
PINS = {
    "torch": "2.13.0",
    "transformers": "5.15.1",
    "peft": "0.20.0",
    "trl": "1.10.0",
    "bitsandbytes": "0.50.1",
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
}

# Python minor series this environment targets.
EXPECTED_PYTHON = (3, 11)

# Packages that must NOT be present, per the Step 11E decision.
FORBIDDEN = ("unsloth",)

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_MISSING = 2


def normalize_version(raw):
    """Strip a local/build suffix from a version string.

    torch reports its version as e.g. "2.13.0+cu130"; the pin is "2.13.0". The
    part after "+" identifies the build, not the release, so it is dropped
    before comparison. A development suffix after "-" is dropped too.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    for separator in ("+", "-"):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text


def version_matches(installed, pinned):
    """True when an installed version equals the pin, ignoring build suffixes."""
    return normalize_version(installed) == normalize_version(pinned)


def describe_python():
    """Return (version_string, matches_expected_minor)."""
    version = platform.python_version()
    return version, sys.version_info[:2] == EXPECTED_PYTHON


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify the training environment against the Step 11E pins. "
        "Read-only; downloads nothing.",
        epilog="On the real training target, run WITHOUT --allow-cpu-only.",
    )
    parser.add_argument(
        "--allow-cpu-only",
        action="store_true",
        help="treat missing CUDA as a warning instead of a failure. For "
        "off-target inspection only — never as the gate before training.",
    )
    args = parser.parse_args(argv)

    strict_cuda = not args.allow_cpu_only

    print("Ibn Sina AI — training environment verification")
    print()
    print("Mode: %s" % ("STRICT (CUDA required)" if strict_cuda
                        else "cpu-only permitted (off-target inspection)"))
    print()

    problems = []      # -> exit 1
    blockers = []      # -> exit 2

    # --- Python ------------------------------------------------------------
    python_version, python_ok = describe_python()
    print("Python: %s (expected %d.%d.x)" % (python_version, *EXPECTED_PYTHON))
    if not python_ok:
        problems.append(
            "Python %s does not match the targeted %d.%d series"
            % (python_version, *EXPECTED_PYTHON)
        )
    print()

    # --- Pinned packages ---------------------------------------------------
    print("Pinned packages:")
    modules = {}
    for name, pinned in sorted(PINS.items()):
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            print("  %-14s MISSING           (pinned %s)" % (name, pinned))
            blockers.append("%s could not be imported: %s" % (name, exc))
            continue
        except Exception as exc:  # a package can import-fail for other reasons
            print("  %-14s IMPORT FAILED     (pinned %s)" % (name, pinned))
            blockers.append("%s raised on import: %s: %s"
                            % (name, type(exc).__name__, exc))
            continue

        modules[name] = module
        installed = getattr(module, "__version__", None)
        if installed is None:
            print("  %-14s version unknown   (pinned %s)" % (name, pinned))
            blockers.append("%s exposes no __version__ to compare" % name)
            continue

        if version_matches(installed, pinned):
            print("  %-14s %-17s OK" % (name, installed))
        else:
            print("  %-14s %-17s MISMATCH (pinned %s)" % (name, installed, pinned))
            problems.append(
                "%s is %s but the pin is %s" % (name, installed, pinned)
            )
    print()

    # --- Packages that must be absent --------------------------------------
    unexpected = []
    for name in FORBIDDEN:
        try:
            present = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # A broken or partially-installed package can raise here. Treat it
            # as present rather than silently passing the check.
            present = True
        if present:
            unexpected.append(name)
            print("Unexpected package present: %s" % name)
            problems.append(
                "%s is installed, but Step 11E decided against it" % name
            )
    if unexpected:
        print()

    # --- CUDA / GPU --------------------------------------------------------
    print("CUDA / GPU:")
    torch = modules.get("torch")
    if torch is None:
        print("  torch unavailable — cannot inspect CUDA")
        # Already recorded as a blocker above.
    else:
        try:
            cuda_runtime = torch.version.cuda
            available = torch.cuda.is_available()
            print("  torch CUDA runtime      : %s" % (cuda_runtime or "none (CPU build)"))
            print("  torch.cuda.is_available : %s" % available)

            if available:
                count = torch.cuda.device_count()
                print("  device count            : %d" % count)
                for index in range(count):
                    name = torch.cuda.get_device_name(index)
                    major, minor = torch.cuda.get_device_capability(index)
                    print("    device %d: %s (compute capability %d.%d)"
                          % (index, name, major, minor))
                    if (major, minor) < (8, 0):
                        problems.append(
                            "device %d (%s) is compute capability %d.%d; "
                            "bf16 needs 8.0 or newer" % (index, name, major, minor)
                        )
            else:
                message = ("CUDA is not available; this is not a usable "
                           "training environment")
                if strict_cuda:
                    problems.append(message)
                else:
                    print("  WARNING: %s." % message)
                    print("           Permitted only because --allow-cpu-only was passed.")
        except Exception as exc:
            print("  inspection failed: %s: %s" % (type(exc).__name__, exc))
            blockers.append("CUDA inspection raised: %s: %s"
                            % (type(exc).__name__, exc))
    print()

    # --- Verdict -----------------------------------------------------------
    if blockers:
        print("Blocking problems:")
        for item in blockers:
            print("  - %s" % item)
        print()
        print("Environment verification FAILED (missing or unimportable packages).")
        return EXIT_MISSING

    if problems:
        print("Problems:")
        for item in problems:
            print("  - %s" % item)
        print()
        print("Environment verification FAILED.")
        return EXIT_MISMATCH

    if not strict_cuda:
        print("Environment verification PASSED (cpu-only mode).")
        print("NOTE: this is NOT a valid pre-training check. Re-run without")
        print("      --allow-cpu-only on the training target.")
        return EXIT_OK

    print("Environment verification PASSED.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
