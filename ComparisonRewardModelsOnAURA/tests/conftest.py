"""Shared pytest setup: CPU only, offline, no Slurm, importable scripts/."""

import os
import subprocess
import sys
from pathlib import Path

# Hide GPUs and block downloads before torch / transformers are imported by any test.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest

sys.dont_write_bytecode = True  # never leave __pycache__ inside the cloned repositories

COMPARISON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPARISON_ROOT / "scripts"))
BLOCKED_COMMANDS = {"sbatch", "scontrol", "scancel", "srun"}


@pytest.fixture
def comparison_root():
    return COMPARISON_ROOT


@pytest.fixture(autouse=True)
def block_slurm(monkeypatch):
    """Fail a test that tries to submit or change a Slurm job."""

    def guard(original):
        def wrapper(args, *a, **kw):
            argv = args.split() if isinstance(args, str) else list(args)
            if argv and Path(str(argv[0])).name in BLOCKED_COMMANDS:
                raise RuntimeError(f"Slurm command is not allowed in tests: {argv[0]}")
            return original(args, *a, **kw)

        return wrapper

    monkeypatch.setattr(subprocess, "run", guard(subprocess.run))
    monkeypatch.setattr(subprocess, "Popen", guard(subprocess.Popen))
