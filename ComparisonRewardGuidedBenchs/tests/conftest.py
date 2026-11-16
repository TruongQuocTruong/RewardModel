"""Shared pytest setup: CPU only, offline, no Slurm, scripts/ importable."""

import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

BLOCKED_COMMANDS = {"sbatch", "scontrol", "scancel", "srun"}


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
