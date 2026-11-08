"""Load shared reward-model prompts without importing model dependencies.

Prompt text is preserved exactly, including whitespace and line breaks.
The legacy ``DEFAULT_SYSTEM_PROMPT`` name refers to the evaluation instruction,
not to the system-role message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "reward_model_template.json"
DEFAULT_PROFILE = "six_metric"


@dataclass(frozen=True)
class PromptConfig:
    """Reusable system and evaluation prompts for one pipeline role."""

    system_prompt: str
    evaluation_instruction: str
    profile: str


def load_prompt_config(
    path: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
) -> PromptConfig:
    """Read and validate a prompt JSON file.

    Defaults to ``prompts/reward_model_template.json`` and the ``six_metric``
    profile. Relative paths are resolved against the project root,
    independently of the current working directory.
    Absolute paths and user-home paths are also supported.

    The JSON must contain a ``profiles`` mapping with a matching profile.
    Invalid JSON or fields raise ``ValueError`` with the file path; file access
    errors propagate to the caller.
    """
    config_path = DEFAULT_PROMPT_PATH if path is None else Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid prompt JSON in {config_path}: {exc}") from exc

    if not isinstance(data, dict) or set(data) != {"profiles"}:
        raise ValueError(
            f"Prompt config {config_path} must contain exactly a 'profiles' object."
        )
    profiles = data["profiles"]
    if not isinstance(profiles, dict) or profile not in profiles:
        available = sorted(profiles) if isinstance(profiles, dict) else []
        raise ValueError(
            f"Prompt profile {profile!r} is missing from {config_path}; "
            f"available profiles: {available}."
        )
    selected = profiles[profile]
    required_fields = {"system_prompt", "evaluation_instruction"}
    if not isinstance(selected, dict) or set(selected) != required_fields:
        raise ValueError(
            f"Prompt profile {profile!r} in {config_path} must contain exactly "
            "'system_prompt' and 'evaluation_instruction'."
        )

    for field in sorted(required_fields):
        value = selected[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Field {field!r} in prompt profile {profile!r} of {config_path} "
                "must be a non-blank string."
            )

    return PromptConfig(
        system_prompt=selected["system_prompt"],
        evaluation_instruction=selected["evaluation_instruction"],
        profile=profile,
    )


DEFAULT_PROMPT_CONFIG = load_prompt_config()
OVERALL_INSPECTOR_PROMPT_CONFIG = load_prompt_config(profile="overall_inspector")
DEFAULT_SYSTEM_MESSAGE = DEFAULT_PROMPT_CONFIG.system_prompt
DEFAULT_EVALUATION_INSTRUCTION = DEFAULT_PROMPT_CONFIG.evaluation_instruction
OVERALL_INSPECTOR_SYSTEM_MESSAGE = OVERALL_INSPECTOR_PROMPT_CONFIG.system_prompt
OVERALL_INSPECTOR_EVALUATION_INSTRUCTION = (
    OVERALL_INSPECTOR_PROMPT_CONFIG.evaluation_instruction
)

# Compatibility with the existing model and training scripts: this is <Instruct>.
DEFAULT_SYSTEM_PROMPT = DEFAULT_EVALUATION_INSTRUCTION
