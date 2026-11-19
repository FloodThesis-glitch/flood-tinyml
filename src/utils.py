"""Shared helpers: config loading, seeding, and project paths.

Every other module in src/ imports from here, so settings and seeds
come from one place (config.yaml).
"""
from __future__ import annotations

import os

# Must be set BEFORE tensorflow is imported anywhere. oneDNN can change
# floating-point round-off between runs, which hurts reproducibility.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def get_project_root() -> Path:
    """Return the project root folder (the parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml into a dictionary.

    Args:
        path: Optional path to a YAML file. Defaults to <root>/config.yaml.
    """
    config_path = Path(path) if path else get_project_root() / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require(config: dict[str, Any], dotted_key: str) -> Any:
    """Read a setting such as 'task.flood_threshold' and refuse null values.

    Raises:
        KeyError: if the key does not exist in the config.
        ValueError: if the value is null, meaning a PLACEHOLDER was never
            filled in (we never guess missing dataset details).
    """
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing config key: '{dotted_key}'")
        value = value[part]
    if value is None:
        raise ValueError(
            f"Config setting '{dotted_key}' is still a PLACEHOLDER (null). "
            "Fill it in config.yaml before running this step."
        )
    return value


def get_path(config: dict[str, Any], key: str, create: bool = False) -> Path:
    """Return an absolute path for an entry under 'paths' in the config.

    Args:
        config: Loaded config dictionary.
        key: One of the names under 'paths' (e.g. 'data_raw', 'models').
        create: If True, create the folder when it does not exist.
    """
    relative = require(config, f"paths.{key}")
    full_path = get_project_root() / relative
    if create:
        full_path.mkdir(parents=True, exist_ok=True)
    return full_path


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow.

    Args:
        seed: The seed value, normally config['project']['seed'].
        deterministic: If True, ask TensorFlow for deterministic operations
            (can be slightly slower, but runs are repeatable).
    """
    random.seed(seed)
    np.random.seed(seed)
    # Only affects subprocesses started after this point, not this process.
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
    except ImportError:
        return  # TensorFlow not installed: Python and NumPy are seeded.

    # Seeds Python, NumPy and TensorFlow together.
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
