"""Compatibility helpers for loading checkpoints with current experiment configs."""

from __future__ import annotations

from typing import Any


def resolve_checkpoint_configs(
    checkpoint: dict[str, Any],
    submitted_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the saved model config and the current runtime data config.

    Model architecture settings must match the checkpoint state dictionary, so
    they come from the checkpoint when available. Data locations are runtime
    settings and therefore come from the config supplied on the command line.
    This keeps older checkpoints usable after datasets or crop caches move.
    """
    saved_config = checkpoint.get("config", submitted_config)
    if not isinstance(saved_config, dict) or "model" not in saved_config:
        raise ValueError("Checkpoint does not contain a valid model configuration.")
    if "data" not in submitted_config:
        raise ValueError("Submitted configuration does not contain data settings.")
    return saved_config["model"], submitted_config["data"]
