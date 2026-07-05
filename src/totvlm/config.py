"""
totvlm/config.py
================
Tiny YAML config loader — SPEC.md rule: all tunable numbers live in
configs/*.yaml, never hard-coded. Every entrypoint takes a --config path and
reads knobs through this.

A config may set `base: <sibling.yaml>` to inherit another config and override
only what differs (deep-merged, child wins). That is how the two training
conditions stay honestly "identical except one thing": vlm_task.yaml is a thin
overlay on vlm.yaml (it only adds the task title), so shared knobs — including
the privileged-feature `lupi:` block both conditions use — cannot drift.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base` (child wins). Nested dicts are
    merged; scalars and lists are replaced wholesale."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    base = cfg.pop("base", None)
    if base is not None:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    return cfg
