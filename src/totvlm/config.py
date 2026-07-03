"""
totvlm/config.py
================
Tiny YAML config loader — CLAUDE.md rule: all tunable numbers live in
configs/*.yaml, never hard-coded. Every entrypoint takes a --config path
and reads knobs through this.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return cfg
