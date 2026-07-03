"""
SPEC.md read-only guard: the external validation set must NEVER be
referenced from training/tuning code. Only the dedicated external-validation
entrypoints (scripts/prepare_external.py, scripts/validate_external.py) and
their config (configs/external.yaml) may mention the data/external path.

This test fails the moment anyone wires data/external into a model, feature,
split, label, training, or evaluation module — the leak SPEC.md forbids.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FORBIDDEN = "data/external"

# The ONLY files allowed to reference the external path:
ALLOWED = {
    REPO / "scripts" / "prepare_external.py",
    REPO / "scripts" / "validate_external.py",
    REPO / "configs" / "external.yaml",
}


def _scan(paths: list[Path]) -> list[Path]:
    return [
        p for p in paths
        if p not in ALLOWED and FORBIDDEN in p.read_text(errors="ignore")
    ]


def test_no_totvlm_module_touches_external():
    """src/totvlm is shared by training/tuning — zero tolerance there."""
    offenders = _scan(sorted((REPO / "src" / "totvlm").rglob("*.py")))
    assert not offenders, (
        f"src/totvlm modules reference {FORBIDDEN!r}: "
        f"{[str(p) for p in offenders]} — the external set is READ-ONLY and "
        f"must never feed training/tuning (SPEC.md)."
    )


def test_no_other_script_or_config_touches_external():
    """Entry points and configs outside the external pair stay clean too."""
    paths = sorted((REPO / "scripts").glob("*.py")) + sorted(
        (REPO / "configs").glob("*.yaml"))
    offenders = _scan(paths)
    assert not offenders, (
        f"non-external scripts/configs reference {FORBIDDEN!r}: "
        f"{[str(p) for p in offenders]}"
    )


def test_allowed_files_exist():
    """If the external entrypoints move, the allowlist must move with them."""
    for p in ALLOWED:
        assert p.exists(), f"guard allowlist points at a missing file: {p}"


def test_guard_catches_a_leak(tmp_path, monkeypatch):
    """The scanner itself works: a planted reference is detected."""
    bad = tmp_path / "sneaky.py"
    bad.write_text("df = pd.read_parquet('data/external/vsgui10k/x.parquet')")
    clean = tmp_path / "fine.py"
    clean.write_text("x = 1")
    assert _scan([bad, clean]) == [bad]


@pytest.mark.parametrize("path", ["src/totvlm/train.py",
                                  "scripts/train_baseline.py",
                                  "scripts/build_dataset.py",
                                  "scripts/evaluate.py"])
def test_training_and_tuning_entrypoints_named_clean(path):
    """Explicit spot-checks on the files where a leak would be worst."""
    assert FORBIDDEN not in (REPO / path).read_text()
