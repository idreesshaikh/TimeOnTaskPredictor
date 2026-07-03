"""
totvlm/splits.py
================
Domain-disjoint train/val/test splits (CLAUDE.md non-negotiable rule).

- Unit of assignment = REGISTRABLE DOMAIN (eTLD+1). A domain lives in exactly
  one of {train, val, test}; rows inherit their domain's split.
- 70/15/15 BY DOMAIN COUNT (row counts per split will differ — a few domains
  dominate row volume; the dataset card reports both).
- Deterministic: domains are sorted, then shuffled with `random.Random(seed)`,
  seed 42. Assignments are saved to artifacts/splits.json and REUSED — if the
  file exists, it is loaded, never recomputed, so every stage sees the same
  split even as the row set evolves.
- Rows whose registrable_domain is missing (or unseen by the saved assignment)
  get split=None and are excluded everywhere; counts are reported, not hidden.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS_PATH = Path("artifacts/splits.json")
SPLIT_ORDER = ("train", "val", "test")


def assign_domain_splits(
    domains: Iterable[str],
    seed: int = SEED,
    fractions: dict[str, float] | None = None,
) -> dict[str, str]:
    """Deterministically assign each unique domain to one split, 70/15/15 by
    domain count. Sort → seeded shuffle → contiguous slices."""
    import random

    fractions = fractions or FRACTIONS
    uniq = sorted({d for d in domains if isinstance(d, str) and d})
    random.Random(seed).shuffle(uniq)

    n = len(uniq)
    n_train = round(n * fractions["train"])
    n_val = round(n * fractions["val"])
    assignment = {}
    for i, d in enumerate(uniq):
        if i < n_train:
            assignment[d] = "train"
        elif i < n_train + n_val:
            assignment[d] = "val"
        else:
            assignment[d] = "test"
    assert_disjoint(assignment)
    return assignment


def assert_disjoint(assignment: dict[str, str]) -> None:
    """Assert every domain sits in exactly one split and splits don't overlap."""
    by_split: dict[str, set[str]] = {s: set() for s in SPLIT_ORDER}
    for domain, split in assignment.items():
        if split not in by_split:
            raise AssertionError(f"unknown split {split!r} for domain {domain!r}")
        by_split[split].add(domain)
    for a in SPLIT_ORDER:
        for b in SPLIT_ORDER:
            if a < b:
                overlap = by_split[a] & by_split[b]
                assert not overlap, f"{a}/{b} share domains: {sorted(overlap)[:5]}"
    total = sum(len(v) for v in by_split.values())
    assert total == len(assignment), "domain assigned to multiple splits"


def save_splits(
    assignment: dict[str, str],
    path: str | Path = SPLITS_PATH,
    seed: int = SEED,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(1 for v in assignment.values() if v == s)
              for s in SPLIT_ORDER}
    payload = {
        "seed": seed,
        "unit": "registrable_domain (eTLD+1)",
        "fractions": FRACTIONS,
        "n_domains": len(assignment),
        "domains_per_split": counts,
        "assignment": dict(sorted(assignment.items())),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_splits(path: str | Path = SPLITS_PATH) -> dict[str, str]:
    assignment = json.loads(Path(path).read_text())["assignment"]
    assert_disjoint(assignment)
    return assignment


def load_or_create_splits(
    domains: Iterable[str],
    path: str | Path = SPLITS_PATH,
    seed: int = SEED,
) -> dict[str, str]:
    """Load the saved assignment if present (source of truth), else create it
    from `domains`, assert disjointness, and save."""
    path = Path(path)
    if path.exists():
        return load_splits(path)
    assignment = assign_domain_splits(domains, seed=seed)
    save_splits(assignment, path, seed=seed)
    return assignment


def add_split_column(
    df: pd.DataFrame,
    assignment: dict[str, str],
    domain_col: str = "registrable_domain",
) -> pd.DataFrame:
    """Return a copy with `split` mapped from the domain column. Rows whose
    domain is missing/unassigned get split=None (excluded downstream)."""
    df = df.copy()
    df["split"] = df[domain_col].map(assignment)
    df.loc[df["split"].isna(), "split"] = None
    return df


def sample_trajectory_rows(
    df: pd.DataFrame,
    row_budgets: dict[str, int],
    seed: int = SEED,
) -> pd.Series:
    """
    Disk/time-bounded image-download sampling: boolean mask over `df` marking
    the rows whose screenshots should be fetched.

    - Sampling unit = a trajectory's rows WITHIN one split (a few trajectories
      cross domains and therefore splits; their per-split parts are sampled
      independently). Whole units are taken, so no sampled task has missing
      screens — which keeps the task-level (per-trajectory sum) evaluation
      honest.
    - Per split, trajectory ids are sorted then shuffled with
      `random.Random(seed)` and taken in order until the ROW budget is met
      (the last trajectory may overshoot slightly). PREFIX-STABLE: growing a
      budget only appends trajectories, so the image cache is always reused.
    - `df` must carry `split` and `trajectory_id`. Splits absent from
      `row_budgets` select nothing.
    """
    import random

    mask = pd.Series(False, index=df.index)
    for split, budget in row_budgets.items():
        part = df[df["split"] == split]
        if part.empty:
            continue
        sizes = part.groupby("trajectory_id").size()
        ids = sorted(sizes.index)
        random.Random(seed).shuffle(ids)
        chosen, total = [], 0
        for tid in ids:
            if total >= budget:
                break
            chosen.append(tid)
            total += int(sizes[tid])
        mask.loc[part.index[part["trajectory_id"].isin(chosen)]] = True
    return mask


# ── Split-aware target preparation (CLAUDE.md §8–§9) ─────────────────────────
# The winsor cap is a TRAIN-split statistic; computing it anywhere else leaks
# val/test into the target definition. dwell_s_raw stays untouched for
# reporting; the model target is y = log1p(winsorized dwell).

def train_winsor_cap(
    df: pd.DataFrame,
    percentile: float,
    split_col: str = "split",
    dwell_col: str = "dwell_s_raw",
) -> float:
    """Winsor cap = `percentile` quantile of raw dwell on the TRAIN split only."""
    train = df.loc[df[split_col] == "train", dwell_col].dropna()
    if train.empty:
        raise ValueError("no train rows — cannot compute winsor cap")
    return float(train.quantile(percentile))


def add_targets(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Add `dwell_s` (winsorized at `cap`, applied to ALL splits) and the model
    target `y = log1p(dwell_s)`. Keeps `dwell_s_raw` for reporting."""
    df = df.copy()
    df["dwell_s"] = df["dwell_s_raw"].clip(upper=cap)
    df["y"] = np.log1p(df["dwell_s"])
    return df


def dataset_card_stats(df: pd.DataFrame, cap: float, percentile: float) -> dict:
    """Per-split stats for the dataset card. Expects split/dwell_s_raw columns;
    uses img_resolved if present."""
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    per_split = {}
    for split in SPLIT_ORDER:
        sub = df[df["split"] == split]
        if sub.empty:
            per_split[split] = {"rows": 0}
            continue
        d = sub["dwell_s_raw"]
        per_split[split] = {
            "rows": int(len(sub)),
            "domains": int(sub["registrable_domain"].nunique()),
            "trajectories": int(sub["trajectory_id"].nunique()),
            "pct_navigation": round(100 * float(sub["is_navigation"].mean()), 2),
            "dwell_s_raw_quantiles": {
                f"p{int(q * 100)}": round(float(d.quantile(q)), 3)
                for q in quantiles
            },
            "pct_capped_by_winsor": round(
                100 * float((d > cap).mean()), 2
            ),
            **(
                {"pct_img_resolved": round(
                    100 * float(sub["img_resolved"].mean()), 2)}
                if "img_resolved" in sub else {}
            ),
        }
    n_unassigned = int(df["split"].isna().sum())
    return {
        "n_rows": int(len(df)),
        "n_rows_unassigned_split": n_unassigned,
        "winsor": {
            "percentile": percentile,
            "cap_dwell_s": round(cap, 3),
            "computed_on": "train split only",
        },
        "target": "y = log1p(dwell_s) with dwell_s = min(dwell_s_raw, cap)",
        "per_split": per_split,
    }


def write_dataset_card(stats: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = stats["winsor"]
    lines = [
        "# Dataset card — per-screen dwell rows with domain-disjoint splits",
        "",
        f"- Rows: **{stats['n_rows']}** "
        f"(unassigned split: {stats['n_rows_unassigned_split']})",
        f"- Winsor cap: **{w['cap_dwell_s']} s** "
        f"(p{int(w['percentile'] * 100)} of raw dwell, {w['computed_on']})",
        f"- Target: `{stats['target']}`",
        "",
        "| split | rows | domains | trajectories | %navigation | median dwell "
        "| p95 dwell | %capped | %img resolved |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split in SPLIT_ORDER:
        s = stats["per_split"][split]
        if s.get("rows", 0) == 0:
            lines.append(f"| {split} | 0 | – | – | – | – | – | – | – |")
            continue
        q = s["dwell_s_raw_quantiles"]
        lines.append(
            f"| {split} | {s['rows']} | {s['domains']} | {s['trajectories']} "
            f"| {s['pct_navigation']}% | {q['p50']} s | {q['p95']} s "
            f"| {s['pct_capped_by_winsor']}% "
            f"| {s.get('pct_img_resolved', 'n/a')}% |"
        )
    lines += ["", "## Full stats (JSON)", "", "```json",
              json.dumps(stats, indent=2), "```", ""]
    out_path.write_text("\n".join(lines))
