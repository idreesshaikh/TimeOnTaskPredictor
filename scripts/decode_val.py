"""Decode the FULL validation split with a trained adapter — the λ-selection
metric.

    uv run python scripts/decode_val.py --config configs/sweeps/lam050.yaml

Why this exists: the per-pass val decodes inside the train cards use only
`eval.decode_samples` rows — far too few to rank λ values whose true MAE(log)
differs by ~0.01. This script batch-decodes the WHOLE val split once with the
deployed best checkpoint (`<output_dir>/final`) and writes
`<output_dir>/val_full_metrics.json`, which scripts/select_lambda.py prefers
over the card decodes. Selection stays a pure VAL decision — the TEST split is
never read here.

Conventions match scripts/evaluate.py: lenient tiered parsing, failures imputed
with the TRAIN median winsorized dwell (never dropped), predictions clipped to
the training-target range. Needs a CUDA GPU + `uv sync --extra vlm`.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from totvlm.config import load_config
from totvlm.data import (
    PARSE_TIERS,
    build_vlm_examples,
    parse_dwell_output_lenient,
)
from totvlm.model import load_vlm_for_inference, predict_dwell_batch
from totvlm.scoring import metrics_by_navigation


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full-val decode of a trained adapter (λ selection)."
    )
    ap.add_argument("--config", required=True,
                    help="the config the adapter was trained with, e.g. "
                         "configs/sweeps/lam050.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="adapter dir (default: <output_dir>/final)")
    ap.add_argument("--out", default=None,
                    help="metrics JSON (default: "
                         "<output_dir>/val_full_metrics.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_dir"])
    checkpoint = args.checkpoint or str(out_dir / "final")
    if not Path(checkpoint).is_dir():
        raise SystemExit(f"no adapter at {checkpoint} — train this λ first "
                         f"(python -m totvlm.train --config {args.config})")
    out_path = Path(args.out) if args.out else out_dir / "val_full_metrics.json"

    df = pd.read_parquet(cfg["paths"]["rows_final"])
    train = df[df["split"] == "train"]
    # dwell_s is already winsorized at the train-split p95 — its train max IS
    # the cap (same recovery as totvlm.train / scripts/evaluate.py)
    winsor_cap = float(train["dwell_s"].max())
    train_median_s = float(train["dwell_s"].median())

    val = df[df["split"] == "val"]
    img = cfg["image"]
    examples = build_vlm_examples(
        val,
        winsor_cap=winsor_cap,
        max_side=img["max_side"],
        min_pixels=img["min_pixels"],
        max_pixels=img["max_pixels"],
        include_task_title=cfg["data"]["include_task_title"],
        scaffold=bool(cfg["data"].get("scaffold")),
    )
    # gold labels in the same row order build_vlm_examples uses (same mask,
    # same iteration order) — avoids loading every image twice
    val_rows = val[val["img_resolved"] & val["img_path"].notna()]
    if len(val_rows) != len(examples):
        raise SystemExit("row/example count mismatch — build_vlm_examples "
                         "filter changed? refuse to score misaligned rows")
    gold_s = val_rows["dwell_s"].to_numpy(dtype=float)
    is_nav = val_rows["is_navigation"].to_numpy(dtype=bool)

    print(f"Decoding FULL val split: {len(examples)} rows · "
          f"checkpoint {checkpoint}", flush=True)
    model, processor = load_vlm_for_inference(
        checkpoint, max_seq_length=cfg["model"]["max_seq_length"]
    )
    outputs = predict_dwell_batch(
        model, processor, examples,
        batch_size=cfg["eval"]["decode_batch_size"],
        max_new_tokens=cfg["eval"]["max_new_tokens"],
    )

    parsed = [parse_dwell_output_lenient(t) for t in outputs]
    tiers = [t for _, t in parsed]
    pred_s = np.array([
        train_median_s if v is None else min(max(v, 0.0), winsor_cap)
        for v, _ in parsed
    ])
    metrics = metrics_by_navigation(
        np.log1p(gold_s), np.log1p(pred_s), is_nav
    )
    tier_counts = {t: tiers.count(t) for t in PARSE_TIERS}

    payload = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": args.config,
        "checkpoint": checkpoint,
        "lambda": cfg.get("lupi", {}).get("lambda"),
        "n_val": len(examples),
        "tier_counts": tier_counts,
        "parse_failure_rate": tier_counts["fail"] / max(len(tiers), 1),
        "fallback": f"TRAIN median winsorized dwell ({train_median_s:.3f} s)",
        "clipped_to": [0.0, round(winsor_cap, 3)],
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    o = metrics["overall"]
    print(f"\nλ={payload['lambda']}  FULL-val n={o['n']}  "
          f"MAE(log) {o['mae_log']:.4f}  MAE(s) {o['mae_s']:.2f}  "
          f"ρ {o['spearman_rho']:.3f}  "
          f"parse-fail {payload['parse_failure_rate']:.2%}")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
