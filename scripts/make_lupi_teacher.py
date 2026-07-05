"""LUPI teacher: out-of-fold LightGBM predictions over the VLM's TRAIN rows.

    uv run python scripts/make_lupi_teacher.py [--config configs/lupi.yaml]
                                               [--limit N]

Thin driver: loads the VLM's train rows, fetches the PRIVILEGED features
(axTree structural stats, nav flag, step index, click-target area — train-time
only), and calls totvlm.lupi.oof_teacher_predictions to score them out of
fold. Those predictions become the soft half of BOTH trained conditions'
target; the blend happens in totvlm.train via `lupi.lambda` (configs/vlm.yaml).
The leak discipline (domain-grouped folds, val/test never touched) lives in
totvlm.lupi. Writes lupi_teacher_preds.parquet + the teacher card."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from totvlm.config import load_config
from totvlm.data import ROW_KEY
from totvlm.features import (
    build_feature_frame,
    feature_columns,
    fetch_axtree_features,
)
from totvlm.lupi import oof_teacher_predictions
from totvlm.scoring import metrics_by_navigation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lupi.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the row count after the stable sort "
                         "(smoke tests only — the card records it)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bcfg = load_config(cfg["baseline_config"])
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    fcfg = bcfg["features"]
    k = cfg["teacher"]["k_folds"]

    # Teacher preds are needed exactly where the VLM trains: TRAIN rows with
    # a resolved screenshot. Rows without an axTree URL cannot be covered —
    # they keep the true label downstream, and the coverage is reported.
    df = pd.read_parquet(cfg["paths"]["rows_final"])
    vlm_train = df[(df["split"] == "train") & df["img_resolved"]]
    has_ax = vlm_train["axtree_ref"].notna() & (vlm_train["axtree_ref"] != "")
    rows = vlm_train[has_ax].sort_values(ROW_KEY, kind="mergesort")
    n_no_axtree = int((~has_ax).sum())
    if args.limit is not None:
        rows = rows.head(args.limit)
    log.info(
        f"{len(vlm_train)} VLM train rows → {len(rows)} with an axTree URL "
        f"({n_no_axtree} uncoverable)"
        + (f" · LIMIT {args.limit}" if args.limit else "")
    )

    feats = fetch_axtree_features(
        rows["axtree_ref"],
        cache_dir=fcfg["axtree_cache_dir"],
        concurrency=fcfg["concurrency"],
        timeout_s=fcfg["timeout_s"],
        max_depth=fcfg["max_tree_depth"],
    )
    vocab = list(fcfg["action_vocab"])
    frame = build_feature_frame(rows, feats, vocab)
    n_fetch_failed = int((~frame["ax_resolved"]).sum())
    frame = frame[frame["ax_resolved"]]
    log.info(f"axTree resolved for {len(frame)} rows "
             f"({n_fetch_failed} fetch/parse failures excluded)")
    if frame.empty:
        raise SystemExit("no rows with resolved axTree features — nothing "
                         "for the teacher to learn from")

    lgbm_params = dict(bcfg["lightgbm"], random_state=seed)
    frame, fold_stats = oof_teacher_predictions(
        frame, feature_columns(vocab),
        k=k, seed=seed,
        lgbm_params=lgbm_params,
        early_stopping_rounds=bcfg["early_stopping_rounds"],
    )

    oof_metrics = metrics_by_navigation(
        frame["y"].to_numpy(),
        frame["teacher_pred_log"].to_numpy(),
        frame["is_navigation"].to_numpy(),
    )

    out = frame[ROW_KEY + ["teacher_pred_log", "fold"]].reset_index(drop=True)
    dest = Path(cfg["paths"]["teacher_preds"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, compression="zstd", index=False)

    coverage = len(out) / max(len(vlm_train), 1)
    payload = {
        "k_folds": k,
        "folds": fold_stats,
        "oof_metrics": oof_metrics,
        "rows": {
            "vlm_train": len(vlm_train),
            "with_axtree_url": len(rows),
            "covered": len(out),
            "coverage_of_vlm_train": round(coverage, 4),
        },
        "excluded": {
            "no_axtree_url": n_no_axtree,
            "axtree_fetch_failed": n_fetch_failed,
        },
        "limit": args.limit,
    }
    lines = [
        "# LUPI teacher card — out-of-fold LightGBM on privileged features",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"config `{args.config}` · seed {seed} · k={k} domain-grouped folds_",
        "",
        "Privileged features (axTree stats, nav flag, step index, click "
        "area) exist at TRAIN time only. Each train row's prediction comes "
        "from a booster that never saw its registrable domain; these become "
        "the soft half of both trained conditions' target (λ in "
        "configs/vlm.yaml). Val/test targets are never blended; inference "
        "stays screenshot-only.",
        "",
        f"- Coverage: **{len(out)}** of {len(vlm_train)} VLM train rows "
        f"(**{coverage:.1%}**) — uncovered rows keep the true label",
        f"- Excluded: {n_no_axtree} without an axTree URL · "
        f"{n_fetch_failed} fetch/parse failures",
        *([f"- ⚠️ `--limit {args.limit}` was set — smoke run, not the full "
           f"teacher"] if args.limit else []),
        "",
        "## Out-of-fold quality (log1p target space)",
        "",
        "| subset | n | MAE (log) | MAE (s) | Spearman ρ |",
        "|---|---|---|---|---|",
        *[
            f"| {name} | {m['n']} | {m['mae_log']:.4f} | {m['mae_s']:.2f} "
            f"| {m['spearman_rho']:.4f} |"
            for name, m in oof_metrics.items() if m.get("n")
        ],
        "",
        "## Folds",
        "",
        "| fold | train rows | early-stop rows | predicted | best iter |",
        "|---|---|---|---|---|",
        *[
            f"| {f['fold']} | {f['n_train']} | {f['n_early_stop']} "
            f"| {f['n_predicted']} | {f['best_iteration']} |"
            for f in fold_stats
        ],
        "",
        "## Full stats (JSON)",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    card = Path(cfg["paths"]["card"])
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text("\n".join(lines))
    log.info(f"teacher preds → {dest} · card → {card}")
    log.info(f"OOF overall: {json.dumps(oof_metrics['overall'])}")


if __name__ == "__main__":
    main()
