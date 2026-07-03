"""
validate_external.py
====================
Zero-shot external validation (O3): run the FROZEN VLM checkpoint once over
the read-only external set and report rank agreement (Spearman + bootstrap
CI) against measured human times.

    uv run python scripts/validate_external.py [--config configs/external.yaml]
                                               [--predict-only | --report-only]
                                               [--allow-rerun]

CLAUDE.md contract, enforced here:
- The external set is evaluated EXACTLY ONCE, zero-shot. The predict stage
  refuses to run if predictions already exist, and the report stage refuses
  to overwrite an existing report, unless --allow-rerun is passed — in which
  case the rerun is recorded prominently in the report itself.
- Nothing computed here feeds back into training/tuning: no thresholds,
  no feature choices, no winsor cap. Rank agreement only.

Two stages (same pattern as scripts/evaluate.py):
1. PREDICT (GPU): frozen adapters, batched greedy decode of every external
   screenshot with the exact training prompt → artifacts/external_preds.parquet.
2. REPORT (CPU): lenient parse (failure rate stated; unparseable items are
   EXCLUDED from the correlation, never imputed — a constant would inject
   fake rank information), Spearman + seeded bootstrap CI overall and per
   category, scatter plot, artifacts/external_report.md.

Baseline: the no-image LightGBM consumes axTree features, which do not exist
for external screenshots (VSGUI10K ships images + gaze only) — reported as
not applicable rather than silently skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from totvlm.config import load_config
from totvlm.data import PARSE_TIERS, parse_dwell_output_lenient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_items(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["paths"]["external_root"]) / cfg["source"]
    items_csv = root / "items.csv"
    if not items_csv.exists():
        sys.exit(f"{items_csv} missing — run scripts/prepare_external.py "
                 f"first.")
    return pd.read_csv(items_csv).sort_values("item_id",
                                              kind="mergesort").reset_index(
        drop=True)


# ── Stage 1: zero-shot decode (GPU, once) ─────────────────────────────────────

def run_predict(cfg: dict, vcfg: dict, items: pd.DataFrame) -> None:
    import torch

    from totvlm.data import build_inference_examples
    from totvlm.model import load_vlm_for_inference, predict_dwell_batch

    if not torch.cuda.is_available():
        sys.exit("PREDICT stage needs a CUDA GPU (`uv sync --extra vlm`). "
                 "If artifacts/external_preds.parquet exists, use "
                 "--report-only.")
    adapters = Path(cfg["paths"]["adapters"])
    if not adapters.exists():
        sys.exit(f"no frozen checkpoint at {adapters} — train first.")

    log.info(f"loading FROZEN adapters from {adapters} (zero-shot, no "
             f"further tuning)")
    model, processor = load_vlm_for_inference(
        str(adapters),
        max_seq_length=vcfg["model"]["max_seq_length"],
        load_in_4bit=vcfg["model"]["load_in_4bit"],
    )
    img = vcfg["image"]
    examples = build_inference_examples(
        items["screenshot_path"].tolist(),
        max_side=img["max_side"],
        min_pixels=img["min_pixels"],
        max_pixels=img["max_pixels"],
    )
    log.info(f"decoding {len(examples)} external screenshots ...")
    raw = predict_dwell_batch(
        model, processor, examples,
        batch_size=cfg["predict"]["batch_size"],
        max_new_tokens=cfg["predict"]["max_new_tokens"],
    )
    out = items[["item_id"]].copy()
    out["raw_output"] = raw
    dest = Path(cfg["paths"]["preds"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, compression="zstd", index=False)
    log.info(f"raw predictions → {dest}")


# ── Stage 2: rank agreement + report ──────────────────────────────────────────

def spearman_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int, ci: float,
                seed: int) -> dict:
    """Spearman rho with a seeded percentile-bootstrap CI over items."""
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # a degenerate resample (constant array) yields nan; keep it — it
        # widens the CI honestly rather than hiding instability
        boots[b] = spearmanr(x[idx], y[idx])[0]
    alpha = (1 - ci) / 2
    lo, hi = np.nanquantile(boots, [alpha, 1 - alpha])
    return {"n": n, "rho": round(rho, 4), "ci": ci,
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "n_boot": n_boot}


def save_scatter(df: pd.DataFrame, overall: dict, dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"web": "#4059ad", "desktop": "#e07a3f", "mobile": "#3f8e6d"}
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=150)
    for cat, part in df.groupby("category"):
        ax.scatter(part["human_time_s"], part["pred_s"], s=16, alpha=0.6,
                   c=colors.get(cat, "#666"), label=f"{cat} (n={len(part)})")
    ax.set_xlabel("measured human time (s)")
    ax.set_ylabel("VLM predicted dwell (s)")
    ax.set_title(
        "Zero-shot rank agreement on the external set\n"
        f"Spearman ρ = {overall['rho']:.3f} "
        f"[{overall['ci_lo']:.3f}, {overall['ci_hi']:.3f}] "
        f"({int(overall['ci'] * 100)}% bootstrap CI, n={overall['n']})")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def run_report(cfg: dict, items: pd.DataFrame, rerun: bool) -> None:
    rcfg = cfg["report"]
    preds = pd.read_parquet(cfg["paths"]["preds"])
    df = items.merge(preds, on="item_id", how="left", validate="1:1")
    if df["raw_output"].isna().any():
        sys.exit("preds cache does not cover the item set — stale cache?")

    parsed = [parse_dwell_output_lenient(t) for t in df["raw_output"]]
    df["parse_tier"] = [t for _, t in parsed]
    df["pred_s"] = [v for v, _ in parsed]
    tiers = {t: int((df["parse_tier"] == t).sum()) for t in PARSE_TIERS}
    fail_rate = tiers["fail"] / len(df)
    ok = df[df["pred_s"].notna()].copy()
    log.info(f"parsed {len(ok)}/{len(df)} (failure rate {fail_rate:.2%}, "
             f"failures EXCLUDED from correlation)")
    if len(ok) < 3:
        sys.exit("fewer than 3 parsed predictions — nothing to correlate")

    overall = spearman_ci(
        ok["pred_s"].to_numpy(float), ok["human_time_s"].to_numpy(float),
        n_boot=rcfg["bootstrap_samples"], ci=rcfg["ci"], seed=cfg["seed"])
    by_cat = {
        cat: spearman_ci(
            part["pred_s"].to_numpy(float),
            part["human_time_s"].to_numpy(float),
            n_boot=rcfg["bootstrap_samples"], ci=rcfg["ci"],
            seed=cfg["seed"])
        for cat, part in ok.groupby("category") if len(part) >= 3
    }

    scatter = Path(cfg["paths"]["scatter_png"])
    save_scatter(ok, overall, scatter)

    def fmt(name: str, m: dict) -> str:
        return (f"| {name} | {m['n']} | {m['rho']:.4f} "
                f"| [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}] |")

    src = cfg["source"]
    primary = rcfg["primary_category"]
    lines = [
        "# External validation report — frozen VLM, zero-shot (O3)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"config `configs/external.yaml` · seed {cfg['seed']} · "
        f"source **{src}**_",
        "",
        "**This read-only set was evaluated exactly once, zero-shot, with "
        "the frozen checkpoint.** Nothing here feeds back into training or "
        "tuning (CLAUDE.md; path access enforced by "
        "`tests/test_external_guard.py`).",
    ]
    if rerun:
        lines += [
            "",
            "> ⚠️ **RERUN**: this report was regenerated with "
            "--allow-rerun; the evaluate-once guarantee was manually "
            "overridden. Treat with care.",
        ]
    if src == "vsgui10k":
        lines += [
            "",
            "⚠️ **Measure caveat**: VSGUI10K (Putkonen et al. 2025, "
            "osf.io/hmg9b) measures visual **search** time — one component "
            "of Time-on-Task, not the full per-screen dwell. TaskSense "
            "(first choice) has no public data (see "
            "`artifacts/external_card.md`). Rank agreement here is a weaker "
            "but fully independent check.",
        ]
    lines += [
        "",
        "## Parse accounting",
        "",
        f"- Items: **{len(df)}** · parsed: **{len(ok)}** · "
        f"tiers: strict {tiers['strict']} / labeled {tiers['labeled']} / "
        f"bare number {tiers['bare_number']} / failed {tiers['fail']}",
        f"- **Parse failure rate: {fail_rate:.2%}** — unparseable items are "
        "excluded from the correlation (imputing a constant would inject "
        "fake rank information), and the exclusion is reported here.",
        "",
        "## Rank agreement (Spearman ρ, "
        f"{int(rcfg['ci'] * 100)}% bootstrap CI, "
        f"{rcfg['bootstrap_samples']} resamples)",
        "",
        "| subset | n | ρ | CI |",
        "|---|---|---|---|",
        fmt("ALL", overall),
        *[fmt(f"{cat}" + (" (in-domain, primary)" if cat == primary else ""),
              m) for cat, m in sorted(by_cat.items())],
        "",
        f"![scatter]({scatter.name})",
        "",
        "## Baseline",
        "",
        "The no-image LightGBM baseline consumes axTree structural "
        "features; the external screenshots ship without axTrees "
        "(VSGUI10K provides images + gaze data only), so the baseline is "
        "**not applicable** on this set — stated rather than silently "
        "skipped.",
        "",
        "## Full results (JSON)",
        "",
        "```json",
        json.dumps({"overall": overall, "by_category": by_cat,
                    "parse_tiers": tiers,
                    "parse_failure_rate": round(fail_rate, 4),
                    "rerun": rerun}, indent=2),
        "```",
        "",
    ]
    report = Path(cfg["paths"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    log.info(f"report → {report} · scatter → {scatter}")
    log.info(f"Spearman rho={overall['rho']} "
             f"CI=[{overall['ci_lo']}, {overall['ci_hi']}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--predict-only", action="store_true")
    group.add_argument("--report-only", action="store_true")
    ap.add_argument("--allow-rerun", action="store_true",
                    help="override the evaluate-once guard (recorded in the "
                         "report)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    vcfg = load_config(cfg["vlm_config"])
    items = load_items(cfg)
    log.info(f"external items: {len(items)} (source {cfg['source']})")

    preds_path = Path(cfg["paths"]["preds"])
    report_path = Path(cfg["paths"]["report"])

    if not args.report_only:
        if preds_path.exists() and not args.allow_rerun:
            sys.exit(
                f"{preds_path} already exists — the external set is "
                f"evaluated exactly ONCE (CLAUDE.md). Use --report-only to "
                f"re-render the report, or --allow-rerun to override (the "
                f"override is recorded in the report).")
        if not preds_path.exists() or args.allow_rerun:
            run_predict(cfg, vcfg, items)
    if args.predict_only:
        return

    if not preds_path.exists():
        sys.exit(f"{preds_path} missing — run the PREDICT stage on a GPU "
                 f"machine first.")
    if report_path.exists() and not args.allow_rerun:
        sys.exit(f"{report_path} already exists — evaluate-once. Use "
                 f"--allow-rerun to regenerate (recorded in the report).")
    run_report(cfg, items, rerun=args.allow_rerun)


if __name__ == "__main__":
    main()
