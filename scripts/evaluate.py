"""
evaluate.py
===========
TEST-split head-to-head (O2): does the screenshot VLM beat the no-image
LightGBM baseline — and is its edge cognitive signal (in-page steps) or
page-load latency (navigation steps)?

    uv run python scripts/evaluate.py [--config configs/eval.yaml]
                                      [--predict-only | --report-only]

Two stages, all knobs in configs/eval.yaml:

1. PREDICT (CUDA GPU): load the trained adapters, batched greedy decode over
   every TEST row with a resolved screenshot, cache RAW outputs to
   artifacts/vlm_test_preds.parquet. Runs only when the cache is missing
   (or with --predict-only). Re-running the report never re-runs the GPU.

2. REPORT (CPU, any machine): parse raw outputs (lenient tiers; failures
   imputed with the TRAIN median and the rate reported — never hidden),
   reload the LightGBM baseline and score it on the SAME rows (axTree
   features fetched via the resumable cache), compute floors, and write:
     - artifacts/eval_report.md   (head-to-head + nav breakdown + calibration)
     - artifacts/calibration.png
     - artifacts/qualitative/     (6 annotated screenshots, pred vs actual)

Evaluation set = TEST rows with img_resolved. The head-to-head table uses
the subset where the baseline's axTree also resolved, so every model is
scored on identical rows; VLM-only metrics on the full set are reported too.
This evaluates the internal TEST split only — the read-only external
validation set is untouched (see scripts/validate_external.py).
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

from totvlm.config import load_config
from totvlm.data import (
    PARSE_TIERS,
    build_vlm_examples,
    parse_dwell_output_lenient,
)
from totvlm.scoring import calibration_table, metrics_by_navigation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

KEY = ["trajectory_id", "tab_id", "unit_index"]
SUBSETS = ("overall", "navigation", "in_page")


# ── Shared data selection ─────────────────────────────────────────────────────

def eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    """TEST rows the VLM can score: resolved screenshot, stable order."""
    return df[(df["split"] == "test") & df["img_resolved"]].sort_values(
        KEY, kind="mergesort"
    ).reset_index(drop=True)


def winsor_cap_from_train(df: pd.DataFrame) -> float:
    """dwell_s is winsorized at the train p95 — its train max IS the cap."""
    return float(df.loc[df["split"] == "train", "dwell_s"].max())


# ── Stage 1: GPU prediction (cached) ──────────────────────────────────────────

def run_predict(cfg: dict, vcfg: dict, rows: pd.DataFrame,
                winsor_cap: float) -> None:
    import torch

    from totvlm.model import load_vlm_for_inference, predict_dwell_batch

    if not torch.cuda.is_available():
        sys.exit("PREDICT stage needs a CUDA GPU (`uv sync --extra vlm`). "
                 "If artifacts/vlm_test_preds.parquet already exists, use "
                 "--report-only.")

    adapters = cfg["paths"]["adapters"]
    if not Path(adapters).exists():
        sys.exit(f"No trained adapters at {adapters} — run totvlm.train first.")

    log.info(f"loading adapters from {adapters}")
    model, processor = load_vlm_for_inference(
        adapters,
        max_seq_length=vcfg["model"]["max_seq_length"],
        load_in_4bit=vcfg["model"]["load_in_4bit"],
    )
    img = vcfg["image"]
    examples = build_vlm_examples(
        rows,
        winsor_cap=winsor_cap,
        max_side=img["max_side"],
        min_pixels=img["min_pixels"],
        max_pixels=img["max_pixels"],
        include_task_title=vcfg["data"]["include_task_title"],
    )
    assert len(examples) == len(rows)

    log.info(f"decoding {len(examples)} TEST screens ...")
    raw = predict_dwell_batch(
        model, processor, examples,
        batch_size=cfg["predict"]["batch_size"],
        max_new_tokens=cfg["predict"]["max_new_tokens"],
    )

    out = rows[KEY].copy()
    out["raw_output"] = raw
    dest = Path(cfg["paths"]["vlm_preds"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, compression="zstd", index=False)
    log.info(f"raw predictions → {dest}")


# ── Stage 2a: parse VLM outputs (fallback documented, rate reported) ──────────

def parse_vlm_preds(rows: pd.DataFrame, preds: pd.DataFrame, *,
                    winsor_cap: float, train_median_s: float,
                    clip_to_winsor: bool) -> tuple[pd.DataFrame, dict]:
    """Join raw outputs onto eval rows; parse; impute failures.
    Returns (rows + vlm_pred_s/vlm_pred_log/parse_tier, parse stats)."""
    merged = rows.merge(preds, on=KEY, how="left", validate="1:1")
    n_missing = int(merged["raw_output"].isna().sum())
    if n_missing:
        raise SystemExit(
            f"{n_missing} eval rows have no cached prediction — the preds "
            f"cache is stale; rerun the PREDICT stage.")

    parsed = [parse_dwell_output_lenient(t) for t in merged["raw_output"]]
    tiers = [t for _, t in parsed]
    values = []
    for v, _tier in parsed:
        if v is None:
            v = train_median_s        # documented fallback: TRAIN median
        elif clip_to_winsor:
            v = min(max(v, 0.0), winsor_cap)   # training-target range
        values.append(v)

    merged["parse_tier"] = tiers
    merged["vlm_pred_s"] = values
    merged["vlm_pred_log"] = np.log1p(merged["vlm_pred_s"])

    n = len(merged)
    tier_counts = {t: tiers.count(t) for t in PARSE_TIERS}
    stats = {
        "n": n,
        "tier_counts": tier_counts,
        "parse_failure_rate": tier_counts["fail"] / n,
        "fallback": f"TRAIN median winsorized dwell ({train_median_s:.3f} s)",
        "clipped_to": [0.0, round(winsor_cap, 3)] if clip_to_winsor else None,
    }
    return merged, stats


# ── Stage 2b: baseline + floors on the same rows ──────────────────────────────

def baseline_preds(cfg: dict, bcfg: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """LightGBM predictions for eval rows whose axTree resolves.
    Returns rows[KEY] + lgbm_pred_log, one row per resolved axTree."""
    import lightgbm as lgb

    from totvlm.features import (
        build_feature_frame,
        feature_columns,
        fetch_axtree_features,
    )

    fcfg = bcfg["features"]
    log.info(f"fetching axTree features for {len(rows)} eval rows "
             f"(resumable cache: {fcfg['axtree_cache_dir']})")
    feats = fetch_axtree_features(
        rows["axtree_ref"],
        cache_dir=fcfg["axtree_cache_dir"],
        concurrency=fcfg["concurrency"],
        timeout_s=fcfg["timeout_s"],
        max_depth=fcfg["max_tree_depth"],
    )
    vocab = list(fcfg["action_vocab"])
    frame = build_feature_frame(rows, feats, vocab)
    n_failed = int((~frame["ax_resolved"]).sum())
    frame = frame[frame["ax_resolved"]]
    log.info(f"axTree resolved for {len(frame)}/{len(rows)} rows "
             f"({n_failed} fetch/parse failures excluded from baseline)")

    booster = lgb.Booster(model_file=cfg["paths"]["baseline_model"])
    out = frame[KEY].copy()
    out["lgbm_pred_log"] = booster.predict(frame[feature_columns(vocab)])
    return out


# ── Report pieces ─────────────────────────────────────────────────────────────

def metrics_table(models: dict[str, np.ndarray], y: np.ndarray,
                  nav: np.ndarray) -> tuple[list[str], dict]:
    """One combined markdown table: model × subset rows. Returns (lines,
    {model: metrics_by_navigation dict})."""
    all_metrics = {
        name: metrics_by_navigation(y, pred, nav)
        for name, pred in models.items()
    }
    lines = [
        "| model | subset | n | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) "
        "| Spearman ρ | mean actual s | mean pred s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, by_nav in all_metrics.items():
        for subset in SUBSETS:
            m = by_nav[subset]
            if m.get("n", 0) == 0:
                lines.append(f"| {name} | {subset} | 0 | – | – | – | – | – "
                             f"| – | – |")
                continue
            rho = m["spearman_rho"]
            lines.append(
                f"| {name} | {subset} | {m['n']} | {m['mae_log']:.4f} "
                f"| {m['rmse_log']:.4f} | {m['mae_s']:.2f} "
                f"| {m['rmse_s']:.2f} "
                f"| {'–' if np.isnan(rho) else f'{rho:.4f}'} "
                f"| {m['mean_actual_s']:.2f} | {m['mean_pred_s']:.2f} |"
            )
    return lines, all_metrics


def edge_paragraph(vlm: dict, lgbm: dict) -> str:
    """Where does the VLM's edge over LightGBM live — in-page (cognitive
    signal) or navigation (page-load latency bundled into the dwell)?"""
    d_in = lgbm["in_page"]["mae_log"] - vlm["in_page"]["mae_log"]
    d_nav = lgbm["navigation"]["mae_log"] - vlm["navigation"]["mae_log"]
    n_in, n_nav = vlm["in_page"]["n"], vlm["navigation"]["n"]

    where = (
        f"On identical rows, the VLM's MAE(log) edge over LightGBM is "
        f"**{d_in:+.4f} on in-page steps** (n={n_in}) and "
        f"**{d_nav:+.4f} on navigation steps** (n={n_nav}); positive = VLM "
        f"better."
    )
    if d_in <= 0 and d_nav <= 0:
        verdict = (
            "The VLM does not beat the baseline on either subset, so there "
            "is no edge to attribute — the screenshot alone recovers no more "
            "than the interpretable structural features here."
        )
    elif d_in > d_nav:
        verdict = (
            "The advantage concentrates in **in-page** interactions, where "
            "no page load is bundled into the dwell — consistent with the "
            "screenshot capturing genuine cognitive/visual-complexity "
            "signal rather than latency artifacts."
        )
    else:
        verdict = (
            "The advantage concentrates in **navigation** steps, whose "
            "dwells bundle page-load latency — so much of the edge likely "
            "reflects the model recognizing slow-loading page types, not "
            "cognitive processing cost. Treat the in-page numbers as the "
            "honest estimate of recoverable cognitive signal."
        )
    return f"{where} {verdict}"


def save_calibration_png(cal: pd.DataFrame, dest: Path) -> None:
    """Decile reliability plot: mean predicted vs mean actual seconds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)
    lim = 1.05 * max(cal["mean_pred_s"].max(), cal["mean_actual_s"].max())
    ax.plot([0, lim], [0, lim], ls="--", lw=1, c="#9aa0a6",
            label="perfect calibration", zorder=1)
    ax.plot(cal["mean_pred_s"], cal["mean_actual_s"], c="#4059ad", lw=1.5,
            zorder=2)
    ax.scatter(cal["mean_pred_s"], cal["mean_actual_s"], s=28, c="#4059ad",
               zorder=3, label="prediction decile")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mean predicted dwell (s)")
    ax.set_ylabel("mean actual dwell (s)")
    ax.set_title("VLM calibration on TEST\n(decile bins by prediction)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def save_qualitative(rows: pd.DataFrame, n: int, banner_h: int,
                     out_dir: Path, seed: int) -> list[dict]:
    """Pick n screens spanning the error spectrum (best/median/worst thirds
    by |log error|), annotate pred vs actual above each screenshot."""
    from PIL import Image, ImageDraw, ImageFont

    ranked = rows.assign(
        abs_err_log=(rows["vlm_pred_log"] - rows["y"]).abs()
    ).sort_values("abs_err_log", kind="mergesort").reset_index(drop=True)

    k = n // 3
    mid = len(ranked) // 2
    picks = pd.concat([
        ranked.head(k).assign(bucket="best"),
        ranked.iloc[mid - k // 2: mid - k // 2 + k].assign(bucket="median"),
        ranked.tail(n - 2 * k).assign(bucket="worst"),
    ])

    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    entries = []
    for i, r in enumerate(picks.itertuples(index=False)):
        shot = Image.open(r.img_path).convert("RGB")
        canvas = Image.new("RGB", (shot.width, shot.height + banner_h),
                           "white")
        canvas.paste(shot, (0, banner_h))
        caption = (
            f"[{r.bucket}] pred {r.vlm_pred_s:.1f} s · actual {r.dwell_s:.1f} s"
            f" · {'navigation' if r.is_navigation else 'in-page'}"
            f" · {r.trajectory_id}/{int(r.unit_index)}"
        )
        ImageDraw.Draw(canvas).text((8, banner_h // 4), caption,
                                    fill="black", font=font)
        fname = f"{i:02d}_{r.bucket}_{r.trajectory_id}_{int(r.unit_index)}.png"
        canvas.save(out_dir / fname)
        entries.append({
            "file": fname, "bucket": r.bucket,
            "pred_s": round(float(r.vlm_pred_s), 1),
            "actual_s": round(float(r.dwell_s), 1),
            "is_navigation": bool(r.is_navigation),
            "raw_output": r.raw_output,
        })
    return entries


# ── Stage 2: report ───────────────────────────────────────────────────────────

def run_report(cfg: dict, bcfg: dict, df: pd.DataFrame,
               rows: pd.DataFrame, cfg_path: str) -> None:
    scfg = cfg["scoring"]
    winsor_cap = winsor_cap_from_train(df)
    train_median_s = float(df.loc[df["split"] == "train", "dwell_s"].median())

    preds = pd.read_parquet(cfg["paths"]["vlm_preds"])
    rows, parse_stats = parse_vlm_preds(
        rows, preds,
        winsor_cap=winsor_cap,
        train_median_s=train_median_s,
        clip_to_winsor=scfg["clip_pred_to_winsor"],
    )

    # Baseline on the same rows (subset where the axTree resolved)
    bl = baseline_preds(cfg, bcfg, rows)
    common = rows.merge(bl, on=KEY, how="inner", validate="1:1")

    # Floors: TRAIN mean/median of y, constant on every row (same recipe as
    # the baseline report)
    y_train = df.loc[df["split"] == "train", "y"].to_numpy()
    floors = {"train-mean floor": float(np.mean(y_train)),
              "train-median floor": float(np.median(y_train))}

    # Head-to-head on identical rows
    y_c = common["y"].to_numpy()
    nav_c = common["is_navigation"].to_numpy()
    models = {name: np.full_like(y_c, const)
              for name, const in floors.items()}
    models["LightGBM (no image)"] = common["lgbm_pred_log"].to_numpy()
    models["VLM (screenshot)"] = common["vlm_pred_log"].to_numpy()
    head_lines, head_metrics = metrics_table(models, y_c, nav_c)

    # VLM on the FULL eval set (rows the baseline couldn't cover included)
    y_f = rows["y"].to_numpy()
    nav_f = rows["is_navigation"].to_numpy()
    full_lines, _ = metrics_table(
        {"VLM (screenshot)": rows["vlm_pred_log"].to_numpy()}, y_f, nav_f)

    # Calibration (full eval set) + qualitative screenshots
    cal, ece = calibration_table(y_f, rows["vlm_pred_log"].to_numpy(),
                                 n_bins=scfg["calibration_bins"])
    cal_png = Path(cfg["paths"]["calibration_png"])
    save_calibration_png(cal, cal_png)

    qual_dir = Path(cfg["paths"]["qualitative_dir"])
    qual = save_qualitative(rows, cfg["qualitative"]["n"],
                            cfg["qualitative"]["banner_height"],
                            qual_dir, cfg["seed"])

    paragraph = edge_paragraph(head_metrics["VLM (screenshot)"],
                               head_metrics["LightGBM (no image)"])

    tc = parse_stats["tier_counts"]
    lines = [
        "# Evaluation report — VLM vs baseline on held-out TEST domains (O2)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"config `{cfg_path}` · seed {cfg['seed']}_",
        "",
        "Target: `y = log1p(dwell_s)`, dwell winsorized at the train-split "
        f"p95 cap ({winsor_cap:.3f} s). Splits are domain-disjoint "
        "(`artifacts/splits.json`); TEST domains were never seen in "
        "training or tuning. Metrics via `totvlm.scoring` — identical to "
        "the baseline report.",
        "",
        "## Coverage & parse accounting",
        "",
        f"- Eval set: **{parse_stats['n']}** TEST rows with a resolved "
        f"screenshot ({int(nav_f.sum())} navigation / "
        f"{int((~nav_f).sum())} in-page)",
        f"- Head-to-head subset (axTree also resolved for LightGBM): "
        f"**{len(common)}** rows",
        f"- VLM parse tiers: strict **{tc['strict']}** · labeled "
        f"**{tc['labeled']}** · bare number **{tc['bare_number']}** · "
        f"failed **{tc['fail']}**",
        f"- **Parse failure rate: {parse_stats['parse_failure_rate']:.2%}** "
        f"— failures imputed with the {parse_stats['fallback']}, never "
        f"dropped",
        ]
    if parse_stats["clipped_to"]:
        lines.append(
            f"- Parsed predictions clipped to {parse_stats['clipped_to']} s "
            f"(the training-target range)")
    lines += [
        "",
        "## Head-to-head (identical rows)",
        "",
        *head_lines,
        "",
        "## VLM on the full eval set",
        "",
        *full_lines,
        "",
        "## Where does the edge live? (O2 answer)",
        "",
        paragraph,
        "",
        "## Calibration (VLM, full eval set, decile bins by prediction)",
        "",
        f"![calibration]({cal_png.name})",
        "",
        f"ECE-style weighted gap: **{ece:.2f} s**",
        "",
        "| bin | n | mean pred (s) | mean actual (s) | gap (s) |",
        "|---|---|---|---|---|",
        *[f"| {int(r.bin)} | {int(r.n)} | {r.mean_pred_s:.2f} "
          f"| {r.mean_actual_s:.2f} | {r.gap_s:.2f} |"
          for r in cal.itertuples()],
        "",
        f"## Qualitative examples (`{qual_dir}/`)",
        "",
        "| file | bucket | pred (s) | actual (s) | step | raw output |",
        "|---|---|---|---|---|---|",
        *[f"| {q['file']} | {q['bucket']} | {q['pred_s']} | {q['actual_s']} "
          f"| {'navigation' if q['is_navigation'] else 'in-page'} "
          f"| `{q['raw_output']}` |" for q in qual],
        "",
        "## Full metrics (JSON)",
        "",
        "```json",
        json.dumps(
            {
                "head_to_head": head_metrics,
                "parse_stats": parse_stats,
                "calibration_ece_s": ece,
                "rows": {"eval_set": parse_stats["n"],
                         "head_to_head": len(common)},
            },
            indent=2,
        ),
        "```",
        "",
    ]
    report = Path(cfg["paths"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    log.info(f"report → {report} · calibration → {cal_png} · "
             f"qualitative → {qual_dir}/")
    log.info(json.dumps(
        {m: head_metrics[m]["overall"]["mae_log"] for m in head_metrics},
        indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--predict-only", action="store_true",
                       help="run the GPU decode stage and stop")
    group.add_argument("--report-only", action="store_true",
                       help="fail rather than run the GPU stage")
    args = ap.parse_args()

    cfg = load_config(args.config)
    vcfg = load_config(cfg["vlm_config"])
    bcfg = load_config(cfg["baseline_config"])
    np.random.seed(cfg["seed"])

    df = pd.read_parquet(cfg["paths"]["rows_final"])
    rows = eval_rows(df)
    log.info(f"eval set: {len(rows)} TEST rows with resolved screenshots")

    preds_path = Path(cfg["paths"]["vlm_preds"])
    if args.report_only and not preds_path.exists():
        sys.exit(f"--report-only but {preds_path} is missing — run the "
                 f"PREDICT stage on a GPU machine first.")
    if not args.report_only and not preds_path.exists():
        run_predict(cfg, vcfg, rows, winsor_cap_from_train(df))
    elif not args.report_only:
        log.info(f"using cached predictions: {preds_path}")
    if args.predict_only:
        return

    run_report(cfg, bcfg, df, rows, args.config)


if __name__ == "__main__":
    main()
