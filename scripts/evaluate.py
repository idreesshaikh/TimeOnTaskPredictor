"""TEST-split head-to-head: floors < LightGBM < VLM(screen) < VLM(screen+task),
per screen AND per task (per-trajectory sums).

    uv run python scripts/evaluate.py [--config configs/eval.yaml]
                                      [--predict-only | --report-only]

Stage 1 PREDICT (GPU): batch-decode TEST rows per condition, cached — the
report never re-runs the GPU. Stage 2 REPORT (CPU): lenient parse (failures
imputed with the TRAIN median, rate reported), score the baseline + floors on
the SAME rows, write eval_report.md / eval_metrics.json /
eval_predictions.parquet + qualitative screenshots. Head-to-head rows =
img_resolved AND axTree-resolved so every model sees identical rows; plots
live in make_figures.py. The read-only external set is untouched here."""

from __future__ import annotations

import argparse
import json
import logging
import re
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
from totvlm.scoring import (
    calibration_table,
    metrics_by_navigation,
    regression_metrics,
    task_totals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

KEY = ["trajectory_id", "tab_id", "unit_index"]
SUBSETS = ("overall", "navigation", "in_page")


# Shared data selection


def eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    """TEST rows the VLM can score: resolved screenshot, stable order."""
    return (
        df[(df["split"] == "test") & df["img_resolved"]]
        .sort_values(KEY, kind="mergesort")
        .reset_index(drop=True)
    )


def winsor_cap_from_train(df: pd.DataFrame) -> float:
    """dwell_s is winsorized at the train p95 — its train max IS the cap."""
    return float(df.loc[df["split"] == "train", "dwell_s"].max())


# Stage 1: GPU prediction (cached, one cache per VLM condition)


def adapters_available(ref: str) -> bool:
    """True if `ref` is a local adapter dir OR a HF hub id (`org/name`) —
    a hub id runs the base model zero-shot (the is-fine-tuning-necessary
    row), no local files needed."""
    return Path(ref).exists() or re.fullmatch(r"[\w.-]+/[\w.-]+", ref) is not None


def run_predict(
    cfg: dict, vcfg: dict, rows: pd.DataFrame, winsor_cap: float, mcfg: dict
) -> None:
    import torch

    from totvlm.model import load_vlm_for_inference, predict_dwell_batch

    if not torch.cuda.is_available():
        sys.exit(
            "PREDICT stage needs a CUDA GPU (`uv sync --extra vlm`). "
            "Use --report-only to score whichever prediction caches "
            "already exist."
        )

    adapters = mcfg["adapters"]
    if not adapters_available(adapters):
        sys.exit(
            f"No trained adapters at {adapters} for {mcfg['name']!r} — "
            f"run totvlm.train with its config first."
        )

    log.info(f"[{mcfg['name']}] loading adapters from {adapters}")
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
        include_task_title=mcfg["include_task_title"],
    )
    assert len(examples) == len(rows)

    log.info(f"[{mcfg['name']}] decoding {len(examples)} TEST screens ...")
    raw = predict_dwell_batch(
        model,
        processor,
        examples,
        batch_size=cfg["predict"]["batch_size"],
        max_new_tokens=cfg["predict"]["max_new_tokens"],
    )

    out = rows[KEY].copy()
    out["raw_output"] = raw
    dest = Path(mcfg["preds"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, compression="zstd", index=False)
    log.info(f"[{mcfg['name']}] raw predictions → {dest}")

    del model
    torch.cuda.empty_cache()


# Stage 2a: parse VLM outputs (fallback documented, rate reported)


def parse_vlm_preds(
    rows: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    winsor_cap: float,
    train_median_s: float,
    clip_to_winsor: bool,
) -> tuple[pd.DataFrame, dict]:
    """Join one condition's raw outputs onto eval rows; parse; impute
    failures. Returns (rows + vlm_pred_s/vlm_pred_log/parse_tier, stats).
    Left-merge preserves `rows` order, so pred columns align across
    conditions."""
    merged = rows.merge(preds, on=KEY, how="left", validate="1:1")
    n_missing = int(merged["raw_output"].isna().sum())
    if n_missing:
        raise SystemExit(
            f"{n_missing} eval rows have no cached prediction — the preds "
            f"cache is stale; rerun the PREDICT stage."
        )

    parsed = [parse_dwell_output_lenient(t) for t in merged["raw_output"]]
    tiers = [t for _, t in parsed]
    values = []
    for v, _tier in parsed:
        if v is None:
            v = train_median_s  # documented fallback: TRAIN median
        elif clip_to_winsor:
            v = min(max(v, 0.0), winsor_cap)  # training-target range
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


# Stage 2b: baseline + floors on the same rows


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
    log.info(
        f"fetching axTree features for {len(rows)} eval rows "
        f"(resumable cache: {fcfg['axtree_cache_dir']})"
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
    n_failed = int((~frame["ax_resolved"]).sum())
    frame = frame[frame["ax_resolved"]]
    log.info(
        f"axTree resolved for {len(frame)}/{len(rows)} rows "
        f"({n_failed} fetch/parse failures excluded from baseline)"
    )

    booster = lgb.Booster(model_file=cfg["paths"]["baseline_model"])
    out = frame[KEY].copy()
    out["lgbm_pred_log"] = booster.predict(frame[feature_columns(vocab)])
    return out


# Report pieces


def metrics_table(
    models: dict[str, np.ndarray], y: np.ndarray, nav: np.ndarray
) -> tuple[list[str], dict]:
    """One combined markdown table: model × subset rows. Returns (lines,
    {model: metrics_by_navigation dict})."""
    all_metrics = {
        name: metrics_by_navigation(y, pred, nav) for name, pred in models.items()
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
                lines.append(f"| {name} | {subset} | 0 | – | – | – | – | – | – | – |")
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


def task_level_table(
    models: dict[str, np.ndarray], traj_ids: np.ndarray, y: np.ndarray
) -> tuple[list[str], dict]:
    """Per-task rollup (RQ v2): sum each model's predicted seconds within a
    trajectory, score against the summed actual — the KLM-successor claim
    ('predict how long the task takes'). Totals cover only evaluated steps,
    identically for targets and every model."""
    groups, y_task = task_totals(traj_ids, y)
    lines = [
        "| model | tasks | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) "
        "| Spearman ρ | mean actual s | mean pred s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    all_metrics = {}
    for name, pred in models.items():
        _, p_task = task_totals(traj_ids, pred)
        m = regression_metrics(y_task, p_task)
        all_metrics[name] = m
        rho = m["spearman_rho"]
        lines.append(
            f"| {name} | {m['n']} | {m['mae_log']:.4f} | {m['rmse_log']:.4f} "
            f"| {m['mae_s']:.2f} | {m['rmse_s']:.2f} "
            f"| {'–' if np.isnan(rho) else f'{rho:.4f}'} "
            f"| {m['mean_actual_s']:.2f} | {m['mean_pred_s']:.2f} |"
        )
    return lines, all_metrics


def edge_paragraph(vlm: dict, lgbm: dict) -> str:
    """Where does the screen model's edge over LightGBM live — in-page
    (cognitive signal) or navigation (page-load latency in the dwell)?"""
    d_in = lgbm["in_page"]["mae_log"] - vlm["in_page"]["mae_log"]
    d_nav = lgbm["navigation"]["mae_log"] - vlm["navigation"]["mae_log"]
    n_in, n_nav = vlm["in_page"]["n"], vlm["navigation"]["n"]

    where = (
        f"On identical rows, the screen model's MAE(log) edge over LightGBM "
        f"is **{d_in:+.4f} on in-page steps** (n={n_in}) and "
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


def goal_increment_paragraph(screen: dict, task: dict) -> str:
    """RQ v2's second clause: how much does knowing the user's task add on
    top of the screen? (screen vs screen+task, identical rows)."""
    d_all = screen["overall"]["mae_log"] - task["overall"]["mae_log"]
    d_in = screen["in_page"]["mae_log"] - task["in_page"]["mae_log"]
    d_nav = screen["navigation"]["mae_log"] - task["navigation"]["mae_log"]
    verdict = (
        "knowing the goal adds real predictive information on top of the "
        "pixels — the goal-driven share of dwell is nonzero and now "
        "quantified."
        if d_all > 0
        else "the task title adds little or nothing on top of the pixels here — "
        "on this corpus, the predictable share of dwell is carried by the "
        "screen itself."
    )
    return (
        f"Adding the task title changes MAE(log) by **{d_all:+.4f} overall** "
        f"({d_in:+.4f} in-page, {d_nav:+.4f} navigation; positive = "
        f"screen+task better). Read: {verdict}"
    )


def save_qualitative(
    rows: pd.DataFrame, n: int, banner_h: int, out_dir: Path, seed: int
) -> list[dict]:
    """Pick n screens spanning the error spectrum (best/median/worst thirds
    by |log error|), annotate pred vs actual above each screenshot."""
    from PIL import Image, ImageDraw, ImageFont

    ranked = (
        rows.assign(abs_err_log=(rows["vlm_pred_log"] - rows["y"]).abs())
        .sort_values("abs_err_log", kind="mergesort")
        .reset_index(drop=True)
    )

    k = n // 3
    mid = len(ranked) // 2
    picks = pd.concat(
        [
            ranked.head(k).assign(bucket="best"),
            ranked.iloc[mid - k // 2 : mid - k // 2 + k].assign(bucket="median"),
            ranked.tail(n - 2 * k).assign(bucket="worst"),
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    entries = []
    for i, r in enumerate(picks.itertuples(index=False)):
        shot = Image.open(r.img_path).convert("RGB")
        canvas = Image.new("RGB", (shot.width, shot.height + banner_h), "white")
        canvas.paste(shot, (0, banner_h))
        caption = (
            f"[{r.bucket}] pred {r.vlm_pred_s:.1f} s · actual {r.dwell_s:.1f} s"
            f" · {'navigation' if r.is_navigation else 'in-page'}"
            f" · {r.trajectory_id}/{int(r.unit_index)}"
        )
        ImageDraw.Draw(canvas).text(
            (8, banner_h // 4), caption, fill="black", font=font
        )
        fname = f"{i:02d}_{r.bucket}_{r.trajectory_id}_{int(r.unit_index)}.png"
        canvas.save(out_dir / fname)
        entries.append(
            {
                "file": fname,
                "bucket": r.bucket,
                "pred_s": round(float(r.vlm_pred_s), 1),
                "actual_s": round(float(r.dwell_s), 1),
                "is_navigation": bool(r.is_navigation),
                "raw_output": r.raw_output,
            }
        )
    return entries


# Stage 2: report


def run_report(
    cfg: dict,
    bcfg: dict,
    df: pd.DataFrame,
    rows: pd.DataFrame,
    available: list[dict],
    pending: list[str],
    cfg_path: str,
) -> None:
    scfg = cfg["scoring"]
    winsor_cap = winsor_cap_from_train(df)
    train_median_s = float(df.loc[df["split"] == "train", "dwell_s"].median())

    # Parse every available VLM condition (aligned to `rows` order)
    vlm_frames: dict[str, pd.DataFrame] = {}
    vlm_stats: dict[str, dict] = {}
    for mcfg in available:
        frame, stats = parse_vlm_preds(
            rows,
            pd.read_parquet(mcfg["preds"]),
            winsor_cap=winsor_cap,
            train_median_s=train_median_s,
            clip_to_winsor=scfg["clip_pred_to_winsor"],
        )
        vlm_frames[mcfg["name"]] = frame
        vlm_stats[mcfg["name"]] = stats

    primary_name = cfg["report"]["primary_model"]
    if primary_name not in vlm_frames:
        primary_name = available[0]["name"]
        log.info(
            f"primary model not available — using {primary_name!r} for "
            f"calibration/qualitative"
        )
    primary = vlm_frames[primary_name]

    # Baseline on the same rows (subset where the axTree resolved)
    bl = baseline_preds(cfg, bcfg, rows)
    common = rows.merge(bl, on=KEY, how="inner", validate="1:1")
    common_mask = rows.set_index(KEY).index.isin(common.set_index(KEY).index)

    # Floors: TRAIN mean/median of y, constant on every row (same recipe as
    # the baseline report)
    y_train = df.loc[df["split"] == "train", "y"].to_numpy()
    floors = {
        "train-mean floor": float(np.mean(y_train)),
        "train-median floor": float(np.median(y_train)),
    }

    # Head-to-head on identical rows: floors < LightGBM < VLM condition(s)
    y_c = common["y"].to_numpy()
    nav_c = common["is_navigation"].to_numpy()
    models = {name: np.full_like(y_c, const) for name, const in floors.items()}
    models["LightGBM (no image)"] = common["lgbm_pred_log"].to_numpy()
    for name, frame in vlm_frames.items():
        models[name] = frame.loc[common_mask, "vlm_pred_log"].to_numpy()
    head_lines, head_metrics = metrics_table(models, y_c, nav_c)

    # Task-level rollup on the same rows (RQ v2: whole-task Time-on-Task).
    # Only tasks with >= task_min_steps covered screens: singleton "tasks"
    # (scattered legacy cache rows) are per-screen rows in disguise.
    min_steps = scfg["task_min_steps"]
    task_sizes = common.groupby("trajectory_id")["y"].transform("size")
    t_mask = (task_sizes >= min_steps).to_numpy()
    task_lines, task_metrics = task_level_table(
        {name: pred[t_mask] for name, pred in models.items()},
        common.loc[t_mask, "trajectory_id"].to_numpy(),
        y_c[t_mask],
    )
    steps_per_task = common[t_mask].groupby("trajectory_id").size()
    n_singleton_tasks = int((common.groupby("trajectory_id").size() < min_steps).sum())

    # VLM conditions on the FULL eval set (rows the baseline couldn't cover)
    y_f = rows["y"].to_numpy()
    nav_f = rows["is_navigation"].to_numpy()
    full_lines, _ = metrics_table(
        {name: frame["vlm_pred_log"].to_numpy() for name, frame in vlm_frames.items()},
        y_f,
        nav_f,
    )

    # Calibration + qualitative for the primary condition
    cal, ece = calibration_table(
        y_f, primary["vlm_pred_log"].to_numpy(), n_bins=scfg["calibration_bins"]
    )

    qual_dir = Path(cfg["paths"]["qualitative_dir"])
    qual = save_qualitative(
        primary,
        cfg["qualitative"]["n"],
        cfg["qualitative"]["banner_height"],
        qual_dir,
        cfg["seed"],
    )

    paragraphs = [
        edge_paragraph(head_metrics[primary_name], head_metrics["LightGBM (no image)"])
    ]
    if "VLM (screen)" in head_metrics and "VLM (screen+task)" in head_metrics:
        paragraphs.append(
            goal_increment_paragraph(
                head_metrics["VLM (screen)"], head_metrics["VLM (screen+task)"]
            )
        )

    # Machine-readable outputs: the metrics JSON and the per-row prediction
    # matrix (head-to-head rows, one pred_log:<model> column per contestant)
    # — scripts/make_figures.py regenerates every paper figure from these
    # without re-running any model.
    payload = {
        "head_to_head_per_screen": head_metrics,
        "task_level": task_metrics,
        "parse_stats": vlm_stats,
        "calibration_ece_s": ece,
        "primary_model": primary_name,
        "pending_models": pending,
        "rows": {
            "eval_set": len(rows),
            "head_to_head": len(common),
            "tasks": int(len(steps_per_task)),
        },
    }
    metrics_json = Path(cfg["paths"]["metrics_json"])
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(payload, indent=2))

    pred_frame = common[KEY + ["y", "dwell_s", "is_navigation"]].reset_index(drop=True)
    for name, pred in models.items():
        pred_frame[f"pred_log:{name}"] = pred
    preds_out = Path(cfg["paths"]["predictions_parquet"])
    pred_frame.to_parquet(preds_out, compression="zstd", index=False)
    log.info(f"metrics JSON → {metrics_json} · prediction matrix → {preds_out}")

    lines = [
        "# Evaluation report — Time-on-Task on held-out TEST domains (O2)",
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
        f"- Eval set: **{len(rows)}** TEST rows with a resolved screenshot "
        f"({int(nav_f.sum())} navigation / {int((~nav_f).sum())} in-page)",
        f"- Head-to-head subset (axTree also resolved for LightGBM): "
        f"**{len(common)}** rows across **{len(steps_per_task)}** tasks "
        f"(median {int(steps_per_task.median())} covered steps/task)",
    ]
    for name, stats in vlm_stats.items():
        tc = stats["tier_counts"]
        lines.append(
            f"- {name}: parse tiers strict **{tc['strict']}** · labeled "
            f"**{tc['labeled']}** · bare number **{tc['bare_number']}** · "
            f"failed **{tc['fail']}** — **failure rate "
            f"{stats['parse_failure_rate']:.2%}**, failures imputed with "
            f"the {stats['fallback']}, never dropped"
        )
    first_stats = vlm_stats[primary_name]
    if first_stats["clipped_to"]:
        lines.append(
            f"- Parsed predictions clipped to {first_stats['clipped_to']} s "
            f"(the training-target range)"
        )
    if pending:
        lines.append(
            f"- ⏳ Pending conditions (no prediction cache yet, NOT in the "
            f"tables): {', '.join(pending)}"
        )
    lines += [
        "",
        "## Per-screen head-to-head (identical rows)",
        "",
        *head_lines,
        "",
        "## Task-level Time-on-Task (per-trajectory sums, identical rows)",
        "",
        "Per-screen predictions are summed within each task and scored "
        "against the summed actual — the data-driven analogue of KLM's "
        "operator-sum. Totals cover only evaluated steps, identically for "
        "targets and every model, so the comparison is apples-to-apples "
        "(covered-task time, not wall-clock task time). "
        f"Tasks with < {min_steps} covered screens are excluded "
        f"({n_singleton_tasks} such tasks — scattered legacy-cache rows, "
        "already counted in the per-screen tables).",
        "",
        *task_lines,
        "",
        "## VLM conditions on the full eval set",
        "",
        *full_lines,
        "",
        "## Where does the signal live? (O2 answer)",
        "",
        *[p + "\n" for p in paragraphs],
        f"## Calibration ({primary_name}, full eval set, decile bins by prediction)",
        "",
        f"ECE-style weighted gap: **{ece:.2f} s** "
        "(reliability plot: `scripts/make_figures.py` → fig_calibration)",
        "",
        "| bin | n | mean pred (s) | mean actual (s) | gap (s) |",
        "|---|---|---|---|---|",
        *[
            f"| {int(r.bin)} | {int(r.n)} | {r.mean_pred_s:.2f} "
            f"| {r.mean_actual_s:.2f} | {r.gap_s:.2f} |"
            for r in cal.itertuples()
        ],
        "",
        f"## Qualitative examples ({primary_name}, `{qual_dir}/`)",
        "",
        "| file | bucket | pred (s) | actual (s) | step | raw output |",
        "|---|---|---|---|---|---|",
        *[
            f"| {q['file']} | {q['bucket']} | {q['pred_s']} | {q['actual_s']} "
            f"| {'navigation' if q['is_navigation'] else 'in-page'} "
            f"| `{q['raw_output']}` |"
            for q in qual
        ],
        "",
        "## Full metrics (JSON)",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    report = Path(cfg["paths"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    log.info(f"report → {report} · qualitative → {qual_dir}/")
    log.info(
        "per-screen MAE(log): "
        + json.dumps({m: head_metrics[m]["overall"]["mae_log"] for m in head_metrics})
    )
    log.info(
        "task-level MAE(log): "
        + json.dumps({m: task_metrics[m]["mae_log"] for m in task_metrics})
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--predict-only",
        action="store_true",
        help="run the GPU decode stage(s) and stop",
    )
    group.add_argument(
        "--report-only", action="store_true", help="fail rather than run the GPU stage"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    vcfg = load_config(cfg["vlm_config"])
    bcfg = load_config(cfg["baseline_config"])
    np.random.seed(cfg["seed"])

    df = pd.read_parquet(cfg["paths"]["rows_final"])
    rows = eval_rows(df)
    log.info(f"eval set: {len(rows)} TEST rows with resolved screenshots")

    conditions = cfg["vlm_models"]
    if not args.report_only:
        for mcfg in conditions:
            if Path(mcfg["preds"]).exists():
                log.info(f"[{mcfg['name']}] using cached predictions: {mcfg['preds']}")
            elif adapters_available(mcfg["adapters"]):
                run_predict(cfg, vcfg, rows, winsor_cap_from_train(df), mcfg)
            else:
                log.info(
                    f"[{mcfg['name']}] no adapters yet at "
                    f"{mcfg['adapters']} — skipping (will be listed as "
                    f"pending)"
                )
    if args.predict_only:
        return

    available = [m for m in conditions if Path(m["preds"]).exists()]
    pending = [m["name"] for m in conditions if not Path(m["preds"]).exists()]
    if not available:
        sys.exit(
            "No prediction caches exist for any configured VLM "
            "condition — run the PREDICT stage on a GPU machine first."
        )

    run_report(cfg, bcfg, df, rows, available, pending, args.config)


if __name__ == "__main__":
    main()
