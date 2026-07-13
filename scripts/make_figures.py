"""The paper's figure set, regenerated from cached artifacts on any CPU box.

    uv run python scripts/make_figures.py [--config configs/figures.yaml]

Reads ONLY what earlier stages wrote — never runs a model, never fetches,
never touches the external set. Figures with missing inputs are SKIPPED and
listed in figures_card.md. Each model keeps one fixed color everywhere."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from totvlm.config import load_config
from totvlm.scoring import calibration_table, task_totals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# One visual system (fixed identities — color follows the entity)

INK = "#0b0b0b"  # primary text
INK_2 = "#52514e"  # secondary text (axis labels)
MUTED = "#898781"  # tick labels, annotations
GRID = "#e1e0d9"  # hairline gridlines
BASELINE = "#c3c2b7"  # axis lines / identity reference

MODEL_COLORS = {  # the paper's ladder, one color per model everywhere
    "train-mean floor": "#c3c2b7",
    "train-median floor": "#898781",
    "LightGBM (no image)": "#1baf7a",
    "VLM (screen, distilled)": "#2a78d6",
    "VLM (screen+task, distilled)": "#4a3aa7",
}
SPLIT_COLORS = {"train": "#2a78d6", "val": "#1baf7a", "test": "#eda100"}
SUBSETS = ("overall", "navigation", "in_page")
SUBSET_LABELS = {"overall": "overall", "navigation": "navigation", "in_page": "in-page"}
PRED_PREFIX = "pred_log:"


def model_order(names) -> list[str]:
    """Ladder order (floors → LightGBM → VLMs); unknown names keep file order."""
    ladder = list(MODEL_COLORS)
    return sorted(names, key=lambda n: ladder.index(n) if n in ladder else len(ladder))


def color_of(name: str) -> str:
    return MODEL_COLORS.get(name, "#e34948")


def apply_style(font_size: int) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def tidy(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


# Figure builders (each returns the written path)


def fig_dwell_distribution(rows: pd.DataFrame, cfg: dict, dest: Path) -> Path:
    """Per-split raw dwell histograms (log-x) with the train-p95 winsor cap —
    the label the whole project regresses onto."""
    cap = float(rows.loc[rows["split"] == "train", "dwell_s"].max())
    bins = np.geomspace(
        rows["dwell_s_raw"].min(),
        rows["dwell_s_raw"].max(),
        cfg["dwell_distribution"]["bins"],
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=cfg["style"]["dpi"])
    for split in ("train", "val", "test"):
        d = rows.loc[rows["split"] == split, "dwell_s_raw"]
        ax.hist(
            d,
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=SPLIT_COLORS[split],
            label=f"{split} (n={len(d):,})",
        )
    ax.axvline(cap, color=INK_2, lw=1, ls="--")
    ax.text(
        cap * 1.12,
        ax.get_ylim()[1] * 0.92,
        f"winsor cap {cap:.1f} s\n(train p95)",
        color=INK_2,
        fontsize=8,
        va="top",
    )
    ax.set_xscale("log")
    ax.set_xlabel("per-screen dwell (s, log scale)")
    ax.set_ylabel("density")
    ax.set_title("Per-screen dwell time by domain-disjoint split")
    ax.legend(loc="upper left")
    tidy(ax)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def _bar_panel(ax, names: list[str], values: list[float], fmt: str) -> None:
    """Horizontal bars, ladder order top→bottom, value labels at the tip."""
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, values, height=0.62, color=[color_of(n) for n in names])
    span = max(abs(v) for v in values) or 1.0
    for y, v in zip(ypos, values):
        ax.text(
            v + 0.02 * span, y, format(v, fmt), va="center", color=INK_2, fontsize=8
        )
    ax.set_yticks(ypos, names)
    ax.set_xlim(0, span * 1.18)
    ax.grid(axis="y", visible=False)
    tidy(ax)


def fig_head_to_head(metrics: dict, cfg: dict, dest: Path) -> Path:
    """Model × subset small multiples: MAE(log) (lower better) and Spearman ρ
    (higher better) on identical TEST rows."""
    per_screen = metrics["head_to_head_per_screen"]
    names = model_order(per_screen)
    fig, axes = plt.subplots(
        2, len(SUBSETS), figsize=(10.5, 4.6), dpi=cfg["style"]["dpi"], sharey="row"
    )
    for j, subset in enumerate(SUBSETS):
        mae = [per_screen[n][subset]["mae_log"] for n in names]
        rho = [np.nan_to_num(per_screen[n][subset]["spearman_rho"]) for n in names]
        n_rows = per_screen[names[0]][subset]["n"]
        _bar_panel(axes[0, j], names, mae, ".3f")
        _bar_panel(axes[1, j], names, rho, ".3f")
        axes[0, j].set_title(f"{SUBSET_LABELS[subset]} (n={n_rows:,})", fontsize=9)
        axes[1, j].set_xlabel("Spearman ρ  (higher better)")
        if j:
            axes[0, j].set_yticklabels([])
            axes[1, j].set_yticklabels([])
    axes[0, 0].set_xlabel("")
    fig.suptitle(
        "Per-screen head-to-head on identical held-out TEST rows — "
        "MAE(log) top (lower better), rank agreement bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(dest)
    plt.close(fig)
    return dest


def fig_task_level(metrics: dict, cfg: dict, dest: Path) -> Path:
    """Task-level rollup: per-trajectory summed seconds vs summed actual."""
    task = metrics["task_level"]
    names = model_order(task)
    n_tasks = task[names[0]]["n"]
    fig, axes = plt.subplots(
        1, 2, figsize=(9.4, 2.9), dpi=cfg["style"]["dpi"], sharey=True
    )
    _bar_panel(axes[0], names, [task[n]["mae_log"] for n in names], ".3f")
    _bar_panel(
        axes[1], names, [np.nan_to_num(task[n]["spearman_rho"]) for n in names], ".3f"
    )
    axes[0].set_xlabel("MAE(log)  (lower better)")
    axes[1].set_xlabel("Spearman ρ  (higher better)")
    axes[1].set_yticklabels([])
    fig.suptitle(
        f"Task-level Time-on-Task (per-trajectory sums, n={n_tasks:,} tasks)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(dest)
    plt.close(fig)
    return dest


def _learned_models(preds: pd.DataFrame) -> list[str]:
    """Prediction columns minus the constant floors (flat scatters say
    nothing)."""
    names = [c[len(PRED_PREFIX) :] for c in preds.columns if c.startswith(PRED_PREFIX)]
    return model_order(n for n in names if "floor" not in n)


def _scatter_panels(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: dict,
    dest: Path,
    *,
    unit: str,
    title: str,
) -> Path:
    """One actual-vs-predicted panel per model, log-log, shared limits."""
    scfg = cfg["scatter"]
    rng = np.random.default_rng(cfg["seed"])
    eps = 1e-2  # log axes: clamp near-zero seconds
    pairs = {n: (np.maximum(a, eps), np.maximum(p, eps)) for n, (a, p) in pairs.items()}
    names = list(pairs)
    lo = max(min(min(a.min(), p.min()) for a, p in pairs.values()) * 0.8, eps)
    hi = max(max(a.max(), p.max()) for a, p in pairs.values()) * 1.25
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(3.3 * len(names) + 0.4, 3.5),
        dpi=cfg["style"]["dpi"],
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        actual, pred = pairs[name]
        rho = float(pd.Series(actual).corr(pd.Series(pred), method="spearman"))
        if len(actual) > scfg["max_points"]:
            idx = rng.choice(len(actual), scfg["max_points"], replace=False)
            actual, pred = actual[idx], pred[idx]
        ax.plot([lo, hi], [lo, hi], color=BASELINE, lw=1, ls="--", zorder=1)
        ax.scatter(
            actual,
            pred,
            s=scfg["point_size"],
            alpha=scfg["alpha"],
            color=color_of(name),
            edgecolors="none",
            zorder=2,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{name}\nρ = {rho:.3f}", fontsize=9)
        ax.set_xlabel(f"actual {unit}")
        tidy(ax)
    axes[0].set_ylabel(f"predicted {unit}")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(dest)
    plt.close(fig)
    return dest


def fig_screen_scatter(preds: pd.DataFrame, cfg: dict, dest: Path) -> Path:
    actual = np.expm1(preds["y"].to_numpy())
    pairs = {
        name: (actual, np.expm1(preds[PRED_PREFIX + name].to_numpy()))
        for name in _learned_models(preds)
    }
    return _scatter_panels(
        pairs,
        cfg,
        dest,
        unit="dwell (s)",
        title=f"Per-screen predictions on identical TEST rows "
        f"(n={len(preds):,}; dashed = perfect)",
    )


def fig_task_scatter(
    preds: pd.DataFrame, cfg: dict, min_steps: int, dest: Path
) -> Path:
    """Summed within-trajectory seconds — the whole-task claim, per model."""
    sizes = preds.groupby("trajectory_id")["y"].transform("size")
    kept = preds[sizes >= min_steps]
    traj = kept["trajectory_id"].to_numpy()
    _, y_task = task_totals(traj, kept["y"].to_numpy())
    pairs = {}
    for name in _learned_models(preds):
        _, p_task = task_totals(traj, kept[PRED_PREFIX + name].to_numpy())
        pairs[name] = (np.expm1(y_task), np.expm1(p_task))
    return _scatter_panels(
        pairs,
        cfg,
        dest,
        unit="task time (s)",
        title=f"Task-level Time-on-Task: per-trajectory sums "
        f"(n={len(y_task):,} tasks with ≥{min_steps} covered screens)",
    )


def fig_calibration(preds: pd.DataFrame, cfg: dict, dest: Path) -> Path:
    """Decile reliability curves for every learned model, one axes —
    the honest 'can you trust the seconds' picture."""
    fig, ax = plt.subplots(figsize=(5.4, 4.9), dpi=cfg["style"]["dpi"])
    y = preds["y"].to_numpy()
    lim = 0.0
    for name in _learned_models(preds):
        cal, ece = calibration_table(
            y, preds[PRED_PREFIX + name].to_numpy(), n_bins=cfg["calibration"]["n_bins"]
        )
        c = color_of(name)
        ax.plot(
            cal["mean_pred_s"],
            cal["mean_actual_s"],
            color=c,
            lw=2,
            marker="o",
            ms=4.5,
            mec="white",
            mew=1,
            label=f"{name} (ECE {ece:.2f} s)",
        )
        lim = max(lim, cal["mean_pred_s"].max(), cal["mean_actual_s"].max())
    lim *= 1.06
    ax.plot([0, lim], [0, lim], color=BASELINE, lw=1, ls="--", zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mean predicted dwell (s), decile bins by prediction")
    ax.set_ylabel("mean actual dwell (s)")
    ax.set_title("Calibration on identical TEST rows (dashed = perfect)")
    ax.legend(loc="upper left", fontsize=8)
    tidy(ax)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def fig_feature_importance(booster_path: Path, cfg: dict, dest: Path) -> Path:
    """What the no-image bar leans on — context for the VLM's edge."""
    import lightgbm as lgb

    from totvlm.model import lgbm_feature_importances

    booster = lgb.Booster(model_file=str(booster_path))
    imp = lgbm_feature_importances(booster, booster.feature_name())
    top = imp[: cfg["feature_importance"]["top_n"]]
    fig, ax = plt.subplots(figsize=(5.8, 0.3 * len(top) + 1.3), dpi=cfg["style"]["dpi"])
    ypos = np.arange(len(top))[::-1]
    ax.barh(
        ypos,
        [t["pct"] for t in top],
        height=0.62,
        color=MODEL_COLORS["LightGBM (no image)"],
    )
    for y, t in zip(ypos, top):
        ax.text(
            t["pct"] + 0.4, y, f"{t['pct']:.1f}%", va="center", color=INK_2, fontsize=8
        )
    ax.set_yticks(ypos, [t["feature"] for t in top])
    ax.set_xlim(0, max(t["pct"] for t in top) * 1.16)
    ax.set_xlabel("share of total gain (%)")
    ax.set_title(f"LightGBM baseline — top {len(top)} features by gain")
    ax.grid(axis="y", visible=False)
    tidy(ax)
    fig.savefig(dest)
    plt.close(fig)
    return dest


# Orchestration: build what has inputs, card what doesn't


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/figures.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg = load_config(cfg["eval_config"])
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    out = paths["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    apply_style(cfg["style"]["font_size"])

    rows = (
        pd.read_parquet(paths["rows_final"]) if paths["rows_final"].exists() else None
    )
    preds = (
        pd.read_parquet(paths["eval_predictions"])
        if paths["eval_predictions"].exists()
        else None
    )
    metrics = (
        json.loads(paths["eval_metrics"].read_text())
        if paths["eval_metrics"].exists()
        else None
    )

    jobs = [
        (
            "fig_dwell_distribution",
            "rows_final",
            rows is not None and rows["split"].notna().any(),
            lambda d: fig_dwell_distribution(rows[rows["split"].notna()], cfg, d),
        ),
        (
            "fig_head_to_head",
            "eval_metrics",
            metrics is not None,
            lambda d: fig_head_to_head(metrics, cfg, d),
        ),
        (
            "fig_task_level",
            "eval_metrics",
            metrics is not None,
            lambda d: fig_task_level(metrics, cfg, d),
        ),
        (
            "fig_screen_scatter",
            "eval_predictions",
            preds is not None,
            lambda d: fig_screen_scatter(preds, cfg, d),
        ),
        (
            "fig_task_scatter",
            "eval_predictions",
            preds is not None,
            lambda d: fig_task_scatter(
                preds, cfg, ecfg["scoring"]["task_min_steps"], d
            ),
        ),
        (
            "fig_calibration",
            "eval_predictions",
            preds is not None,
            lambda d: fig_calibration(preds, cfg, d),
        ),
        (
            "fig_feature_importance",
            "baseline_model",
            paths["baseline_model"].exists(),
            lambda d: fig_feature_importance(paths["baseline_model"], cfg, d),
        ),
    ]

    made, skipped = [], []
    for name, needs, ready, build in jobs:
        if not ready:
            skipped.append((name, needs))
            log.info(f"SKIP {name} — missing input: {needs}")
            continue
        dest = build(out / f"{name}.png")
        made.append(name)
        log.info(f"wrote {dest}")

    lines = [
        "# Figures card",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"config `{args.config}` · seed {cfg['seed']}_",
        "",
        "Regenerable anytime: `uv run python scripts/make_figures.py` "
        "(CPU-only, reads cached artifacts, never runs a model).",
        "",
        "| figure | status | input |",
        "|---|---|---|",
        *[f"| {n} | ✅ written | — |" for n in made],
        *[f"| {n} | ⏳ skipped | `{needs}` not present yet |" for n, needs in skipped],
        "",
        "Fixed model colors across every figure: "
        + " · ".join(f"{n} `{c}`" for n, c in MODEL_COLORS.items()),
        "",
    ]
    paths["card"].write_text("\n".join(lines))
    log.info(
        f"{len(made)} figures → {out}/ · card → {paths['card']}"
        + (f" · {len(skipped)} pending earlier stages" if skipped else "")
    )


if __name__ == "__main__":
    main()
