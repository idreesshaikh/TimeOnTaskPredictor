"""Smart λ search — bracket-and-refine on VALIDATION, instead of the full grid.

    uv run python scripts/run_sweep.py

VAL error as a function of λ is assumed unimodal: it improves as the teacher
blend grows, bottoms out, then worsens — the standard distillation trade-off.
Under that assumption the full 8-point grid is wasteful. Instead:

1. Train the coarse anchors (`sweep.coarse` in configs/sweeps/_base.yaml).
   λ=0 is one of them — it is the "does distillation help at all?" ablation
   reference, so it must be trained no matter what.
2. Refine: while the current best-VAL λ has an untrained neighbour on the
   grid, train that neighbour. The search stops the moment both neighbours of
   the best λ are confirmed worse — i.e. as soon as we see performance start
   to degrade on either side, no further runs are spent there.

Typical cost 4-6 runs; worst case = the full grid (never more than before).
Finished runs (output_dir/final exists) are never retrained and interrupted
ones auto-resume from their checkpoints, so this stays idempotent + resumable
like the rest of the pipeline. Selection reads VALIDATION cards only — TEST is
never touched. scripts/select_lambda.py still makes (and writes) the final
pick over every trained card afterwards.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_lambda import _summarise  # noqa: E402  (best VAL MAE from a card)

SWEEPS_DIR = Path(__file__).resolve().parent.parent / "configs" / "sweeps"


def _lam_key(x) -> float:
    """Normalise a λ for use as a dict key (guards float-repr drift)."""
    return round(float(x), 6)


def load_runs(sweeps_dir: Path = SWEEPS_DIR) -> tuple[list[float], dict, list[float]]:
    """Read the sweep overlays. Returns (sorted grid of λ values,
    {λ: {config, output_dir, card}}, coarse anchor λ list from _base.yaml)."""
    runs: dict[float, dict] = {}
    for cfg_path in sorted(sweeps_dir.glob("lam*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        lam = _lam_key(cfg["lupi"]["lambda"])
        runs[lam] = {
            "config": cfg_path,
            "output_dir": Path(cfg["paths"]["output_dir"]),
            "card": Path(cfg["paths"]["train_card"]),
        }
    base = yaml.safe_load((sweeps_dir / "_base.yaml").read_text())
    coarse = [_lam_key(x) for x in base["sweep"]["coarse"]]
    grid = sorted(runs)
    missing = [x for x in coarse if x not in runs]
    if missing:
        raise SystemExit(f"sweep.coarse anchors {missing} have no lam*.yaml overlay")
    return grid, runs, coarse


def next_lams(grid: list[float], scores: dict[float, float],
              coarse: list[float]) -> list[float]:
    """The pure decision rule: which λ values to train next.

    Coarse anchors first; once all are scored, only the untrained grid
    neighbours of the current best-VAL λ. Empty list ⇒ both neighbours of the
    best λ are already confirmed worse ⇒ stop (unimodality assumption)."""
    todo = [x for x in coarse if x not in scores]
    if todo:
        return todo
    best = min(scores, key=scores.get)
    i = grid.index(best)
    return [grid[j] for j in (i - 1, i + 1)
            if 0 <= j < len(grid) and grid[j] not in scores]


def _score(run: dict) -> float:
    """Best VAL decode MAE(log) from a finished run's train card."""
    row = _summarise(str(run["card"]))
    if row is None:
        raise SystemExit(
            f"{run['card']}: no completed eval pass — the run under "
            f"{run['output_dir']} finished without a usable VAL decode; "
            f"inspect its log before continuing the sweep.")
    return row["best_val_mae_log"]


def _train(run: dict) -> None:
    print(f"--- training {run['config']} "
          f"(auto-resumes from {run['output_dir']} if checkpoints exist)",
          flush=True)
    subprocess.run(
        [sys.executable, "-m", "totvlm.train", "--config", str(run["config"])],
        check=True)


def main() -> None:
    grid, runs, coarse = load_runs()
    print(f"λ grid: {grid}   coarse anchors: {coarse}", flush=True)

    # Seed with anything already trained (e.g. from an earlier submission).
    scores = {lam: _score(run) for lam, run in runs.items()
              if (run["output_dir"] / "final").is_dir()}
    if scores:
        print(f"already trained: { {k: round(v, 4) for k, v in scores.items()} }",
              flush=True)

    while todo := next_lams(grid, scores, coarse):
        stage = "coarse" if any(x in coarse for x in todo) else "refine"
        print(f"=== {stage} stage → λ {todo}", flush=True)
        for lam in todo:
            run = runs[lam]
            if not (run["output_dir"] / "final").is_dir():
                _train(run)
            scores[lam] = _score(run)
            print(f"λ={lam}: best VAL MAE(log) = {scores[lam]:.4f}", flush=True)

    best = min(scores, key=scores.get)
    skipped = [x for x in grid if x not in scores]
    print(f"\nSearch done after {len(scores)}/{len(grid)} runs "
          f"(skipped λ {skipped or 'none'} — both neighbours of the best λ "
          f"were worse).")
    print(f"Best on VALIDATION: λ={best} "
          f"(MAE(log) {scores[best]:.4f}). Run select_lambda.py --write to "
          f"record it in configs/vlm.yaml.")


if __name__ == "__main__":
    main()
