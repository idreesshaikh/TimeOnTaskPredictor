"""Clear stale FINAL checkpoints + prediction caches when λ changed.

    uv run python scripts/clear_stale_finals.py [--dry-run]

Run by the gate job (scripts/finalize_select.sbatch) AFTER select_lambda.py
has written the sweep winner into configs/vlm.yaml. train.sbatch skips a
condition whose `final/` exists and reuses TEST-prediction caches — correct
for resuming, catastrophic after λ changed: it would evaluate models trained
at the OLD λ. This script compares the λ now in configs/vlm.yaml against the
λ each existing checkpoint dir was trained with (recorded in its train card)
and, on mismatch, removes exactly the artifacts train.sbatch gates on:

  - both VLM checkpoint dirs (whole dir — a leftover mid-run checkpoint at
    the old λ would otherwise be auto-resumed into the new run)
  - the per-condition TEST prediction caches (configs/eval.yaml)
  - the external report/preds/scatter + AIM report (configs/external.yaml),
    so the one-shot external check re-runs for the new final model

Matching λ (or nothing trained yet) → no-op. Every action is printed.
All paths come from the configs — nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from totvlm.config import load_config

_JSON_BLOCK = re.compile(r"```json\s*(\{.*\})\s*```", re.DOTALL)


def _card_lambda(card: Path) -> float | None:
    """The λ a finished run recorded in its train card, or None."""
    if not card.is_file():
        return None
    m = _JSON_BLOCK.search(card.read_text())
    if not m:
        return None
    lupi = json.loads(m.group(1)).get("effective_config", {}).get("lupi")
    if not isinstance(lupi, dict) or lupi.get("lambda") is None:
        return None
    return float(lupi["lambda"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be cleared, delete nothing")
    args = ap.parse_args()

    winner = float(load_config("configs/vlm.yaml")["lupi"]["lambda"])
    print(f"configs/vlm.yaml now holds λ = {winner}")

    stale_reasons: list[str] = []
    for cfg_path in ("configs/vlm.yaml", "configs/vlm_task.yaml"):
        cfg = load_config(cfg_path)
        out = Path(cfg["paths"]["output_dir"])
        if not out.is_dir() or not any(out.iterdir()):
            print(f"  {out}: nothing trained — ok")
            continue
        prev = _card_lambda(Path(cfg["paths"]["train_card"]))
        if prev is not None and abs(prev - winner) < 1e-9:
            print(f"  {out}: trained at λ={prev} — matches, keeping")
        else:
            why = ("no readable train card (interrupted run at unknown λ)"
                   if prev is None else f"trained at λ={prev}")
            stale_reasons.append(f"{out}: {why}")

    if not stale_reasons:
        print("Nothing stale — train.sbatch may resume/skip safely.")
        return

    print("STALE — the existing finals do not match the selected λ:")
    for r in stale_reasons:
        print(f"  {r}")

    # everything train.sbatch gates on, straight from the configs
    eval_cfg = load_config("configs/eval.yaml")
    ext_cfg = load_config("configs/external.yaml")
    dirs = [Path(load_config(p)["paths"]["output_dir"])
            for p in ("configs/vlm.yaml", "configs/vlm_task.yaml")]
    files = [Path(m["preds"]) for m in eval_cfg["vlm_models"]]
    files += [Path(ext_cfg["paths"][k]) for k in ("report", "preds",
                                                  "scatter_png")]
    files += [Path(ext_cfg["aim"]["report"])]

    verb = "would remove" if args.dry_run else "removing"
    for d in dirs:
        if d.is_dir():
            print(f"  {verb} dir  {d}")
            if not args.dry_run:
                shutil.rmtree(d)
    for f in files:
        if f.is_file():
            print(f"  {verb} file {f}")
            if not args.dry_run:
                f.unlink()
    if args.dry_run:
        print("(dry run — nothing deleted)")
    else:
        print(f"Cleared. train.sbatch will now retrain both conditions at "
              f"λ={winner} and re-run TEST + external.")


if __name__ == "__main__":
    main()
