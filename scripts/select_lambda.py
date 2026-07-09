"""Pick the distillation weight λ from the sweep — on VALIDATION only.

    uv run python scripts/select_lambda.py [artifacts_lam50/sweeps/*_card.md]

Each sweep overlay (configs/sweeps/lam*.yaml) trains the image+features model
with one global λ and writes a train card. This reads those cards, and for each
reports the BEST validation decode MAE(log) over the run's eval passes — i.e.
the metric the deployed (best) checkpoint achieves. It ranks the λ values and
names the winner. With --write it also patches `lupi.lambda` in configs/vlm.yaml
to that winner (comments preserved), so the final two-condition run needs no
manual edit; vlm_task.yaml inherits the value via `base: vlm.yaml`.

This never opens a prediction cache or the TEST split: selection is a pure
VAL-set decision, so the evaluate-once discipline for TEST stays intact."""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _load_card(path: str) -> dict:
    m = _JSON_BLOCK.search(open(path).read())
    if not m:
        raise SystemExit(f"{path}: no ```json block — is this a train card?")
    return json.loads(m.group(1))


def _summarise(path: str) -> dict | None:
    """(λ, VAL MAE(log), best VAL eval_loss) for one sweep card, or None if
    the run produced nothing scoreable yet.

    Preferred source: `<output_dir>/val_full_metrics.json` written by
    scripts/decode_val.py — the deployed checkpoint decoded on the FULL val
    split. Fallback (with a warning): the best per-pass card decode, which
    uses only eval.decode_samples rows — too few to separate nearby λ values."""
    card = _load_card(path)
    lam = card.get("effective_config", {}).get("lupi", {})
    lam = lam.get("lambda") if isinstance(lam, dict) else lam
    losses = [e["eval_loss"] for e in card.get("eval_losses", [])
              if e.get("eval_loss") is not None]

    full_path = Path(path[: -len("_card.md")]) / "val_full_metrics.json"
    if full_path.is_file():
        full = json.loads(full_path.read_text())
        overall = full["metrics"]["overall"]
        return {
            "card": path,
            "lambda": full.get("lambda", lam),
            "best_val_mae_log": overall["mae_log"],
            "best_val_eval_loss": min(losses) if losses else float("nan"),
            "source": f"FULL val n={overall['n']}",
        }

    maes = [
        (h["decoded_metrics"]["mae_log"], h["decoded_metrics"]["n"])
        for h in card.get("decode_history", [])
        if h.get("decoded_metrics", {}).get("n")
    ]
    if not maes:
        return None
    best_mae, n = min(maes)
    return {
        "card": path,
        "lambda": lam,
        "best_val_mae_log": best_mae,
        "best_val_eval_loss": min(losses) if losses else float("nan"),
        "source": f"card decodes n={n} (NOISY — run scripts/decode_val.py)",
    }


def write_lambda(target: str, lam) -> None:
    """Patch the single `lupi.lambda:` scalar in `target` (a vlm config) to
    `lam`, preserving the line's inline comment and the rest of the file."""
    path = Path(target)
    text = path.read_text()
    new, n = re.subn(
        r"(?m)^(\s*lambda:\s*)[-\d.]+(.*)$",
        lambda m: f"{m.group(1)}{lam}{m.group(2)}", text, count=1,
    )
    if n == 0:
        raise SystemExit(f"no scalar 'lambda:' line found in {target} to update")
    path.write_text(new)
    print(f"wrote  lupi.lambda: {lam}  → {target}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cards", nargs="*",
                    default=sorted(glob.glob("artifacts_lam50/sweeps/*_card.md")),
                    help="sweep train cards (default: artifacts_lam50/sweeps/*_card.md)")
    ap.add_argument("--write", action="store_true",
                    help="patch lupi.lambda in --target to the winner")
    ap.add_argument("--target", default="configs/vlm.yaml",
                    help="config to patch with --write (default: configs/vlm.yaml)")
    args = ap.parse_args()
    if not args.cards:
        sys.exit("no sweep cards found — train configs/sweeps/lam*.yaml first")

    rows = [r for c in args.cards if (r := _summarise(c))]
    if not rows:
        sys.exit("no sweep card has a completed eval pass yet")
    rows.sort(key=lambda r: r["best_val_mae_log"])

    print(f"{'λ':>6}  {'VAL MAE(log)':>13}  {'best VAL eval_loss':>18}  "
          f"{'source':>14}  card")
    for i, r in enumerate(rows):
        mark = "  ← winner" if i == 0 else ""
        print(f"{r['lambda']:>6}  {r['best_val_mae_log']:>13.4f}  "
              f"{r['best_val_eval_loss']:>18.4f}  {r['source']:>14}  "
              f"{r['card']}{mark}")

    noisy = [r for r in rows if "NOISY" in r["source"]]
    if noisy:
        print(f"\nWARNING: {len(noisy)}/{len(rows)} λ values are ranked on "
              f"tiny card decodes, not the full val split — run "
              f"scripts/decode_val.py on them before trusting this ranking.")

    winner = rows[0]["lambda"]
    if args.write and noisy:
        sys.exit("\nREFUSING --write: the ranking mixes full-val and tiny "
                 "card-decode numbers, so the winner may be noise. Run "
                 "scripts/decode_val.py for every λ above, then re-run.")
    if args.write:
        write_lambda(args.target, winner)
        print(f"\nSelected on VALIDATION: lupi.lambda = {winner} "
              f"(written to {args.target}). Now run the final two conditions.")
    else:
        print(f"\nSelected on VALIDATION: set  lupi.lambda: {winner}  "
              f"in configs/vlm.yaml (or re-run with --write), then run the "
              f"final two conditions.")


if __name__ == "__main__":
    main()