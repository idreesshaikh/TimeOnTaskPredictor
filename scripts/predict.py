"""
predict.py — play with the trained model on YOUR screenshots.
=============================================================
The manual-inference playground: point it at any screenshot (a booking
site's seat picker, your own redesign mockup, anything) and it prints the
predicted dwell seconds. This is the designer use case in miniature —
compare two designs of the same screen before any user ever sees them.

    # screen-only condition (what does the UI alone predict?)
    uv run python scripts/predict.py my_screen.png

    # screen+task condition (the user's goal is known)
    uv run python scripts/predict.py my_screen.png --task "Pick a window seat"

    # compare designs A and B for the same goal
    uv run python scripts/predict.py seat_v1.png seat_v2.png --task "Pick a seat"

Needs a CUDA GPU + `uv sync --extra vlm` (same as training) and a trained
adapter. --task switches the default config/adapter to the screen+task
condition; --checkpoint may point at any adapter dir or a base model id
(base model = zero-shot, expect rough numbers).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from totvlm.config import load_config
from totvlm.data import build_inference_examples, parse_dwell_output_lenient
from totvlm.model import load_vlm_for_inference, predict_dwell_batch


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Predict dwell seconds for screenshots."
    )
    ap.add_argument("images", nargs="+", help="screenshot file(s)")
    ap.add_argument("--task", default=None,
                    help="the user's goal, e.g. 'Book a window seat' "
                         "(switches to the screen+task model)")
    ap.add_argument("--config", default=None,
                    help="default: configs/vlm.yaml, or configs/vlm_task.yaml "
                         "when --task is given")
    ap.add_argument("--checkpoint", default=None,
                    help="adapter dir (default: <output_dir>/final from the "
                         "config) or a base model id for zero-shot")
    args = ap.parse_args()

    cfg = load_config(
        args.config
        or ("configs/vlm_task.yaml" if args.task else "configs/vlm.yaml")
    )
    if cfg["data"].get("include_features"):
        raise SystemExit(
            "feature-input conditions (configs/vlm_feat*.yaml) need the "
            "screen's axTree ui-stats in the prompt — arbitrary screenshots "
            "have none, so predict.py cannot serve them; use a "
            "screenshot-only condition's config."
        )
    checkpoint = args.checkpoint or f"{cfg['paths']['output_dir']}/final"
    if not args.checkpoint and not Path(checkpoint).exists():
        raise SystemExit(
            f"No trained adapter at {checkpoint} — train first "
            f"(python -m totvlm.train --config ...) or pass --checkpoint."
        )

    missing = [p for p in args.images if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"screenshot not found: {', '.join(missing)}")

    print(f"model: {checkpoint} · task: {args.task or '(screen-only)'}")
    model, processor = load_vlm_for_inference(
        checkpoint, max_seq_length=cfg["model"]["max_seq_length"]
    )
    examples = build_inference_examples(
        args.images,
        max_side=cfg["image"]["max_side"],
        min_pixels=cfg["image"]["min_pixels"],
        max_pixels=cfg["image"]["max_pixels"],
        task_title=args.task,
        scaffold=bool(cfg["data"].get("scaffold")),
    )
    outputs = predict_dwell_batch(
        model, processor, examples,
        batch_size=cfg["train"]["eval_batch_size"],
        max_new_tokens=cfg["eval"]["max_new_tokens"],
    )
    for path, raw in zip(args.images, outputs):
        seconds, tier = parse_dwell_output_lenient(raw)
        pretty = f"{seconds:.1f} s" if seconds is not None else f"UNPARSED ({raw!r})"
        print(f"{path}  →  {pretty}" + (f"  [{tier}]" if tier != "strict" else ""))


if __name__ == "__main__":
    main()
