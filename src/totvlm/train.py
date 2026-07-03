"""
totvlm/train.py
===============
QLoRA SFT of Qwen3-VL-4B-Instruct for per-screen dwell prediction — Path A:
the model learns to EMIT the number as text (`dwell_seconds: X.X`).

Scaffold = Unsloth's official Qwen3-VL vision SFT pipeline: FastVisionModel +
TRL SFTTrainer + UnslothVisionDataCollator. The collator owns vision-token
packing and the chat template — nothing is hand-rolled here.

Run (needs a CUDA GPU — install extras first: `uv sync --extra vlm`):
    uv run python -m totvlm.train --config configs/vlm.yaml

Dry run (memory smoke test: SAME batch/grad-accum/pixels, 50 examples, few
steps; confirms loss drops and outputs parse — ALWAYS run before a full run):
    uv run python -m totvlm.train --config configs/vlm.yaml --dry-run

Outputs: checkpoints → artifacts/vlm_ckpt/ (per epoch + final adapters),
train/val loss + parse rate → wandb + stdout, stage card →
artifacts/vlm_train_card.md. All knobs live in configs/vlm.yaml.
"""
from __future__ import annotations

# unsloth MUST be imported before transformers/trl/peft so its
# performance/compat patches apply to them.
# isort: off
try:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
except ImportError as e:                                  # pragma: no cover
    raise SystemExit(
        "unsloth is not installed — totvlm.train needs it. "
        "Install with: uv sync --extra vlm   (Linux + CUDA)"
    ) from e
# isort: on

import argparse
import dataclasses
import inspect
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from totvlm.config import load_config
from totvlm.data import build_vlm_examples, parse_dwell_output
from totvlm.model import load_vlm
from totvlm.scoring import regression_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Version-tolerant TRL shims ────────────────────────────────────────────────
# The installed unsloth pin decides the TRL version; two arg names moved
# across TRL releases. Map them instead of pinning the whole stack.

def _make_sft_config(**kwargs) -> SFTConfig:
    names = {f.name for f in dataclasses.fields(SFTConfig)}
    if "max_seq_length" not in names and "max_length" in names:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    if "eval_strategy" not in names and "evaluation_strategy" in names:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return SFTConfig(**kwargs)


def _make_trainer(model, processor, **kwargs) -> SFTTrainer:
    params = inspect.signature(SFTTrainer.__init__).parameters
    key = "processing_class" if "processing_class" in params else "tokenizer"
    return SFTTrainer(model=model, **{key: processor}, **kwargs)


# ── Per-epoch VAL decode hook ─────────────────────────────────────────────────

class ValDecodeCallback(TrainerCallback):
    """
    After each evaluation pass: greedy-decode a seeded sample of val screens,
    report parse-success rate + log/seconds metrics on the parsed subset,
    and keep a couple of raw decodes for the run card.
    """

    def __init__(self, processor, examples, n_samples, max_new_tokens, seed):
        rng = random.Random(seed)
        self.samples = rng.sample(examples, min(n_samples, len(examples)))
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.history: list[dict] = []       # one entry per eval pass

    def _decode_one(self, model, messages) -> str:
        inputs = self.processor.apply_chat_template(
            messages[:-1],                  # drop the gold assistant turn
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,            # greedy: parseability, not variety
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
        FastVisionModel.for_inference(model)
        decoded: list[dict] = []
        try:
            for ex in self.samples:
                text = self._decode_one(model, ex["messages"])
                decoded.append({
                    "output": text,
                    "gold_s": ex["dwell_s"],
                    "pred_s": parse_dwell_output(text),
                })
        finally:
            FastVisionModel.for_training(model)

        parsed = [d for d in decoded if d["pred_s"] is not None]
        parse_rate = len(parsed) / max(len(decoded), 1)
        m = regression_metrics(
            np.log1p([d["gold_s"] for d in parsed]),
            np.log1p([d["pred_s"] for d in parsed]),
        ) if parsed else {"n": 0}

        entry = {
            "epoch": round(float(state.epoch or 0.0), 2),
            "step": state.global_step,
            "eval_loss": (metrics or {}).get("eval_loss"),
            "parse_rate": round(parse_rate, 4),
            "decoded_metrics": m,
            "samples": decoded[:3],
        }
        self.history.append(entry)

        print(
            f"[val decode] epoch {entry['epoch']}  "
            f"parse {len(parsed)}/{len(decoded)}  "
            + (f"mae_s {m['mae_s']:.2f}  mae_log {m['mae_log']:.3f}  "
               if parsed else "")
            + "e.g. " + " | ".join(repr(d["output"]) for d in decoded[:3]),
            flush=True,
        )
        if args.report_to and "wandb" in args.report_to:
            import wandb
            if wandb.run is not None:
                log = {"val/parse_rate": parse_rate, "step": state.global_step}
                if parsed:
                    log |= {"val/decoded_mae_s": m["mae_s"],
                            "val/decoded_mae_log": m["mae_log"]}
                wandb.log(log)


# ── Data ──────────────────────────────────────────────────────────────────────

def _load_split(df: pd.DataFrame, split: str, n_limit: int | None, seed: int):
    """Deterministic per-split frame: img_resolved rows only, stable order,
    optional seeded subsample (dry-run)."""
    part = df[(df["split"] == split) & df["img_resolved"]].sort_values(
        ["trajectory_id", "tab_id", "unit_index"], kind="mergesort"
    )
    if n_limit is not None and len(part) > n_limit:
        part = part.sample(n=n_limit, random_state=seed)
    return part


# ── Run card ──────────────────────────────────────────────────────────────────

def _write_train_card(
    path: Path, *, cfg_path: str, cfg: dict, dry_run: bool,
    n_train: int, n_val: int, winsor_cap: float, tcfg: dict,
    train_losses: list[dict], eval_losses: list[dict],
    decode_history: list[dict], vram: dict, checkpoint_dir: str,
) -> None:
    first = train_losses[0]["loss"] if train_losses else None
    last = train_losses[-1]["loss"] if train_losses else None
    img = cfg["image"]
    lines = [
        "# VLM train card — Qwen3-VL-4B QLoRA SFT (Path A)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}"
        f" · config `{cfg_path}` · seed {cfg['seed']}"
        f" · mode **{'DRY-RUN (VRAM smoke test)' if dry_run else 'FULL'}**_",
        "",
        f"- Model: `{cfg['model']['checkpoint']}` · 4-bit QLoRA · vision frozen"
        f" · LoRA r={cfg['model']['lora_r']} α={cfg['model']['lora_alpha']}"
        f" on language attn+MLP",
        f"- Target: `dwell_seconds: X.X` (winsor cap {winsor_cap:.3f} s,"
        f" train-split p95 — see artifacts/dataset_card.md)",
        f"- Task title in prompt: **{cfg['data']['include_task_title']}**"
        f" (False = screen-only, the primary RQ)",
        f"- Examples: train **{n_train}** · val **{n_val}** (img_resolved only)",
        "",
        "## Memory footprint",
        "",
        f"- per-device batch **{tcfg['batch_size']}** ×"
        f" grad-accum **{tcfg['grad_accum']}** = effective"
        f" **{tcfg['batch_size'] * tcfg['grad_accum']}**",
        f"- image bounds: max_side {img['max_side']},"
        f" pixels [{img['min_pixels']}, {img['max_pixels']}]"
        f" (≤{img['max_pixels'] // 784} visual tokens)"
        f" · max_seq_length {cfg['model']['max_seq_length']}",
        f"- peak VRAM: **{vram.get('peak_allocated_gb', 'n/a')} GB** allocated"
        f" / {vram.get('peak_reserved_gb', 'n/a')} GB reserved"
        f" of {vram.get('total_gb', 'n/a')} GB"
        f" ({vram.get('device', 'no CUDA device')})",
        "",
        "## Loss",
        "",
        f"- train loss: first {first} → last {last}",
        "",
        "| eval pass | epoch | step | val loss | parse rate | MAE (s) | MAE (log) |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, h in enumerate(decode_history):
        m = h["decoded_metrics"]
        lines.append(
            f"| {i} | {h['epoch']} | {h['step']} "
            f"| {h['eval_loss'] if h['eval_loss'] is not None else '–'} "
            f"| {h['parse_rate']:.2%} "
            f"| {m.get('mae_s', float('nan')):.2f} "
            f"| {m.get('mae_log', float('nan')):.3f} |"
            if m.get("n") else
            f"| {i} | {h['epoch']} | {h['step']} "
            f"| {h['eval_loss'] if h['eval_loss'] is not None else '–'} "
            f"| {h['parse_rate']:.2%} | – | – |"
        )
    lines += [
        "",
        "## Sample decodes (last eval pass)",
        "",
    ]
    if decode_history:
        lines += [
            f"- gold {d['gold_s']:.1f} s → `{d['output']}`"
            for d in decode_history[-1]["samples"]
        ]
    lines += [
        "",
        f"Checkpoints: `{checkpoint_dir}` (per epoch; final adapters in"
        f" `{checkpoint_dir}/final`).",
        "",
        "## Full log (JSON)",
        "",
        "```json",
        json.dumps(
            {
                "train_losses": train_losses,
                "eval_losses": eval_losses,
                "decode_history": decode_history,
                "vram": vram,
                "effective_config": {
                    "train": tcfg, "model": cfg["model"], "image": img,
                    "data": cfg["data"], "dry_run": dry_run,
                },
            },
            indent=2, default=str,
        ),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vlm.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="50 examples, few steps, same batch/pixels — "
                         "doubles as the VRAM smoke test")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("totvlm.train needs a CUDA GPU (Unsloth 4-bit QLoRA). "
                 "Install the training stack with `uv sync --extra vlm`.")

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)

    tcfg = dict(cfg["train"])
    ecfg = dict(cfg["eval"])
    n_train_limit = n_val_limit = max_steps = None
    # Dry runs get their own sandbox: never pollute the real checkpoint dir
    # (auto-resume below would otherwise pick up a 12-step dry checkpoint)
    # and never overwrite the full run's card.
    out_dir = cfg["paths"]["output_dir"]
    card_path = Path(cfg["paths"]["train_card"])
    if args.dry_run:
        d = cfg["dry_run"]
        n_train_limit, n_val_limit = d["n_train"], d["n_val"]
        max_steps = d["max_steps"]
        tcfg["logging_steps"] = d["logging_steps"]
        tcfg["report_to"] = d["report_to"]
        ecfg["decode_samples"] = d["decode_samples"]
        out_dir += "-dryrun"
        card_path = card_path.with_name(
            card_path.stem + "-dryrun" + card_path.suffix)

    # ── Data: rows_final → chat examples (train/val, img_resolved only) ──────
    df = pd.read_parquet(cfg["paths"]["rows_final"])
    # dwell_s is already winsorized at the train-split p95; its train max IS
    # the cap (logged in the dataset card). Recover it for the target clip.
    winsor_cap = float(df.loc[df["split"] == "train", "dwell_s"].max())

    img = cfg["image"]
    build = lambda part: build_vlm_examples(  # noqa: E731
        part,
        winsor_cap=winsor_cap,
        max_side=img["max_side"],
        min_pixels=img["min_pixels"],
        max_pixels=img["max_pixels"],
        include_task_title=cfg["data"]["include_task_title"],
    )
    print("1/4  Building chat examples ...", flush=True)
    train_ds = build(_load_split(df, "train", n_train_limit, seed))
    val_ds = build(_load_split(df, "val", n_val_limit, seed))
    print(f"     train {len(train_ds)} · val {len(val_ds)} "
          f"(winsor cap {winsor_cap:.3f} s)", flush=True)
    if not train_ds or not val_ds:
        sys.exit("No resolved-image rows in train/val — run the image "
                 "resolution stage of build_dataset.py first.")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("2/4  Loading 4-bit Qwen3-VL + LoRA ...", flush=True)
    model, processor = load_vlm(cfg)
    FastVisionModel.for_training(model)

    # ── Trainer (official Unsloth vision SFT recipe) ──────────────────────────
    print("3/4  Building trainer ...", flush=True)
    if tcfg["report_to"] == "wandb":
        os.environ.setdefault("WANDB_PROJECT", tcfg["wandb_project"])
    sft_args = _make_sft_config(
        output_dir=out_dir,
        seed=seed,
        num_train_epochs=tcfg["num_epochs"],
        **({"max_steps": max_steps} if max_steps else {}),
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        per_device_eval_batch_size=tcfg["eval_batch_size"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        optim=tcfg["optim"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=tcfg["save_total_limit"],
        bf16=True,
        fp16=False,
        report_to=tcfg["report_to"],
        run_name=tcfg["run_name"] + ("-dryrun" if args.dry_run else ""),
        # Required for vision finetuning with UnslothVisionDataCollator:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=cfg["model"]["max_seq_length"],
    )
    decode_cb = ValDecodeCallback(
        processor, val_ds,
        n_samples=ecfg["decode_samples"],
        max_new_tokens=ecfg["max_new_tokens"],
        seed=seed,
    )
    trainer = _make_trainer(
        model, processor,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=UnslothVisionDataCollator(model, processor),
        callbacks=[decode_cb],
    )

    # ── Train (auto-resume: a resubmitted/timed-out job continues from the
    # last epoch checkpoint instead of restarting) ────────────────────────────
    last_ckpt = get_last_checkpoint(out_dir) if Path(out_dir).is_dir() else None
    print(f"4/4  {'Resuming from ' + last_ckpt if last_ckpt else 'Training'}"
          " ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=last_ckpt)

    props = torch.cuda.get_device_properties(0)
    vram = {
        "device": props.name,
        "total_gb": round(props.total_memory / 2**30, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }

    # ── Save final adapters + card ────────────────────────────────────────────
    final_dir = Path(out_dir) / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))

    train_losses = [
        {"step": h["step"], "loss": round(h["loss"], 4)}
        for h in trainer.state.log_history if "loss" in h
    ]
    eval_losses = [
        {"step": h["step"], "eval_loss": round(h["eval_loss"], 4)}
        for h in trainer.state.log_history if "eval_loss" in h
    ]
    _write_train_card(
        card_path,
        cfg_path=args.config, cfg=cfg, dry_run=args.dry_run,
        n_train=len(train_ds), n_val=len(val_ds), winsor_cap=winsor_cap,
        tcfg=tcfg, train_losses=train_losses, eval_losses=eval_losses,
        decode_history=decode_cb.history, vram=vram, checkpoint_dir=out_dir,
    )
    print(f"\nAdapters → {final_dir} · card → {card_path}")

    # ── Dry-run acceptance summary ────────────────────────────────────────────
    if args.dry_run and train_losses:
        first, last = train_losses[0]["loss"], train_losses[-1]["loss"]
        rate = decode_cb.history[-1]["parse_rate"] if decode_cb.history else 0.0
        ok = last < first and rate > 0.0
        print(
            f"DRY-RUN {'PASS' if ok else 'FAIL'}: "
            f"loss {first:.4f} → {last:.4f} "
            f"({'dropped' if last < first else 'DID NOT DROP'}), "
            f"parse rate {rate:.0%}, "
            f"peak VRAM {vram['peak_allocated_gb']} GB "
            f"/ {vram['total_gb']} GB"
        )
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
