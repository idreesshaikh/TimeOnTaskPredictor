"""
totvlm/train.py
===============
Fine-tune Qwen3-VL-8B with LoRA for Time-on-Task regression.

Run:
    python -m totvlm.train

Ablation (screenshot-only, no history):
    USE_HISTORY = False
"""
from __future__ import annotations

import os

import torch
import wandb
from transformers import AutoProcessor

from totvlm.data import get_dataloader
from totvlm.model import ToTRegressor

# ── Config ────────────────────────────────────────────────────────────────────
CKPT        = "Qwen/Qwen3-VL-8B-Instruct"
DATASET_DIR = "webchain_dataset"
STEPS       = 1000
LR          = 1e-4
BATCH       = 1       # per-step micro-batch (keep at 1 if memory-bound)
GRAD_ACCUM  = 8       # accumulate N micro-batches -> effective batch = BATCH*GRAD_ACCUM
NUM_WORKERS = 0       # IterableDataset duplicates work across workers -> keep 0
USE_HISTORY = False   # OOM fix: each history image adds ~hundreds of tokens
AUX_LAMBDA  = 0.1    # weight of auxiliary tasksense loss in model (set 0 to disable)
GRAD_CKPT   = True   # True = fit on small GPU (slow). False = big GPU (~30-40% faster)
VAL_EVERY   = 100
SAVE_EVERY  = 200    # write a checkpoint every N steps (crash insurance)
SAVE_DIR    = "checkpoints/lora-tot"
RESUME      = True   # if True, resume from checkpoints/lora-tot/last if it exists


def _run_val(model, val_loader, device) -> dict:
    """Run one pass over val loader, return average losses."""
    model.eval()
    totals = {"loss": 0.0, "main_loss": 0.0, "aux_loss": 0.0}
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            target    = batch.pop("target").to(device)
            tasksense = batch.pop("tasksense", None)
            step_frac = batch.pop("step_frac", None)
            batch     = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            out = model(
                step_frac=step_frac.to(device) if step_frac is not None else None,
                tasksense=tasksense.to(device)  if tasksense is not None else None,
                target=target,
                **batch,
            )
            for k in totals:
                totals[k] += out[k].item()
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def _save_checkpoint(model, opt, step, tag, best_val):
    """
    Save LoRA adapters + both heads + optimizer + step counter under
    SAVE_DIR/<tag>/. Atomic-ish: writes then the caller can trust it exists.

    tag examples: "last" (rolling latest), "best" (lowest val loss),
                  "step_400" (periodic milestone).
    """
    path = os.path.join(SAVE_DIR, tag)
    os.makedirs(path, exist_ok=True)
    # LoRA adapters (PEFT handles its own format)
    model.backbone.save_pretrained(path)
    # Our two regression heads + training state in one file
    torch.save({
        "main_head": model.main_head.state_dict(),
        "aux_head":  model.aux_head.state_dict(),
        "optimizer": opt.state_dict(),
        "step":      step,
        "best_val":  best_val,
    }, os.path.join(path, "training_state.pt"))


def _load_checkpoint(model, opt, device):
    """
    Resume from SAVE_DIR/last if it exists. Returns (start_step, best_val).
    LoRA adapters are reloaded into the existing PEFT model in-place.
    """
    path = os.path.join(SAVE_DIR, "last")
    state_file = os.path.join(path, "training_state.pt")
    if not os.path.exists(state_file):
        return 0, float("inf")

    print(f"Resuming from {path} ...", flush=True)
    # Reload LoRA adapter weights into the current model
    from peft import PeftModel  # local import keeps top of file clean
    # The adapters were saved by save_pretrained; load_adapter merges them back
    model.backbone.load_adapter(path, adapter_name="default", is_trainable=True)

    ckpt = torch.load(state_file, map_location=device)
    model.main_head.load_state_dict(ckpt["main_head"])
    model.aux_head.load_state_dict(ckpt["aux_head"])
    opt.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    best_val   = ckpt.get("best_val", float("inf"))
    print(f"Resumed at step {start_step} (best_val={best_val:.4f})", flush=True)
    return start_step, best_val


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project="tot-vlm",
        config={
            "model":       CKPT,
            "steps":       STEPS,
            "lr":          LR,
            "batch":       BATCH,
            "use_history": USE_HISTORY,
            "aux_lambda":  AUX_LAMBDA,
        },
    )

    # ── Load processor and model ──────────────────────────────────────────────
    print("1/4  Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(CKPT)

    print("2/4  Loading model...", flush=True)
    model = ToTRegressor(CKPT, aux_lambda=AUX_LAMBDA, grad_checkpointing=GRAD_CKPT).to(device)
    model.print_trainable()

    # ── Build data loaders ────────────────────────────────────────────────────
    print(f"3/4  Building loaders (device={device})...", flush=True)
    train_loader = get_dataloader(
        processor,
        split="train",
        dataset_dir=DATASET_DIR,
        use_history=USE_HISTORY,
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
    )
    val_loader = get_dataloader(
        processor,
        split="val",
        dataset_dir=DATASET_DIR,
        use_history=USE_HISTORY,
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        max_chunks=4,   # small subset for quick validation
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print("4/4  Starting training...", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt       = torch.optim.AdamW(trainable, lr=LR)

    # Resume from last checkpoint if requested and one exists
    start_step, best_val = (_load_checkpoint(model, opt, device)
                            if RESUME else (0, float("inf")))

    # ── Training loop (with gradient accumulation) ────────────────────────────
    # We run GRAD_ACCUM forward/backward passes, summing gradients, then take
    # ONE optimizer step. Effective batch size = BATCH * GRAD_ACCUM, but peak
    # memory stays at BATCH. Far fewer optimizer steps = less Python overhead
    # and smoother gradients, without using more VRAM.
    model.train()
    step = start_step  # resume-aware: continue from where we left off
    micro = 0          # counts micro-batches since last optimizer step
    accum_loss = accum_main = accum_aux = 0.0

    for batch in train_loader:
        target    = batch.pop("target").to(device)
        tasksense = batch.pop("tasksense", None)
        step_frac = batch.pop("step_frac", None)
        batch     = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        out = model(
            step_frac = step_frac.to(device) if step_frac is not None else None,
            tasksense = tasksense.to(device)  if tasksense is not None else None,
            target    = target,
            **batch,
        )

        # Scale loss so accumulated gradient equals the average over GRAD_ACCUM
        (out["loss"] / GRAD_ACCUM).backward()
        accum_loss += out["loss"].item()
        accum_main += out["main_loss"].item()
        accum_aux  += out["aux_loss"].item()
        micro += 1

        if micro < GRAD_ACCUM:
            continue   # keep accumulating, don't step yet

        # One optimizer step after GRAD_ACCUM micro-batches
        opt.step()
        opt.zero_grad()
        step += 1

        # Average the accumulated losses for logging
        loss_v = accum_loss / GRAD_ACCUM
        main_v = accum_main / GRAD_ACCUM
        aux_v  = accum_aux  / GRAD_ACCUM
        accum_loss = accum_main = accum_aux = 0.0
        micro = 0

        if step % 10 == 0:
            print(
                f"step {step:4d}  "
                f"loss {loss_v:.4f}  "
                f"main {main_v:.4f}  "
                f"aux {aux_v:.4f}",
                flush=True,
            )
            wandb.log({
                "train/loss":      loss_v,
                "train/main_loss": main_v,
                "train/aux_loss":  aux_v,
                "step":            step,
            })

        # ── Validation ───────────────────────────────────────────────────────
        if step % VAL_EVERY == 0:
            val_metrics = _run_val(model, val_loader, device)
            print(
                f"step {step:4d}  "
                f"VAL loss {val_metrics['loss']:.4f}  "
                f"main {val_metrics['main_loss']:.4f}  "
                f"aux {val_metrics['aux_loss']:.4f}",
                flush=True,
            )
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()} | {"step": step})

            # Track and save the best model by validation loss
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                _save_checkpoint(model, opt, step, "best", best_val)
                print(f"  new best val loss {best_val:.4f} -> saved checkpoints/.../best",
                      flush=True)

        # Periodic crash-insurance checkpoint (rolling "last")
        if step % SAVE_EVERY == 0:
            _save_checkpoint(model, opt, step, "last", best_val)
            print(f"  checkpoint saved at step {step} -> checkpoints/.../last", flush=True)

        if step >= STEPS:
            break

    # ── Final save ────────────────────────────────────────────────────────────
    _save_checkpoint(model, opt, step, "final", best_val)
    # Also drop the LoRA adapters at SAVE_DIR root for easy loading at inference
    model.backbone.save_pretrained(SAVE_DIR)
    print(f"\nSaved final model -> {SAVE_DIR}/final/")
    print(f"Best val loss seen: {best_val:.4f} (in {SAVE_DIR}/best/)")
    print(f"Training complete: {step} steps")
    wandb.finish()


if __name__ == "__main__":
    main()
