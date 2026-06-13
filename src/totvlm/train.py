"""
train.py - the smallest loop that fine-tunes the model on screenshot-only ToT.

Run:  python -m totvlm.train      (from the src/ directory, or pip install -e .)

Set USE_HISTORY=False to get the single-screen baseline (your RQ2 ablation):
same data, same loop, one flag.
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor

from totvlm.data import get_dataloader
from totvlm.model import ToTRegressor

CKPT = "Qwen/Qwen3-VL-8B-Instruct"  # use "Qwen/Qwen3-VL-4B-Instruct" on a smaller GPU
STEPS, LR, BATCH = 1000, 1e-4, 2
USE_HISTORY = True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(CKPT)
    model = ToTRegressor(CKPT).to(device)
    loader = get_dataloader(
        processor, split="train", use_history=USE_HISTORY, batch_size=BATCH
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR)
    loss_fn = torch.nn.MSELoss()

    model.train()
    step = 0
    for batch in loader:
        target = batch.pop("target").to(device)
        batch.pop("tasksense", None)  # train-only signal, unused in this minimal loss
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

        pred = model(**batch)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        opt.zero_grad()

        step += 1
        if step % 10 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")
        if step >= STEPS:
            break

    model.backbone.save_pretrained("checkpoints/lora-tot")
    torch.save(model.head.state_dict(), "checkpoints/head.pt")
    print("saved LoRA adapters + head to checkpoints/")


if __name__ == "__main__":
    main()
