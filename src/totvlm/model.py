"""
model.py - Qwen3-VL backbone with LoRA adapters + a small regression head.

Why a regression head (not text generation)? You want a *number* of seconds, and
training a linear head on the model's pooled hidden state gives a clean, stable
regression target. The head outputs the continuous difficulty score `s` from your
proposal; the BRIDGE step (s -> seconds) comes later, after training.

Only the LoRA adapters and the head are trainable; the 8B backbone stays frozen,
so this fits on a single mid-range GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Qwen3VLForConditionalGeneration


def _hidden_size(cfg):
    # Qwen3-VL config nests the LM config; fall back gracefully.
    return (
        getattr(cfg, "hidden_size", None)
        or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
        or 3584
    )


class ToTRegressor(nn.Module):
    def __init__(self, checkpoint, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.bfloat16
        )
        base.config.use_cache = False
        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(base, lora)
        self.head = nn.Linear(_hidden_size(base.config), 1)

    def forward(self, target=None, tasksense=None, **inputs):
        out = self.backbone(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1]  # (B, T, H)
        mask = inputs["attention_mask"]
        # index of the last real (non-pad) token, robust to left- or right-padding
        last = mask.size(1) - 1 - mask.flip(1).float().argmax(1)
        pooled = h[torch.arange(h.size(0), device=h.device), last]
        return self.head(pooled.float()).squeeze(-1)  # (B,) predicted score
