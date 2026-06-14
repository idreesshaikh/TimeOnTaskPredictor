"""
totvlm/model.py
===============
Qwen3-VL backbone + LoRA adapters + regression heads.

Architecture
------------
                    ┌─────────────────────────┐
screenshot ──────→  │  Qwen3-VL-8B (frozen)   │
instruction ─────→  │  + LoRA on q/k/v/o_proj │ → last-token pooled hidden (H)
                    └─────────────────────────┘
                              │         │
                    ┌─────────┘         └──────────────┐
                    ▼                                   ▼
          cat([pooled, step_frac])         pooled (H)
                  (H+1)                               │
                    │                      aux_head (Linear H→1)
             main_head (Linear H+1→1)                 │
                    │                      log(tasksense_ms)  ← training only
             difficulty_score              (auxiliary supervision)

Training loss:
    loss = MSE(difficulty_score, target) + λ * MSE(pred_tasksense, log(tasksense_ms))

At inference:
    - step_frac is available (we know which step we're on)
    - tasksense aux_head is unused (no AX tree at inference)
    - only main_head output is returned

Why multi-task?
    The aux_head forces the backbone to encode "how many interactive elements
    are on screen / how complex is this action" from pixels alone. This improves
    the main difficulty prediction without needing AX trees at inference.

Only LoRA adapters + both heads are trainable (~1% of parameters).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# TF32: lets cuDNN/cuBLAS use tensor cores for fp32 matmuls.
# ~1.3-2x faster on Ampere+ (A100/A6000/4090) at negligible precision cost.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from peft import LoraConfig, get_peft_model
from transformers import Qwen3VLForConditionalGeneration


# ── Config helpers ────────────────────────────────────────────────────────────
def _hidden_size(cfg) -> int:
    """Extract hidden_size from Qwen3-VL's (potentially nested) config."""
    return (
        getattr(cfg, "hidden_size", None)
        or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
        or 3584   # Qwen3-VL-8B default
    )


# ── Model ─────────────────────────────────────────────────────────────────────
class ToTRegressor(nn.Module):
    """
    Qwen3-VL + LoRA + two regression heads.

    Args:
        checkpoint  : HuggingFace model ID or local path
        lora_r      : LoRA rank
        lora_alpha  : LoRA scaling (usually 2 * r)
        lora_dropout: dropout on LoRA layers
        aux_lambda  : weight of auxiliary tasksense loss (0 = disable aux head)
    """

    def __init__(
        self,
        checkpoint:   str,
        lora_r:       int   = 16,
        lora_alpha:   int   = 32,
        lora_dropout: float = 0.05,
        aux_lambda:   float = 0.1,
        grad_checkpointing: bool = True,   # set False on a big GPU for ~30-40% speedup
    ):
        super().__init__()
        self.aux_lambda = aux_lambda

        # Load backbone in bfloat16 (saves ~50% VRAM vs float32)
        # attn_implementation="sdpa" uses PyTorch's fused scaled-dot-product
        # attention kernel — faster and lower-memory than the default eager path.
        # (If flash-attn is installed, "flash_attention_2" is faster still.)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        base.config.use_cache = False

        # Step 1: register forward hook on embed_tokens so gradients can flow
        # through checkpointed layers. Only needed WHEN checkpointing is on.
        # We do this manually because enable_input_require_grads() fails on
        # Qwen3-VL — the vision encoder (Qwen3VLVisionModel) doesn't implement
        # get_input_embeddings(), so any code that calls it crashes.
        if grad_checkpointing:
            for _name, _mod in base.named_modules():
                if "embed_tokens" in _name:
                    _mod.register_forward_hook(
                        lambda m, inp, out: out.requires_grad_(True)
                    )
                    break

        # Step 2: wrap with LoRA BEFORE enabling gradient checkpointing.
        # If GC is already enabled when get_peft_model runs, PEFT detects it
        # and calls enable_input_require_grads() itself — same crash as above.
        # Wrapping first means PEFT never triggers that path.
        lora_cfg = LoraConfig(
            r              = lora_r,
            lora_alpha     = lora_alpha,
            lora_dropout   = lora_dropout,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type      = "CAUSAL_LM",
        )
        self.backbone = get_peft_model(base, lora_cfg)

        # Step 3: enable gradient checkpointing ONLY if requested.
        # Checkpointing recomputes activations on the backward pass instead of
        # storing them: saves ~4-8x activation memory but adds a full extra
        # forward pass per step (~30-40% slower). Turn it OFF once you have the
        # VRAM headroom — that is the single biggest speedup for a compute-bound
        # run. We also re-enable KV cache use when not checkpointing.
        if grad_checkpointing:
            self.backbone.base_model.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            print("  [model] gradient checkpointing OFF — faster, needs more VRAM")

        H = _hidden_size(base.config)

        # Main head: pooled(H) + step_frac(1) → difficulty_score
        self.main_head = nn.Linear(H + 1, 1)

        # Aux head: pooled(H) → log(tasksense_ms) — training only signal
        self.aux_head = nn.Linear(H, 1)

    def _pool(self, inputs: dict) -> torch.Tensor:
        """Run backbone, return last non-pad token hidden state. Shape: (B, H)."""
        out  = self.backbone(**inputs, output_hidden_states=True)
        h    = out.hidden_states[-1]                    # (B, T, H)
        mask = inputs["attention_mask"]

        # Last real (non-pad) token — works for both left- and right-padding
        last   = mask.size(1) - 1 - mask.flip(1).float().argmax(dim=1)
        pooled = h[torch.arange(h.size(0), device=h.device), last]  # (B, H)

        return pooled.float()

    def forward(
        self,
        step_frac:  torch.Tensor | None = None,
        tasksense:  torch.Tensor | None = None,   # log(tasksense_ms), training only
        target:     torch.Tensor | None = None,   # difficulty_score, training only
        **inputs,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass.

        At training (target is not None):
            Returns dict with "loss", "pred", and "pred_tasksense".

        At inference (target is None):
            Returns difficulty_score tensor of shape (B,).
        """
        pooled = self._pool(inputs)   # (B, H)

        # step_frac: 0.0→1.0 task progress. Zeros if not provided (first step / unknown)
        if step_frac is None:
            step_frac = torch.zeros(pooled.size(0), device=pooled.device)
        step_frac = step_frac.unsqueeze(-1).to(pooled.dtype)  # (B, 1)

        # Main prediction: concat step_frac to pooled
        pred = self.main_head(
            torch.cat([pooled, step_frac], dim=-1)
        ).squeeze(-1)   # (B,)

        if target is None:
            return pred   # inference: return score directly

        # Training: compute loss
        main_loss = nn.functional.mse_loss(pred, target.float())

        aux_loss = torch.tensor(0.0, device=pred.device)
        if self.aux_lambda > 0 and tasksense is not None:
            # Predict log(tasksense_ms) — forces backbone to encode UI complexity
            pred_ts  = self.aux_head(pooled).squeeze(-1)
            log_ts   = torch.log(tasksense.float().clamp(min=1.0))
            aux_loss = nn.functional.mse_loss(pred_ts, log_ts)

        loss = main_loss + self.aux_lambda * aux_loss

        return {
            "loss":          loss,
            "main_loss":     main_loss,
            "aux_loss":      aux_loss,
            "pred":          pred,
        }

    def print_trainable(self):
        """Print parameter count summary."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: {total:,} total, {trainable:,} trainable "
              f"({100 * trainable / total:.2f}%)")
