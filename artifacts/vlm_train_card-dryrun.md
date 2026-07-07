# VLM train card — Qwen3-VL-4B QLoRA SFT (Path A)

_Generated 2026-07-06T09:53:35+00:00 · config `configs/vlm.yaml` · seed 42 · mode **DRY-RUN (VRAM smoke test)**_

- Model: `unsloth/Qwen3-VL-4B-Instruct` · 4-bit QLoRA · vision frozen · LoRA r=16 α=16 on language attn+MLP
- Target: `dwell_seconds: X.X` (winsor cap 24.954 s, train-split p95 — see artifacts/dataset_card.md)
- Task title in prompt: **False** (False = image+features, True = image+features+task)
- Features (train-time only): targets blended toward the privileged-feature teacher (λ=0.5) — **50/50** rows covered (100.0%), mean |shift| 1.454 s; val targets untouched, inference stays screenshot-only
- Examples: train **50** · val **16** (img_resolved only)

## Memory footprint

- per-device batch **4** × grad-accum **4** = effective **16**
- image bounds: max_side 1024, pixels [100352, 602112] (≤768 visual tokens) · max_seq_length 1536
- peak VRAM: **5.23 GB** allocated / 5.38 GB reserved of 39.49 GB (NVIDIA A100-PCIE-40GB)

## Loss

- train loss: first 3.8402 → last 0.8595

| eval pass | epoch | step | val loss | parse rate | MAE (s) | MAE (log) |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 4 | 2.19244122505188 | 100.00% | 7.26 | 0.900 |
| 1 | 2.0 | 8 | 1.1462383270263672 | 100.00% | 4.76 | 0.571 |
| 2 | 3.0 | 12 | 0.8399701714515686 | 100.00% | 4.68 | 0.551 |

## Sample decodes (last eval pass)

- gold 12.2 s → `dwell_seconds: 3.7`
- gold 7.0 s → `dwell_seconds: 3.8`
- gold 4.1 s → `dwell_seconds: 3.1`

Checkpoints: `artifacts/vlm_ckpt-dryrun` (per epoch; final adapters in `artifacts/vlm_ckpt-dryrun/final`).

## Full log (JSON)

```json
{
  "train_losses": [
    {
      "step": 1,
      "loss": 3.8402
    },
    {
      "step": 2,
      "loss": 3.9116
    },
    {
      "step": 3,
      "loss": 3.0434
    },
    {
      "step": 4,
      "loss": 2.5423
    },
    {
      "step": 5,
      "loss": 2.2253
    },
    {
      "step": 6,
      "loss": 1.9438
    },
    {
      "step": 7,
      "loss": 1.6605
    },
    {
      "step": 8,
      "loss": 1.3854
    },
    {
      "step": 9,
      "loss": 1.169
    },
    {
      "step": 10,
      "loss": 1.0039
    },
    {
      "step": 11,
      "loss": 0.9057
    },
    {
      "step": 12,
      "loss": 0.8595
    }
  ],
  "eval_losses": [
    {
      "step": 4,
      "eval_loss": 2.1924
    },
    {
      "step": 8,
      "eval_loss": 1.1462
    },
    {
      "step": 12,
      "eval_loss": 0.84
    }
  ],
  "decode_history": [
    {
      "epoch": 1.0,
      "step": 4,
      "eval_loss": 2.19244122505188,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 8,
        "mae_log": 0.8999446040670345,
        "rmse_log": 1.04367924589061,
        "mae_s": 7.262700000000001,
        "rmse_s": 9.732809839404036,
        "spearman_rho": -0.17878330548357893,
        "mean_actual_s": 7.29795,
        "mean_pred_s": 5.712500000000001
      },
      "samples": [
        {
          "output": "dwell_seconds: 3.2",
          "gold_s": 12.181,
          "pred_s": 3.2
        },
        {
          "output": "dwell_seconds: 3.2",
          "gold_s": 6.988,
          "pred_s": 3.2
        },
        {
          "output": "dwell_seconds: 3.2",
          "gold_s": 4.07,
          "pred_s": 3.2
        }
      ]
    },
    {
      "epoch": 2.0,
      "step": 8,
      "eval_loss": 1.1462383270263672,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 8,
        "mae_log": 0.5709712691475266,
        "rmse_log": 0.8035987892139703,
        "mae_s": 4.7646999999999995,
        "rmse_s": 8.42743599026418,
        "spearman_rho": 0.40050093945740717,
        "mean_actual_s": 7.29795,
        "mean_pred_s": 3.1375
      },
      "samples": [
        {
          "output": "dwell_seconds: 3.7",
          "gold_s": 12.181,
          "pred_s": 3.7
        },
        {
          "output": "dwell_seconds: 3.1",
          "gold_s": 6.988,
          "pred_s": 3.1
        },
        {
          "output": "dwell_seconds: 3.1",
          "gold_s": 4.07,
          "pred_s": 3.1
        }
      ]
    },
    {
      "epoch": 3.0,
      "step": 12,
      "eval_loss": 0.8399701714515686,
      "parse_rate": 1.0,
      "decoded_metrics": {
        "n": 8,
        "mae_log": 0.5512676511220787,
        "rmse_log": 0.7890463122116625,
        "mae_s": 4.6772,
        "rmse_s": 8.390621393556023,
        "spearman_rho": 0.4787529007367305,
        "mean_actual_s": 7.29795,
        "mean_pred_s": 3.225
      },
      "samples": [
        {
          "output": "dwell_seconds: 3.7",
          "gold_s": 12.181,
          "pred_s": 3.7
        },
        {
          "output": "dwell_seconds: 3.8",
          "gold_s": 6.988,
          "pred_s": 3.8
        },
        {
          "output": "dwell_seconds: 3.1",
          "gold_s": 4.07,
          "pred_s": 3.1
        }
      ]
    }
  ],
  "vram": {
    "device": "NVIDIA A100-PCIE-40GB",
    "total_gb": 39.49,
    "peak_allocated_gb": 5.23,
    "peak_reserved_gb": 5.38
  },
  "effective_config": {
    "train": {
      "learning_rate": 0.0002,
      "num_epochs": 2,
      "batch_size": 4,
      "grad_accum": 4,
      "eval_batch_size": 4,
      "warmup_ratio": 0.03,
      "weight_decay": 0.01,
      "lr_scheduler": "cosine",
      "optim": "adamw_8bit",
      "max_grad_norm": 1.0,
      "logging_steps": 1,
      "save_total_limit": 2,
      "report_to": "none",
      "wandb_project": "tot-vlm",
      "run_name": "qwen3vl4b-qlora-pathA"
    },
    "model": {
      "checkpoint": "unsloth/Qwen3-VL-4B-Instruct",
      "load_in_4bit": true,
      "gradient_checkpointing": true,
      "max_seq_length": 1536,
      "finetune_vision_layers": false,
      "lora_r": 16,
      "lora_alpha": 16,
      "lora_dropout": 0.0
    },
    "image": {
      "max_side": 1024,
      "min_pixels": 100352,
      "max_pixels": 602112
    },
    "data": {
      "include_task_title": false
    },
    "dry_run": true,
    "lupi": {
      "lambda": 0.5,
      "n_rows": 50,
      "n_blended": 50,
      "coverage": 1.0,
      "mean_abs_shift_s": 1.454
    }
  }
}
```
