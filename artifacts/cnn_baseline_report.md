# Pixels-only CNN baseline report

_Generated 2026-07-13T11:55:56+00:00 · config `configs/cnn.yaml` · seed 42_

ImageNet-pretrained `resnet50`, fc → 1 output, input 224×224 (direct resize — aspect distortion accepted to keep the full page visible). Target `y = log1p(dwell_s)` (winsorized). No features, no distillation, no scaffold: this row is the pixels-only control between the floors and the VLM conditions. Trained on TRAIN, early-stopped on VAL MAE(log) (best epoch 0), evaluated ONCE on TEST.

## Epochs

| epoch | train loss | val MAE (log) |
|---|---|---|
| 0 | 0.4524 | 0.5899 |
| 1 | 0.3778 | 0.5955 |
| 2 | 0.3425 | 0.6187 |

## Test metrics

| subset | n | MAE (log) | MAE (s) | Spearman ρ |
|---|---|---|---|---|
| overall | 9129 | 0.5826 | 4.16 | 0.2095 |
| navigation | 2305 | 0.6174 | 6.58 | 0.1885 |
| in_page | 6824 | 0.5709 | 3.35 | 0.0723 |

## Full stats (JSON)

```json
{
  "history": [
    {
      "epoch": 0,
      "train_loss": 0.4524,
      "val_mae_log": 0.5899
    },
    {
      "epoch": 1,
      "train_loss": 0.3778,
      "val_mae_log": 0.5955
    },
    {
      "epoch": 2,
      "train_loss": 0.3425,
      "val_mae_log": 0.6187
    }
  ],
  "best_epoch": 0,
  "test_metrics": {
    "overall": {
      "n": 9129,
      "mae_log": 0.5826137284976747,
      "rmse_log": 0.7301795426484965,
      "mae_s": 4.161075154412685,
      "rmse_s": 6.251853372041119,
      "spearman_rho": 0.20954221142678475,
      "mean_actual_s": 6.649561813999343,
      "mean_pred_s": 5.190645050258453
    },
    "navigation": {
      "n": 2305,
      "mae_log": 0.6174000638939506,
      "rmse_log": 0.7705430145464566,
      "mae_s": 6.575348525245296,
      "rmse_s": 9.099382363604702,
      "spearman_rho": 0.18853669671623083,
      "mean_actual_s": 11.644405639913234,
      "mean_pred_s": 6.119180827656993
    },
    "in_page": {
      "n": 6824,
      "mae_log": 0.5708636547742844,
      "rmse_log": 0.7160317367463668,
      "mae_s": 3.345585687858,
      "rmse_s": 4.931564874974821,
      "spearman_rho": 0.07225746601345447,
      "mean_actual_s": 4.962411313012896,
      "mean_pred_s": 4.877005694029901
    }
  },
  "dry_run": false
}
```
