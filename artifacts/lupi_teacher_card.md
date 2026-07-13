# LUPI teacher card — out-of-fold LightGBM on privileged features

_Generated 2026-07-13T11:40:47+00:00 · config `configs/lupi.yaml` · seed 42 · k=5 domain-grouped folds_

Privileged features (axTree stats, nav flag, step index, click area) exist at TRAIN time only. Each train row's prediction comes from a booster that never saw its registrable domain; these become the soft half of both trained conditions' target (λ in configs/vlm.yaml). Val/test targets are never blended; inference stays screenshot-only.

- Coverage: **87921** of 87921 VLM train rows (**100.0%**) — uncovered rows keep the true label
- Excluded: 0 without an axTree URL · 0 fetch/parse failures (train+val)
- Scaffold stats (v3): **92190** train+val rows → `artifacts/scaffold_stats.parquet` — the six screen-describing axTree counts that supervise the target's `ui:` line

## Out-of-fold quality (log1p target space)

| subset | n | MAE (log) | MAE (s) | Spearman ρ |
|---|---|---|---|---|
| overall | 87921 | 0.5334 | 4.04 | 0.4223 |
| navigation | 28652 | 0.5192 | 5.51 | 0.0652 |
| in_page | 59269 | 0.5403 | 3.32 | 0.1881 |

## Folds

| fold | train rows | early-stop rows | predicted | best iter |
|---|---|---|---|---|
| 0 | 50299 | 18384 | 19238 | 20 |
| 1 | 50948 | 18589 | 18384 | 16 |
| 2 | 55020 | 14312 | 18589 | 42 |
| 3 | 56211 | 17398 | 14312 | 44 |
| 4 | 51285 | 19238 | 17398 | 26 |

## Full stats (JSON)

```json
{
  "k_folds": 5,
  "folds": [
    {
      "fold": 0,
      "n_train": 50299,
      "n_early_stop": 18384,
      "n_predicted": 19238,
      "best_iteration": 20
    },
    {
      "fold": 1,
      "n_train": 50948,
      "n_early_stop": 18589,
      "n_predicted": 18384,
      "best_iteration": 16
    },
    {
      "fold": 2,
      "n_train": 55020,
      "n_early_stop": 14312,
      "n_predicted": 18589,
      "best_iteration": 42
    },
    {
      "fold": 3,
      "n_train": 56211,
      "n_early_stop": 17398,
      "n_predicted": 14312,
      "best_iteration": 44
    },
    {
      "fold": 4,
      "n_train": 51285,
      "n_early_stop": 19238,
      "n_predicted": 17398,
      "best_iteration": 26
    }
  ],
  "oof_metrics": {
    "overall": {
      "n": 87921,
      "mae_log": 0.5334427369183629,
      "rmse_log": 0.6607170085589453,
      "mae_s": 4.035350304666995,
      "rmse_s": 6.031364246691628,
      "spearman_rho": 0.4223027362369068,
      "mean_actual_s": 6.9830422492919775,
      "mean_pred_s": 5.817312635451087
    },
    "navigation": {
      "n": 28652,
      "mae_log": 0.5192359878928747,
      "rmse_log": 0.6298593149350037,
      "mae_s": 5.510391446260579,
      "rmse_s": 7.4953881213410565,
      "spearman_rho": 0.06524167055627866,
      "mean_actual_s": 10.452138419656569,
      "mean_pred_s": 8.04029427520977
    },
    "in_page": {
      "n": 59269,
      "mae_log": 0.5403106066829663,
      "rmse_log": 0.6751288065038417,
      "mae_s": 3.3222814357989643,
      "rmse_s": 5.17724754648976,
      "spearman_rho": 0.18806757796309595,
      "mean_actual_s": 5.30600124179588,
      "mean_pred_s": 4.742672099211808
    }
  },
  "rows": {
    "vlm_train": 87921,
    "train_val_with_axtree_url": 92190,
    "covered": 87921,
    "coverage_of_vlm_train": 1.0
  },
  "excluded": {
    "no_axtree_url": 0,
    "axtree_fetch_failed": 0
  },
  "scaffold_stats_rows": 92190,
  "limit": null
}
```
