# No-image LightGBM baseline report

_Generated 2026-07-13 · config `configs/baseline.yaml` · seed 42 · best iteration 21_

Target: `y = log1p(dwell_s)` (winsorized at the train-split p95 cap; see `artifacts/dataset_card.md`). Trained on TRAIN, early-stopped on VAL l1, evaluated ONCE on TEST. Splits are domain-disjoint (`artifacts/splits.json`).

## Row accounting

- Rows in dataset: **206925**
- Excluded — unassigned split: **7**
- Excluded — no axTree URL: **24619**
- Sampled for axTree fetching (seed 42, prefix-stable): **20000** of 182299 eligible (trees average ~1.2 MB; mirroring all ~182k is ~210 GB — features are extracted on the fly and cached instead)
- Excluded — axTree fetch/parse failed: **3**
- Final rows: train **13696** · val **2436** · test **3865**

## Test metrics (LightGBM)

| subset | n | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) | Spearman ρ | mean actual s | mean pred s |
|---|---|---|---|---|---|---|---|---|
| overall | 3865 | 0.5084 | 0.6418 | 3.72 | 5.81 | 0.4813 | 6.74 | 5.18 |
| navigation | 1004 | 0.5244 | 0.6550 | 6.07 | 8.50 | 0.2055 | 11.82 | 7.15 |
| in_page | 2861 | 0.5027 | 0.6371 | 2.89 | 4.49 | 0.2105 | 4.96 | 4.49 |

## Floors (constant predictors, TEST overall)

| subset | n | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) | Spearman ρ | mean actual s | mean pred s |
|---|---|---|---|---|---|---|---|---|
| train-mean floor | 3865 | 0.5824 | 0.7251 | 4.20 | 6.34 | nan | 6.74 | 5.18 |
| train-median floor | 3865 | 0.5780 | 0.7239 | 4.17 | 6.42 | nan | 6.74 | 4.88 |

## Calibration (TEST, decile bins by prediction)

ECE-style weighted gap: **1.56 s**

| bin | n | mean pred (s) | mean actual (s) | gap (s) |
|---|---|---|---|---|
| 0 | 387 | 3.35 | 3.70 | 0.35 |
| 1 | 386 | 3.94 | 4.41 | 0.47 |
| 2 | 387 | 4.23 | 4.40 | 0.17 |
| 3 | 386 | 4.44 | 5.10 | 0.66 |
| 4 | 387 | 4.59 | 5.21 | 0.62 |
| 5 | 386 | 4.82 | 5.76 | 0.94 |
| 6 | 386 | 5.32 | 6.59 | 1.27 |
| 7 | 387 | 5.95 | 7.56 | 1.61 |
| 8 | 386 | 6.91 | 11.76 | 4.85 |
| 9 | 387 | 8.25 | 12.91 | 4.66 |

## Feature importances (gain)

| feature | gain | % |
|---|---|---|
| f_is_navigation | 9981.3 | 40.78% |
| ax_n_nodes | 2971.0 | 12.14% |
| click_target_area | 2270.3 | 9.28% |
| ax_n_inputs | 2040.7 | 8.34% |
| ax_text_len | 1989.2 | 8.13% |
| ax_n_buttons | 1568.4 | 6.41% |
| ax_n_links | 1472.1 | 6.01% |
| ax_n_interactive | 1069.9 | 4.37% |
| f_unit_index | 940.3 | 3.84% |
| action_hover | 101.9 | 0.42% |
| action_click | 70.1 | 0.29% |
| action_drag | 0.0 | 0.0% |
| action_press_enter | 0.0 | 0.0% |
| action_double_click | 0.0 | 0.0% |
| action_type | 0.0 | 0.0% |
| action_select | 0.0 | 0.0% |
| action_other | 0.0 | 0.0% |

## Full metrics (JSON)

```json
{
  "lightgbm_test": {
    "overall": {
      "n": 3865,
      "mae_log": 0.5083765559390342,
      "rmse_log": 0.6417805444672414,
      "mae_s": 3.7177888163763404,
      "rmse_s": 5.805399770401353,
      "spearman_rho": 0.4812652723441226,
      "mean_actual_s": 6.740109598965072,
      "mean_pred_s": 5.18066366833739
    },
    "navigation": {
      "n": 1004,
      "mae_log": 0.5244115531585799,
      "rmse_log": 0.6549861908753596,
      "mae_s": 6.069876679483117,
      "rmse_s": 8.496314755038755,
      "spearman_rho": 0.20548049230890789,
      "mean_actual_s": 11.820488247011951,
      "mean_pred_s": 7.148290453067693
    },
    "in_page": {
      "n": 2861,
      "mae_log": 0.5027494545030243,
      "rmse_log": 0.6370814551690619,
      "mae_s": 2.8923794439334163,
      "rmse_s": 4.4941490614642605,
      "spearman_rho": 0.2105323883386871,
      "mean_actual_s": 4.957271373645578,
      "mean_pred_s": 4.490171780232103
    }
  },
  "floors_test": {
    "train-mean floor": {
      "n": 3865,
      "mae_log": 0.5823802942493025,
      "rmse_log": 0.7250758312603244,
      "mae_s": 4.19599104846688,
      "rmse_s": 6.339643762208949,
      "spearman_rho": NaN,
      "mean_actual_s": 6.740109598965072,
      "mean_pred_s": 5.184664724324692
    },
    "train-median floor": {
      "n": 3865,
      "mae_log": 0.5779954320292331,
      "rmse_log": 0.7239334561221298,
      "mae_s": 4.169472600258732,
      "rmse_s": 6.421769727437109,
      "spearman_rho": NaN,
      "mean_actual_s": 6.740109598965072,
      "mean_pred_s": 4.878
    }
  },
  "calibration_ece_s": 1.5594459306276816,
  "best_iteration": 21,
  "rows": {
    "train": 13696,
    "val": 2436,
    "test": 3865
  },
  "excluded": {
    "unassigned_split": 7,
    "no_axtree_url": 24619,
    "axtree_fetch_failed": 3
  },
  "sample": {
    "requested": 20000,
    "taken": 20000
  }
}
```
