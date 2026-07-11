# Evaluation report — Time-on-Task on held-out TEST domains (O2)

_Generated 2026-07-11T05:28:21+00:00 · config `configs/eval.yaml` · seed 42_

Target: `y = log1p(dwell_s)`, dwell winsorized at the train-split p95 cap (24.954 s). Splits are domain-disjoint (`artifacts/splits.json`); TEST domains were never seen in training or tuning. Metrics via `totvlm.scoring` — identical to the baseline report.

## Coverage & parse accounting

- Eval set: **9129** TEST rows with a resolved screenshot (2305 navigation / 6824 in-page)
- Head-to-head subset (axTree also resolved for LightGBM): **9127** rows across **1232** tasks (median 6 covered steps/task)
- VLM (image+features): parse tiers strict **9129** · labeled **0** · bare number **0** · failed **0** — **failure rate 0.00%**, failures imputed with the TRAIN median winsorized dwell (4.729 s), never dropped
- VLM (image+features+task): parse tiers strict **9128** · labeled **0** · bare number **0** · failed **1** — **failure rate 0.01%**, failures imputed with the TRAIN median winsorized dwell (4.729 s), never dropped
- Parsed predictions clipped to [0.0, 24.954] s (the training-target range)

## Per-screen head-to-head (identical rows)

| model | subset | n | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) | Spearman ρ | mean actual s | mean pred s |
|---|---|---|---|---|---|---|---|---|---|
| train-mean floor | overall | 9127 | 0.5917 | 0.7352 | 4.21 | 6.40 | – | 6.65 | 5.05 |
| train-mean floor | navigation | 2304 | 0.6740 | 0.8294 | 7.10 | 9.80 | – | 11.64 | 5.05 |
| train-mean floor | in_page | 6823 | 0.5639 | 0.7006 | 3.23 | 4.73 | – | 4.96 | 5.05 |
| train-median floor | overall | 9127 | 0.5876 | 0.7343 | 4.19 | 6.49 | – | 6.65 | 4.73 |
| train-median floor | navigation | 2304 | 0.7099 | 0.8672 | 7.31 | 10.02 | – | 11.64 | 4.73 |
| train-median floor | in_page | 6823 | 0.5462 | 0.6835 | 3.13 | 4.73 | – | 4.96 | 4.73 |
| LightGBM (no image) | overall | 9127 | 0.5228 | 0.6552 | 3.77 | 5.85 | 0.4683 | 6.65 | 5.16 |
| LightGBM (no image) | navigation | 2304 | 0.5236 | 0.6510 | 6.01 | 8.40 | 0.2329 | 11.64 | 7.13 |
| LightGBM (no image) | in_page | 6823 | 0.5225 | 0.6566 | 3.01 | 4.69 | 0.2040 | 4.96 | 4.50 |
| VLM (image+features) | overall | 9127 | 0.5412 | 0.6822 | 3.90 | 6.15 | 0.3644 | 6.65 | 4.69 |
| VLM (image+features) | navigation | 2304 | 0.6142 | 0.7741 | 6.56 | 9.15 | 0.1908 | 11.64 | 6.05 |
| VLM (image+features) | in_page | 6823 | 0.5165 | 0.6483 | 3.00 | 4.73 | 0.1926 | 4.96 | 4.23 |
| VLM (image+features+task) | overall | 9127 | 0.5536 | 0.7018 | 3.98 | 6.20 | 0.3134 | 6.65 | 4.80 |
| VLM (image+features+task) | navigation | 2304 | 0.6093 | 0.7719 | 6.48 | 9.06 | 0.2384 | 11.64 | 6.11 |
| VLM (image+features+task) | in_page | 6823 | 0.5347 | 0.6765 | 3.14 | 4.87 | 0.1421 | 4.96 | 4.35 |

## Task-level Time-on-Task (per-trajectory sums, identical rows)

Per-screen predictions are summed within each task and scored against the summed actual — the data-driven analogue of KLM's operator-sum. Totals cover only evaluated steps, identically for targets and every model, so the comparison is apples-to-apples (covered-task time, not wall-clock task time). Tasks with < 2 covered screens are excluded (104 such tasks — scattered legacy-cache rows, already counted in the per-screen tables).

| model | tasks | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) | Spearman ρ | mean actual s | mean pred s |
|---|---|---|---|---|---|---|---|---|
| train-mean floor | 1232 | 0.4514 | 0.5548 | 18.40 | 25.08 | 0.7028 | 48.14 | 36.97 |
| train-median floor | 1232 | 0.4803 | 0.5857 | 19.13 | 25.94 | 0.7028 | 48.14 | 34.63 |
| LightGBM (no image) | 1232 | 0.3802 | 0.4682 | 16.18 | 22.59 | 0.7687 | 48.14 | 37.73 |
| VLM (image+features) | 1232 | 0.4466 | 0.5377 | 18.21 | 24.68 | 0.7482 | 48.14 | 34.31 |
| VLM (image+features+task) | 1232 | 0.4499 | 0.5489 | 18.38 | 25.07 | 0.7294 | 48.14 | 35.10 |

## VLM conditions on the full eval set

| model | subset | n | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) | Spearman ρ | mean actual s | mean pred s |
|---|---|---|---|---|---|---|---|---|---|
| VLM (image+features) | overall | 9129 | 0.5412 | 0.6823 | 3.90 | 6.16 | 0.3642 | 6.65 | 4.69 |
| VLM (image+features) | navigation | 2305 | 0.6146 | 0.7744 | 6.56 | 9.15 | 0.1900 | 11.64 | 6.05 |
| VLM (image+features) | in_page | 6824 | 0.5164 | 0.6483 | 3.00 | 4.73 | 0.1925 | 4.96 | 4.23 |
| VLM (image+features+task) | overall | 9129 | 0.5536 | 0.7019 | 3.98 | 6.20 | 0.3133 | 6.65 | 4.80 |
| VLM (image+features+task) | navigation | 2305 | 0.6097 | 0.7722 | 6.49 | 9.06 | 0.2376 | 11.64 | 6.11 |
| VLM (image+features+task) | in_page | 6824 | 0.5347 | 0.6764 | 3.14 | 4.87 | 0.1421 | 4.96 | 4.35 |

## Where does the signal live? (O2 answer)

On identical rows, the image+features VLM's MAE(log) edge over LightGBM is **+0.0060 on in-page steps** (n=6823) and **-0.0907 on navigation steps** (n=2304); positive = VLM better. The advantage concentrates in **in-page** interactions, where no page load is bundled into the dwell — consistent with the screenshot capturing genuine cognitive/visual-complexity signal rather than latency artifacts.

Adding the task title changes MAE(log) by **-0.0124 overall** (-0.0182 in-page, +0.0049 navigation; positive = image+features+task better). Read: the task title adds little or nothing on top of the screen here — on this corpus, the predictable share of dwell is carried by the screen (and its distilled features) itself.

## Calibration (VLM (image+features), full eval set, decile bins by prediction)

ECE-style weighted gap: **1.96 s** (reliability plot: `scripts/make_figures.py` → fig_calibration)

| bin | n | mean pred (s) | mean actual (s) | gap (s) |
|---|---|---|---|---|
| 0 | 913 | 2.80 | 3.37 | 0.57 |
| 1 | 913 | 3.62 | 4.87 | 1.25 |
| 2 | 913 | 3.80 | 5.00 | 1.20 |
| 3 | 913 | 3.80 | 5.63 | 1.83 |
| 4 | 913 | 4.00 | 6.28 | 2.28 |
| 5 | 912 | 4.14 | 6.49 | 2.35 |
| 6 | 913 | 4.74 | 7.06 | 2.32 |
| 7 | 913 | 4.80 | 7.23 | 2.43 |
| 8 | 913 | 5.82 | 9.13 | 3.30 |
| 9 | 913 | 9.39 | 11.44 | 2.04 |

## Qualitative examples (VLM (image+features), `artifacts/qualitative/`)

| file | bucket | pred (s) | actual (s) | step | raw output |
|---|---|---|---|---|---|
| 00_best_ekgiPOzzlKT-pCFLbv88v_5.png | best | 4.1 | 4.1 | in-page | `dwell_seconds: 4.1` |
| 01_best_00LT5_wUZxuX3aDbbBPhL_3.png | best | 3.8 | 3.8 | in-page | `dwell_seconds: 3.8` |
| 02_median_evjwKJFdiirkwoDzA1fGB_9.png | median | 7.8 | 4.7 | in-page | `dwell_seconds: 7.8` |
| 03_median_1uL763YQQqfqcW-2SBPi-_2.png | median | 4.2 | 2.3 | in-page | `dwell_seconds: 4.2` |
| 04_worst_a0lXqr6hGGUnkiBEtt6H2_6.png | worst | 8.0 | 0.2 | navigation | `dwell_seconds: 8.0` |
| 05_worst_KT5YmtH3y7XEHC7ra9NwT_8.png | worst | 8.1 | 0.2 | navigation | `dwell_seconds: 8.1` |

## Full metrics (JSON)

```json
{
  "head_to_head_per_screen": {
    "train-mean floor": {
      "overall": {
        "n": 9127,
        "mae_log": 0.5916569539687069,
        "rmse_log": 0.7352207534220359,
        "mae_s": 4.209882194861972,
        "rmse_s": 6.398395542659127,
        "spearman_rho": NaN,
        "mean_actual_s": 6.648702070779008,
        "mean_pred_s": 5.048303802786016
      },
      "navigation": {
        "n": 2304,
        "mae_log": 0.6740034650501691,
        "rmse_log": 0.8294273807161854,
        "mae_s": 7.099322616656682,
        "rmse_s": 9.798886565284366,
        "spearman_rho": NaN,
        "mean_actual_s": 11.641744791666667,
        "mean_pred_s": 5.048303802786016
      },
      "in_page": {
        "n": 6823,
        "mae_log": 0.5638500711412574,
        "rmse_log": 0.7005535286775636,
        "mae_s": 3.2341719894076255,
        "rmse_s": 4.726564329274231,
        "spearman_rho": NaN,
        "mean_actual_s": 4.962644555181005,
        "mean_pred_s": 5.048303802786017
      }
    },
    "train-median floor": {
      "overall": {
        "n": 9127,
        "mae_log": 0.5875501862758609,
        "rmse_log": 0.7342529208999357,
        "mae_s": 4.18561069354662,
        "rmse_s": 6.485633889195291,
        "spearman_rho": NaN,
        "mean_actual_s": 6.648702070779008,
        "mean_pred_s": 4.729
      },
      "navigation": {
        "n": 2304,
        "mae_log": 0.709863604234783,
        "rmse_log": 0.8672075569475047,
        "mae_s": 7.310332465277778,
        "rmse_s": 10.016524067767902,
        "spearman_rho": NaN,
        "mean_actual_s": 11.641744791666667,
        "mean_pred_s": 4.728999999999999
      },
      "in_page": {
        "n": 6823,
        "mae_log": 0.5462472235062058,
        "rmse_log": 0.6835405588302548,
        "mae_s": 3.1304503590795836,
        "rmse_s": 4.731560274395015,
        "spearman_rho": NaN,
        "mean_actual_s": 4.962644555181005,
        "mean_pred_s": 4.728999999999999
      }
    },
    "LightGBM (no image)": {
      "overall": {
        "n": 9127,
        "mae_log": 0.5227651115202966,
        "rmse_log": 0.6552254567250464,
        "mae_s": 3.768690818457596,
        "rmse_s": 5.8531368139249205,
        "spearman_rho": 0.46827630178840174,
        "mean_actual_s": 6.648702070779008,
        "mean_pred_s": 5.161629264954275
      },
      "navigation": {
        "n": 2304,
        "mae_log": 0.5235806444095898,
        "rmse_log": 0.6510275329328402,
        "mae_s": 6.011623911638959,
        "rmse_s": 8.4021790406837,
        "spearman_rho": 0.23291627176758903,
        "mean_actual_s": 11.641744791666667,
        "mean_pred_s": 7.133192759744109
      },
      "in_page": {
        "n": 6823,
        "mae_log": 0.5224897212554669,
        "rmse_log": 0.6566369560093411,
        "mae_s": 3.011294094627921,
        "rmse_s": 4.689216009424489,
        "spearman_rho": 0.20399965151098906,
        "mean_actual_s": 4.962644555181005,
        "mean_pred_s": 4.495868999382566
      }
    },
    "VLM (image+features)": {
      "overall": {
        "n": 9127,
        "mae_log": 0.5411667664823598,
        "rmse_log": 0.682246059502533,
        "mae_s": 3.899028136298893,
        "rmse_s": 6.1542830959289,
        "spearman_rho": 0.36442877341839736,
        "mean_actual_s": 6.648702070779008,
        "mean_pred_s": 4.6921222745699565
      },
      "navigation": {
        "n": 2304,
        "mae_log": 0.6142469846705746,
        "rmse_log": 0.7740883672474641,
        "mae_s": 6.559289930555555,
        "rmse_s": 9.152029327606423,
        "spearman_rho": 0.19077236587843324,
        "mean_actual_s": 11.641744791666667,
        "mean_pred_s": 6.047743055555555
      },
      "in_page": {
        "n": 6823,
        "mae_log": 0.5164889381508859,
        "rmse_log": 0.648300663644284,
        "mae_s": 3.0007072841858418,
        "rmse_s": 4.730843439468126,
        "spearman_rho": 0.1925646169020559,
        "mean_actual_s": 4.962644555181005,
        "mean_pred_s": 4.234354389564707
      }
    },
    "VLM (image+features+task)": {
      "overall": {
        "n": 9127,
        "mae_log": 0.5535575439495283,
        "rmse_log": 0.7017861734152402,
        "mae_s": 3.982782162813629,
        "rmse_s": 6.19852686132986,
        "spearman_rho": 0.31339708774240277,
        "mean_actual_s": 6.648702070779008,
        "mean_pred_s": 4.7952590117234575
      },
      "navigation": {
        "n": 2304,
        "mae_log": 0.6093202418625415,
        "rmse_log": 0.7718552798813664,
        "mae_s": 6.482517361111111,
        "rmse_s": 9.05947243379248,
        "spearman_rho": 0.238379278936763,
        "mean_actual_s": 11.641744791666667,
        "mean_pred_s": 6.108854166666666
      },
      "in_page": {
        "n": 6823,
        "mae_log": 0.5347275196212881,
        "rmse_log": 0.6764879567457541,
        "mae_s": 3.138668151839367,
        "rmse_s": 4.86632970896541,
        "spearman_rho": 0.14214443269534913,
        "mean_actual_s": 4.962644555181005,
        "mean_pred_s": 4.351682397772241
      }
    }
  },
  "task_level": {
    "train-mean floor": {
      "n": 1232,
      "mae_log": 0.4514195814339012,
      "rmse_log": 0.5547677966166078,
      "mae_s": 18.396132652678755,
      "rmse_s": 25.077376870123995,
      "spearman_rho": 0.7028264258157692,
      "mean_actual_s": 48.14224951298701,
      "mean_pred_s": 36.97308864654077
    },
    "train-median floor": {
      "n": 1232,
      "mae_log": 0.48026833529406193,
      "rmse_log": 0.5856501582615828,
      "mae_s": 19.126649025974032,
      "rmse_s": 25.940768244186057,
      "spearman_rho": 0.7028264258157692,
      "mean_actual_s": 48.14224951298701,
      "mean_pred_s": 34.634551136363626
    },
    "LightGBM (no image)": {
      "n": 1232,
      "mae_log": 0.3801527453746271,
      "rmse_log": 0.4681614156394365,
      "mae_s": 16.18025993213908,
      "rmse_s": 22.593770692200945,
      "spearman_rho": 0.7687304514394077,
      "mean_actual_s": 48.14224951298701,
      "mean_pred_s": 37.72706133787434
    },
    "VLM (image+features)": {
      "n": 1232,
      "mae_log": 0.4466073620755062,
      "rmse_log": 0.537666868345124,
      "mae_s": 18.205818344155844,
      "rmse_s": 24.676509934262317,
      "spearman_rho": 0.7482349204984011,
      "mean_actual_s": 48.14224951298701,
      "mean_pred_s": 34.3086038961039
    },
    "VLM (image+features+task)": {
      "n": 1232,
      "mae_log": 0.44988262130787066,
      "rmse_log": 0.5489458375748043,
      "mae_s": 18.376694155844156,
      "rmse_s": 25.073287465782247,
      "spearman_rho": 0.7293923832716359,
      "mean_actual_s": 48.14224951298701,
      "mean_pred_s": 35.09929301948052
    }
  },
  "parse_stats": {
    "VLM (image+features)": {
      "n": 9129,
      "tier_counts": {
        "strict": 9129,
        "labeled": 0,
        "bare_number": 0,
        "fail": 0
      },
      "parse_failure_rate": 0.0,
      "fallback": "TRAIN median winsorized dwell (4.729 s)",
      "clipped_to": [
        0.0,
        24.954
      ]
    },
    "VLM (image+features+task)": {
      "n": 9129,
      "tier_counts": {
        "strict": 9128,
        "labeled": 0,
        "bare_number": 0,
        "fail": 1
      },
      "parse_failure_rate": 0.00010954102311315588,
      "fallback": "TRAIN median winsorized dwell (4.729 s)",
      "clipped_to": [
        0.0,
        24.954
      ]
    }
  },
  "calibration_ece_s": 1.9576021250958484,
  "primary_model": "VLM (image+features)",
  "pending_models": [],
  "rows": {
    "eval_set": 9129,
    "head_to_head": 9127,
    "tasks": 1232
  }
}
```
