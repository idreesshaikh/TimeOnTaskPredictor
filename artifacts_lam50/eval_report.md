# Evaluation report — Time-on-Task on held-out TEST domains (O2)

_Generated 2026-07-07T03:50:20+00:00 · config `configs/eval.yaml` · seed 42_

Target: `y = log1p(dwell_s)`, dwell winsorized at the train-split p95 cap (24.954 s). Splits are domain-disjoint (`artifacts/splits.json`); TEST domains were never seen in training or tuning. Metrics via `totvlm.scoring` — identical to the baseline report.

## Coverage & parse accounting

- Eval set: **9129** TEST rows with a resolved screenshot (2305 navigation / 6824 in-page)
- Head-to-head subset (axTree also resolved for LightGBM): **9127** rows across **1232** tasks (median 6 covered steps/task)
- VLM (image+features): parse tiers strict **9129** · labeled **0** · bare number **0** · failed **0** — **failure rate 0.00%**, failures imputed with the TRAIN median winsorized dwell (4.729 s), never dropped
- VLM (image+features+task): parse tiers strict **9129** · labeled **0** · bare number **0** · failed **0** — **failure rate 0.00%**, failures imputed with the TRAIN median winsorized dwell (4.729 s), never dropped
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
| VLM (image+features) | overall | 9127 | 0.5442 | 0.6967 | 3.91 | 6.13 | 0.3761 | 6.65 | 4.65 |
| VLM (image+features) | navigation | 2304 | 0.5996 | 0.7629 | 6.33 | 8.77 | 0.2816 | 11.64 | 6.57 |
| VLM (image+features) | in_page | 6823 | 0.5255 | 0.6729 | 3.10 | 4.92 | 0.2169 | 4.96 | 4.00 |
| VLM (image+features+task) | overall | 9127 | 0.5630 | 0.7202 | 4.02 | 6.20 | 0.3368 | 6.65 | 4.81 |
| VLM (image+features+task) | navigation | 2304 | 0.6144 | 0.7835 | 6.38 | 8.78 | 0.2198 | 11.64 | 6.88 |
| VLM (image+features+task) | in_page | 6823 | 0.5457 | 0.6976 | 3.23 | 5.05 | 0.1845 | 4.96 | 4.11 |

## Task-level Time-on-Task (per-trajectory sums, identical rows)

Per-screen predictions are summed within each task and scored against the summed actual — the data-driven analogue of KLM's operator-sum. Totals cover only evaluated steps, identically for targets and every model, so the comparison is apples-to-apples (covered-task time, not wall-clock task time). Tasks with < 2 covered screens are excluded (104 such tasks — scattered legacy-cache rows, already counted in the per-screen tables).

| model | tasks | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) | Spearman ρ | mean actual s | mean pred s |
|---|---|---|---|---|---|---|---|---|
| train-mean floor | 1232 | 0.4514 | 0.5548 | 18.40 | 25.08 | 0.7028 | 48.14 | 36.97 |
| train-median floor | 1232 | 0.4803 | 0.5857 | 19.13 | 25.94 | 0.7028 | 48.14 | 34.63 |
| LightGBM (no image) | 1232 | 0.3802 | 0.4682 | 16.18 | 22.59 | 0.7687 | 48.14 | 37.73 |
| VLM (image+features) | 1232 | 0.4612 | 0.5602 | 18.60 | 25.36 | 0.7325 | 48.14 | 33.96 |
| VLM (image+features+task) | 1232 | 0.4519 | 0.5544 | 18.41 | 25.26 | 0.7170 | 48.14 | 35.09 |

## VLM conditions on the full eval set

| model | subset | n | MAE (log) | RMSE (log) | MAE (s) | RMSE (s) | Spearman ρ | mean actual s | mean pred s |
|---|---|---|---|---|---|---|---|---|---|
| VLM (image+features) | overall | 9129 | 0.5443 | 0.6968 | 3.91 | 6.13 | 0.3759 | 6.65 | 4.65 |
| VLM (image+features) | navigation | 2305 | 0.6001 | 0.7635 | 6.33 | 8.77 | 0.2808 | 11.64 | 6.57 |
| VLM (image+features) | in_page | 6824 | 0.5254 | 0.6728 | 3.10 | 4.92 | 0.2169 | 4.96 | 4.00 |
| VLM (image+features+task) | overall | 9129 | 0.5631 | 0.7202 | 4.03 | 6.20 | 0.3368 | 6.65 | 4.81 |
| VLM (image+features+task) | navigation | 2305 | 0.6143 | 0.7834 | 6.38 | 8.78 | 0.2200 | 11.64 | 6.88 |
| VLM (image+features+task) | in_page | 6824 | 0.5457 | 0.6976 | 3.23 | 5.05 | 0.1844 | 4.96 | 4.11 |

## Where does the signal live? (O2 answer)

On identical rows, the image+features VLM's MAE(log) edge over LightGBM is **-0.0030 on in-page steps** (n=6823) and **-0.0761 on navigation steps** (n=2304); positive = VLM better. The VLM does not beat the baseline on either subset, so there is no edge to attribute — reading the screenshot at inference recovers no more than LightGBM's interpretable features here.

Adding the task title changes MAE(log) by **-0.0188 overall** (-0.0202 in-page, -0.0147 navigation; positive = image+features+task better). Read: the task title adds little or nothing on top of the screen here — on this corpus, the predictable share of dwell is carried by the screen (and its distilled features) itself.

## Calibration (VLM (image+features), full eval set, decile bins by prediction)

ECE-style weighted gap: **2.00 s** (reliability plot: `scripts/make_figures.py` → fig_calibration)

| bin | n | mean pred (s) | mean actual (s) | gap (s) |
|---|---|---|---|---|
| 0 | 913 | 2.04 | 3.59 | 1.55 |
| 1 | 913 | 2.44 | 4.64 | 2.20 |
| 2 | 913 | 2.95 | 5.19 | 2.24 |
| 3 | 913 | 3.21 | 4.95 | 1.74 |
| 4 | 913 | 3.54 | 6.43 | 2.89 |
| 5 | 912 | 3.98 | 6.41 | 2.44 |
| 6 | 913 | 4.34 | 6.35 | 2.01 |
| 7 | 913 | 5.02 | 7.39 | 2.37 |
| 8 | 913 | 8.49 | 10.93 | 2.44 |
| 9 | 913 | 10.46 | 10.61 | 0.15 |

## Qualitative examples (VLM (image+features), `artifacts/qualitative/`)

| file | bucket | pred (s) | actual (s) | step | raw output |
|---|---|---|---|---|---|
| 00_best_VI4XohvTsmxIjTPDejejG_5.png | best | 3.5 | 3.5 | in-page | `dwell_seconds: 3.5` |
| 01_best_BtOX6hzcMJCzcWlyOqZri_2.png | best | 4.2 | 4.2 | in-page | `dwell_seconds: 4.2` |
| 02_median_rD7TSV3VAYYUTiyL-r3Uf_1.png | median | 4.0 | 2.2 | in-page | `dwell_seconds: 4.0` |
| 03_median_YDb6srZAao2d_7vafnwFU_2.png | median | 3.0 | 1.6 | in-page | `dwell_seconds: 3.0` |
| 04_worst_KT5YmtH3y7XEHC7ra9NwT_14.png | worst | 10.0 | 0.1 | in-page | `dwell_seconds: 10.0` |
| 05_worst_0N9Kq2l0-BRzh9kwbrMyJ_4.png | worst | 1.6 | 25.0 | in-page | `dwell_seconds: 1.6` |

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
        "mae_log": 0.5441890283087386,
        "rmse_log": 0.6966995877058152,
        "mae_s": 3.911012578065082,
        "rmse_s": 6.12569299869876,
        "spearman_rho": 0.3760749894813157,
        "mean_actual_s": 6.648702070779008,
        "mean_pred_s": 4.647551221650049
      },
      "navigation": {
        "n": 2304,
        "mae_log": 0.5996429939402441,
        "rmse_log": 0.7629200108216286,
        "mae_s": 6.325796875,
        "rmse_s": 8.768448206822526,
        "spearman_rho": 0.28164301430247096,
        "mean_actual_s": 11.641744791666667,
        "mean_pred_s": 6.5713541666666675
      },
      "in_page": {
        "n": 6823,
        "mae_log": 0.5254632571208464,
        "rmse_log": 0.672867871852728,
        "mae_s": 3.0955849040011727,
        "rmse_s": 4.922646241145048,
        "spearman_rho": 0.21685124913258377,
        "mean_actual_s": 4.962644555181005,
        "mean_pred_s": 3.9979188040451414
      }
    },
    "VLM (image+features+task)": {
      "overall": {
        "n": 9127,
        "mae_log": 0.5630202771226578,
        "rmse_log": 0.7202359049131337,
        "mae_s": 4.0248026514736495,
        "rmse_s": 6.203015694242746,
        "spearman_rho": 0.33677454695394055,
        "mean_actual_s": 6.648702070779008,
        "mean_pred_s": 4.806782075161609
      },
      "navigation": {
        "n": 2304,
        "mae_log": 0.6143539321765981,
        "rmse_log": 0.7834960382665244,
        "mae_s": 6.379493923611111,
        "rmse_s": 8.777096245589442,
        "spearman_rho": 0.2198380255596538,
        "mean_actual_s": 11.641744791666667,
        "mean_pred_s": 6.877517361111112
      },
      "in_page": {
        "n": 6823,
        "mae_log": 0.5456858580629658,
        "rmse_log": 0.6975796620457932,
        "mae_s": 3.2296672724607944,
        "rmse_s": 5.045434075191686,
        "spearman_rho": 0.18448797844484247,
        "mean_actual_s": 4.962644555181005,
        "mean_pred_s": 4.107533343104206
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
      "mae_log": 0.4611680833748515,
      "rmse_log": 0.5602310623393092,
      "mae_s": 18.602901136363638,
      "rmse_s": 25.361231951685138,
      "spearman_rho": 0.7325074799758706,
      "mean_actual_s": 48.14224951298701,
      "mean_pred_s": 33.956574675324674
    },
    "VLM (image+features+task)": {
      "n": 1232,
      "mae_log": 0.45185372769249804,
      "rmse_log": 0.5544325224730909,
      "mae_s": 18.41214431818182,
      "rmse_s": 25.261022087771185,
      "spearman_rho": 0.7169865814142506,
      "mean_actual_s": 48.14224951298701,
      "mean_pred_s": 35.09488636363636
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
    }
  },
  "calibration_ece_s": 2.002426311753751,
  "primary_model": "VLM (image+features)",
  "pending_models": [],
  "rows": {
    "eval_set": 9129,
    "head_to_head": 9127,
    "tasks": 1232
  }
}
```
