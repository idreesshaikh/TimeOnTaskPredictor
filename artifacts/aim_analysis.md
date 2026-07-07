# AIM theory-grounding analysis — what does the time sense consist of?

_Generated 2026-07-07T03:56:22+00:00 · config `configs/external.yaml` · seed 42 · post-hoc over cached zero-shot predictions (`artifacts/external_preds.parquet`)_

Predictors: per-screen **Aalto Interface Metrics** (visual-clutter models: contour_density, figure_ground_contrast, contour_congestion, subband_entropy, feature_congestion, color_clusters_static, color_clusters_dynamic, luminance_sd, colorfulness). Targets in log1p seconds; predictors z-scored; R² is in-sample OLS (descriptive). Spearman CIs: seeded bootstrap, 10000 resamples.

- Usable items: **894** (0 dropped: parse failure or missing metric)

| subset | n | human~AIM R² | pred~AIM R² | pred vs human ρ | **pred-residual vs human ρ** | partial (both residualized) ρ |
|---|---|---|---|---|---|---|
| web (in-domain, primary) | 299 | 0.128 | 0.082 | 0.067 [-0.050, 0.183] | **-0.008 [-0.123, 0.112]** | -0.011 [-0.126, 0.109] |
| ALL (incl. out-of-domain) | 894 | 0.191 | 0.134 | 0.096 [0.030, 0.161] | **-0.046 [-0.112, 0.019]** | 0.015 [-0.052, 0.082] |

Reading guide: column 3 anchors to the clutter literature (how much of *human* time is quantified clutter); column 4 says how much of the *model's* estimate is reducible to clutter; the bold column is the finding — a residual that still tracks human time means the model learned interface signal *beyond* established complexity models, and a residual that doesn't is itself the honest result (screenshot-predictable time ≈ clutter).

## Standardized coefficients (log-s per SD)

| metric | → human time | → model estimate |
|---|---|---|
| contour_density | +0.0528 | +0.0781 |
| figure_ground_contrast | -0.0426 | -0.0158 |
| contour_congestion | +0.0035 | +0.0041 |
| subband_entropy | +0.0332 | -0.0117 |
| feature_congestion | +0.0086 | +0.0220 |
| color_clusters_static | +0.0402 | +0.0377 |
| color_clusters_dynamic | -0.0171 | -0.0414 |
| luminance_sd | -0.0148 | -0.0507 |
| colorfulness | -0.0336 | +0.0180 |

_Coefficients from the first subset above (n=299)._

## Full results (JSON)

```json
{
  "web (in-domain, primary)": {
    "n": 299,
    "human_vs_aim_r2": 0.1284,
    "pred_vs_aim_r2": 0.0818,
    "pred_vs_human": {
      "n": 299,
      "rho": 0.0674,
      "ci": 0.95,
      "ci_lo": -0.0504,
      "ci_hi": 0.1835,
      "n_boot": 10000
    },
    "pred_residual_vs_human": {
      "n": 299,
      "rho": -0.0079,
      "ci": 0.95,
      "ci_lo": -0.1235,
      "ci_hi": 0.1117,
      "n_boot": 10000
    },
    "partial_residual_vs_residual": {
      "n": 299,
      "rho": -0.0109,
      "ci": 0.95,
      "ci_lo": -0.1261,
      "ci_hi": 0.1094,
      "n_boot": 10000
    },
    "human_coefs": {
      "contour_density": 0.0528,
      "figure_ground_contrast": -0.0426,
      "contour_congestion": 0.0035,
      "subband_entropy": 0.0332,
      "feature_congestion": 0.0086,
      "color_clusters_static": 0.0402,
      "color_clusters_dynamic": -0.0171,
      "luminance_sd": -0.0148,
      "colorfulness": -0.0336
    },
    "pred_coefs": {
      "contour_density": 0.0781,
      "figure_ground_contrast": -0.0158,
      "contour_congestion": 0.0041,
      "subband_entropy": -0.0117,
      "feature_congestion": 0.022,
      "color_clusters_static": 0.0377,
      "color_clusters_dynamic": -0.0414,
      "luminance_sd": -0.0507,
      "colorfulness": 0.018
    }
  },
  "ALL (incl. out-of-domain)": {
    "n": 894,
    "human_vs_aim_r2": 0.1906,
    "pred_vs_aim_r2": 0.1338,
    "pred_vs_human": {
      "n": 894,
      "rho": 0.0964,
      "ci": 0.95,
      "ci_lo": 0.0305,
      "ci_hi": 0.1611,
      "n_boot": 10000
    },
    "pred_residual_vs_human": {
      "n": 894,
      "rho": -0.0464,
      "ci": 0.95,
      "ci_lo": -0.1117,
      "ci_hi": 0.0194,
      "n_boot": 10000
    },
    "partial_residual_vs_residual": {
      "n": 894,
      "rho": 0.0147,
      "ci": 0.95,
      "ci_lo": -0.0523,
      "ci_hi": 0.082,
      "n_boot": 10000
    },
    "human_coefs": {
      "contour_density": 0.0464,
      "figure_ground_contrast": 0.0007,
      "contour_congestion": 0.0508,
      "subband_entropy": 0.1414,
      "feature_congestion": -0.0687,
      "color_clusters_static": 0.0168,
      "color_clusters_dynamic": -0.0635,
      "luminance_sd": -0.0395,
      "colorfulness": -0.0183
    },
    "pred_coefs": {
      "contour_density": 0.0197,
      "figure_ground_contrast": -0.0036,
      "contour_congestion": 0.0543,
      "subband_entropy": -0.0089,
      "feature_congestion": 0.062,
      "color_clusters_static": 0.0187,
      "color_clusters_dynamic": 0.01,
      "luminance_sd": 0.0181,
      "colorfulness": 0.0088
    }
  }
}
```
