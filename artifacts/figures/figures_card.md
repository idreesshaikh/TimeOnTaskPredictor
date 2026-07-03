# Figures card

_Generated 2026-07-03T18:19:02+00:00 · config `configs/figures.yaml` · seed 42_

Regenerable anytime: `uv run python scripts/make_figures.py` (CPU-only, reads cached artifacts, never runs a model).

| figure | status | input |
|---|---|---|
| fig_dwell_distribution | ✅ written | — |
| fig_feature_importance | ✅ written | — |
| fig_head_to_head | ⏳ skipped | `eval_metrics` not present yet |
| fig_task_level | ⏳ skipped | `eval_metrics` not present yet |
| fig_screen_scatter | ⏳ skipped | `eval_predictions` not present yet |
| fig_task_scatter | ⏳ skipped | `eval_predictions` not present yet |
| fig_calibration | ⏳ skipped | `eval_predictions` not present yet |

Fixed model colors across every figure: train-mean floor `#c3c2b7` · train-median floor `#898781` · LightGBM (no image) `#1baf7a` · VLM (screen) `#2a78d6` · VLM (screen+task) `#4a3aa7`
