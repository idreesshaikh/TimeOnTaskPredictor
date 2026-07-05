# External validation set card (READ-ONLY)

_Generated 2026-07-03T05:36:31+00:00 · config `configs/external.yaml` · seed 42_

## Source decision

- **First choice — TaskSense (arXiv 2511.09309)**: NOT available. As of 2026-07-03 the paper releases no dataset, code, screenshots, or per-item human times (checked the arXiv listing and web search); the paper's tables contain only fitted difficulty constants without the underlying screenshots, so no screenshot→time transcription is possible.
- **Used: documented fallback VSGUI10K** (Putkonen et al. 2025, https://osf.io/hmg9b, public). ⚠️ It measures visual SEARCH time — one component of Time-on-Task — so rank agreement against it is a weaker but fully independent check.
- **In-domain primary = `web`** (pre-registered in configs/external.yaml): the training corpus (WebChain) is web-only, so the web category is the fair comparison; desktop/mobile screens are kept and reported as out-of-domain transfer, never as the headline.
- `aim_metrics.csv`: per-screen Aalto Interface Metrics (visual clutter models), shipped with VSGUI10K — consumed post-hoc by the theory-grounding analysis (`scripts/analyze_aim.py`), never by training/tuning.

## Contract (SPEC.md)

- Lives under `data/external/` and is chmod'd **read-only**.
- Evaluated **exactly once, zero-shot** by `scripts/validate_external.py`.
- NEVER used to pick features, hyperparameters, thresholds, or the winsorization cap; no training/tuning code may reference the path (enforced by `tests/test_external_guard.py`).

## Stats

- Items (screens): **894**
- human_time_s: median 2.75 s

```json
{
  "source": "vsgui10k",
  "osf": "https://osf.io/hmg9b",
  "trials_total": 10282,
  "trials_target_present": 9366,
  "screens_min_trials": 3,
  "screens_missing_image": 0,
  "n_items": 894,
  "n_aim_rows": 894,
  "per_category": {
    "web": 299,
    "desktop": 298,
    "mobile": 297
  },
  "human_time_s_quantiles": {
    "p5": 1.57,
    "p25": 2.09,
    "p50": 2.75,
    "p75": 3.62,
    "p95": 6.34
  },
  "aggregate": "median",
  "measure": "visual search time (last fixation timestamp of the search phase), target-present trials only"
}
```
