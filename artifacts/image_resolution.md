# Image resolution report

- Rows: **206925** · resolved (trainable): **2043** (**0.99%**)
- Unique refs: 206454 (URL 181895 / hash 24559)
- URL refs resolved: 2040 · hash refs found locally: 0

> Bounded run (--resolve-limit 40): img_resolved reflects files actually on disk NOW. Rerun without --resolve-limit to populate the full cache (resumable; cached files are skipped) and refresh this report.

```json
{
  "n_rows": 206925,
  "n_rows_resolved": 2043,
  "pct_rows_resolved": 0.99,
  "n_refs": 206454,
  "n_url_refs": 181895,
  "n_hash_refs": 24559,
  "n_url_resolved": 2040,
  "n_hash_resolved": 0,
  "fetch_attempted_this_run": 40,
  "fetch_succeeded_this_run": 40,
  "trajectory_budget": {
    "per_split": {
      "train": {
        "budget_rows": 6000,
        "rows_sampled": 6003,
        "trajectories_sampled": 935,
        "rows_resolved": 72
      },
      "val": {
        "budget_rows": 1200,
        "rows_sampled": 1202,
        "trajectories_sampled": 206,
        "rows_resolved": 16
      },
      "test": {
        "budget_rows": 3000,
        "rows_sampled": 3000,
        "trajectories_sampled": 396,
        "rows_resolved": 56
      }
    },
    "cache_gb_on_disk": 1.34,
    "store": {
      "max_side": 1280,
      "jpeg_quality": 85
    },
    "note": "whole trajectories per split, seeded prefix-stable (growing a budget only adds trajectories; reruns resume) \u2014 no sampled task has missing screens"
  },
  "url_success_rate_this_run": 1.0,
  "mean_image_kb_this_run": 104.5,
  "estimated_pct_rows_trainable_full_run": 88.1,
  "projected_full_cache_gb": 18.1,
  "note": "Bounded run (--resolve-limit 40): img_resolved reflects files actually on disk NOW. Rerun without --resolve-limit to populate the full cache (resumable; cached files are skipped) and refresh this report."
}
```
