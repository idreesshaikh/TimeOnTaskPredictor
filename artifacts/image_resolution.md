# Image resolution report

- Rows: **206925** · resolved (trainable): **101319** (**48.96%**)
- Unique refs: 206454 (URL 181895 / hash 24559)
- URL refs resolved: 100997 · hash refs found locally: 0

```json
{
  "n_rows": 206925,
  "n_rows_resolved": 101319,
  "pct_rows_resolved": 48.96,
  "n_refs": 206454,
  "n_url_refs": 181895,
  "n_hash_refs": 24559,
  "n_url_resolved": 100997,
  "n_hash_resolved": 0,
  "fetch_attempted_this_run": 0,
  "fetch_succeeded_this_run": 0,
  "trajectory_budget": {
    "per_split": {
      "train": {
        "budget_rows": 100000,
        "rows_sampled": 100003,
        "trajectories_sampled": 15660,
        "rows_resolved": 87779
      },
      "val": {
        "budget_rows": 5000,
        "rows_sampled": 5000,
        "trajectories_sampled": 817,
        "rows_resolved": 4250
      },
      "test": {
        "budget_rows": 10000,
        "rows_sampled": 10001,
        "trajectories_sampled": 1368,
        "rows_resolved": 9119
      }
    },
    "cache_gb_on_disk": 11.14,
    "store": {
      "max_side": 1280,
      "jpeg_quality": 85
    },
    "note": "whole trajectories per split, seeded prefix-stable (growing a budget only adds trajectories; reruns resume) \u2014 no sampled task has missing screens"
  }
}
```
