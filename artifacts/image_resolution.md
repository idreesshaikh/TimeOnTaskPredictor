# Image resolution report

- Rows: **206925** · resolved (trainable): **25681** (**12.41%**)
- Unique refs: 206454 (URL 181895 / hash 24559)
- URL refs resolved: 25612 · hash refs found locally: 0

```json
{
  "n_rows": 206925,
  "n_rows_resolved": 25681,
  "pct_rows_resolved": 12.41,
  "n_refs": 206454,
  "n_url_refs": 181895,
  "n_hash_refs": 24559,
  "n_url_resolved": 25612,
  "n_hash_resolved": 0,
  "fetch_attempted_this_run": 14655,
  "fetch_succeeded_this_run": 14655,
  "trajectory_budget": {
    "per_split": {
      "train": {
        "budget_rows": 20000,
        "rows_sampled": 20002,
        "trajectories_sampled": 3159,
        "rows_resolved": 17544
      },
      "val": {
        "budget_rows": 2000,
        "rows_sampled": 2000,
        "trajectories_sampled": 331,
        "rows_resolved": 1721
      },
      "test": {
        "budget_rows": 5000,
        "rows_sampled": 5002,
        "trajectories_sampled": 667,
        "rows_resolved": 4619
      }
    },
    "cache_gb_on_disk": 3.95,
    "store": {
      "max_side": 1280,
      "jpeg_quality": 85
    },
    "note": "whole trajectories per split, seeded prefix-stable (growing a budget only adds trajectories; reruns resume) \u2014 no sampled task has missing screens"
  },
  "url_success_rate_this_run": 1.0,
  "mean_image_kb_this_run": 108.1
}
```
