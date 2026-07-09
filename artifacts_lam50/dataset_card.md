# Dataset card — per-screen dwell rows with domain-disjoint splits

- Rows: **206925** (unassigned split: 7)
- Winsor cap: **24.954 s** (p95 of raw dwell, train split only)
- Target: `y = log1p(dwell_s) with dwell_s = min(dwell_s_raw, cap)`

| split | rows | domains | trajectories | %navigation | median dwell | p95 dwell | %capped | %img resolved |
|---|---|---|---|---|---|---|---|---|
| train | 142609 | 135 | 22318 | 31.59% | 4.729 s | 24.954 s | 5.0% | 61.65% |
| val | 26317 | 29 | 4305 | 37.28% | 5.328 s | 23.478 s | 4.53% | 16.22% |
| test | 37992 | 29 | 5256 | 25.22% | 4.575 s | 25.672 s | 5.23% | 24.03% |

## Full stats (JSON)

```json
{
  "n_rows": 206925,
  "n_rows_unassigned_split": 7,
  "winsor": {
    "percentile": 0.95,
    "cap_dwell_s": 24.954,
    "computed_on": "train split only"
  },
  "target": "y = log1p(dwell_s) with dwell_s = min(dwell_s_raw, cap)",
  "per_split": {
    "train": {
      "rows": 142609,
      "domains": 135,
      "trajectories": 22318,
      "pct_navigation": 31.59,
      "dwell_s_raw_quantiles": {
        "p5": 0.879,
        "p25": 2.688,
        "p50": 4.729,
        "p75": 8.688,
        "p90": 16.259,
        "p95": 24.954,
        "p99": 59.231
      },
      "pct_capped_by_winsor": 5.0,
      "pct_img_resolved": 61.65
    },
    "val": {
      "rows": 26317,
      "domains": 29,
      "trajectories": 4305,
      "pct_navigation": 37.28,
      "dwell_s_raw_quantiles": {
        "p5": 1.119,
        "p25": 2.907,
        "p50": 5.328,
        "p75": 9.108,
        "p90": 15.824,
        "p95": 23.478,
        "p99": 59.404
      },
      "pct_capped_by_winsor": 4.53,
      "pct_img_resolved": 16.22
    },
    "test": {
      "rows": 37992,
      "domains": 29,
      "trajectories": 5256,
      "pct_navigation": 25.22,
      "dwell_s_raw_quantiles": {
        "p5": 0.808,
        "p25": 2.601,
        "p50": 4.575,
        "p75": 8.455,
        "p90": 16.47,
        "p95": 25.672,
        "p99": 65.88
      },
      "pct_capped_by_winsor": 5.23,
      "pct_img_resolved": 24.03
    }
  }
}
```
