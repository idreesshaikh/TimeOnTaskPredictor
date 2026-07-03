# Raw WebChain audit

- Trajectories: **31675**
- Steps: **317682**
- Steps with `createdTime`: **86.59%**
- `img` is http(s) URL: **76.8%** · opaque hash: **9.84%** · missing: **13.36%**
- `axTree` present: **76.8%**
- Steps/trajectory: min 1 · p25 6.0 · median 9.0 · p90 17.0 · p99 29.0 · max 69 · mean 10.03
- Consecutive identical-img (same tab) runs ≥2: **24376** covering 49677 steps

## Step type counts

- `click`: 231541
- `launchApp`: 40831
- `type`: 21190
- `hover`: 10586
- `drag`: 4747
- `select`: 3121
- `press_enter`: 1844
- `paste`: 1242
- `back`: 1230
- `double_click`: 1150
- `copy`: 147
- `right_click`: 53

## Raw stats (JSON)

```json
{
  "n_trajectories": 31675,
  "n_steps": 317682,
  "step_type_counts": {
    "click": 231541,
    "launchApp": 40831,
    "type": 21190,
    "hover": 10586,
    "drag": 4747,
    "select": 3121,
    "press_enter": 1844,
    "paste": 1242,
    "back": 1230,
    "double_click": 1150,
    "copy": 147,
    "right_click": 53
  },
  "pct_steps_with_created_time": 86.59,
  "img": {
    "pct_http_url": 76.8,
    "pct_opaque_hash": 9.84,
    "pct_missing": 13.36
  },
  "pct_axtree_present": 76.8,
  "steps_per_trajectory": {
    "min": 1,
    "p25": 6.0,
    "median": 9.0,
    "p90": 17.0,
    "p99": 29.0,
    "max": 69,
    "mean": 10.03
  },
  "consecutive_identical_img_runs": {
    "n_runs": 24376,
    "n_steps_in_runs": 49677
  }
}
```
