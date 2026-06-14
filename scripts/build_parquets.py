"""
build_parquets.py
=================
Build training parquets from WebChain JSON trajectories.

Key design decisions based on actual JSON structure:
─────────────────────────────────────────────────────
1. launchApp steps: no createdTime, no image, no rect → skip entirely,
   don't use for dwell calculation

2. Hash images (e.g. "pyuut6reea1k90pw7fgjdb5wbu5miw"): not URLs, can't
   fetch → use their createdTime for dwell math, don't create training rows

3. Dwell = createdTime[next_step] - createdTime[current_step] using ALL
   steps that have timestamps (including hash-image steps). Never skip a
   timestamp in the dwell chain — that inflates the previous step's dwell.

4. Training row emitted only when: step has a real https:// image URL AND
   has a valid dwell time (next step with timestamp exists)

5. From each step we use ONLY what has cognitive meaning:
   Already in JSON (no fetch):
     rect.width/height → motor difficulty (Fitts' Law)
     type              → cognitive type (click/type/scroll)
     createdTime       → dwell time
     value             → element text (what was clicked)
     step index        → Orient difficulty
   Needs one fetch:
     axTree URL        → count elements → store integer, discard tree
     img URL           → store bytes

   Discarded (no cognitive signal):
     href, selector, path, clientX/Y, screenX/Y, bubbles, cancelable,
     detail, buttonText/Color, imgMarkValue, devicePixelRatio, etc.

6. Regression target: z-scored log1p(dwell) within trajectory
   Normalises for task-speed differences between users/sessions.

Usage:
  python build_parquets.py <json_dir> <output_dir> [--workers N] [--chunk N]
"""

import sys
import json
import math
import time
import hashlib
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("build_parquets.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── TaskSense constants (ms) from Table 3, TaskSense paper ───────────────────
K_TYPE = {
    "Orient":     4.7,
    "Find":     563.2,
    "Extract": 1415.9,
    "Recall":   446.6,
    "Decide":   742.0,
    "Compute": 5120.1,
    "Create":  1422.1,
    "Verify":   778.1,
}
INTERCEPT_MS   = 859.1   # motor baseline (intercept from paper)
MEMORY_DECAY_T = 5       # steps for Recall decay


# ── Utilities ─────────────────────────────────────────────────────────────────
def is_url(s) -> bool:
    return isinstance(s, str) and s.startswith("https://")


def traj_split(traj_id: str) -> str:
    """Stable 70/15/15 train/val/test by hashing trajectory ID."""
    h = int(hashlib.md5(str(traj_id).encode()).hexdigest(), 16) % 100
    if h < 70:   return "train"
    elif h < 85: return "val"
    else:        return "test"


def action_to_cog(action: str) -> str:
    a = (action or "").lower()
    if a in ("click", "dblclick", "tap"):         return "Find"
    elif a in ("type", "input", "fill"):           return "Create"
    elif a in ("scroll", "swipe"):                 return "Extract"
    elif a in ("navigate", "goto", "launchapp"):   return "Orient"
    elif a in ("select", "choose"):                return "Decide"
    elif a in ("drag", "drop"):                    return "Compute"
    else:                                          return "Find"


# ── AX tree element count ─────────────────────────────────────────────────────
INTERACTIVE = {
    "button","link","a","input","combobox","checkbox","radio",
    "menuitem","tab","textbox","option","spinbutton","switch","treeitem",
}

def count_elements(node, depth=0) -> int:
    if not isinstance(node, dict) or depth > 15:
        return 0
    role  = (node.get("role") or node.get("type") or node.get("name") or "").lower()
    count = 1 if role in INTERACTIVE else 0
    for child in node.get("children", []):
        count += count_elements(child, depth + 1)
    return count


# ── TaskSense score ───────────────────────────────────────────────────────────
def compute_tasksense(action, n_el, rect, prev_rect, step_idx):
    """
    Compute TaskSense cognitive difficulty score (ms).

    Uses only fields available in JSON (rect) + one fetched integer (n_el).
    No AX tree content stored — just the count.
    """
    cog = action_to_cog(action)
    k   = K_TYPE.get(cog, K_TYPE["Find"])

    # Cognitive difficulty index (per TaskSense Table 1)
    if cog == "Find":
        n     = max(n_el, 1) if n_el > 0 else 5
        i_cog = math.log(n + 1)
    elif cog == "Orient":
        # log(completed_steps) + log(remaining) — use step_idx as proxy
        i_cog = math.log(max(step_idx, 1)) + math.log(2)
    elif cog == "Recall":
        i_cog = 1 - math.exp(-max(step_idx, 1) / MEMORY_DECAY_T)
    elif cog == "Create":
        i_cog = 1.0
    else:
        i_cog = math.log(6)

    # Motor difficulty — Fitts' Law from rect (already in JSON, no fetch)
    if rect:
        w    = max(rect.get("width", 50), 1)
        h    = max(rect.get("height", 30), 1)
        size = math.sqrt(w * h)
        if prev_rect:
            cx1  = prev_rect.get("x", 0) + prev_rect.get("width",  0) / 2
            cy1  = prev_rect.get("y", 0) + prev_rect.get("height", 0) / 2
            cx2  = rect.get("x", 0)      + w / 2
            cy2  = rect.get("y", 0)      + h / 2
            dist = max(math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2), 1)
        else:
            dist = 300
        d_motor = 100 * math.log2(2 * dist / size) + INTERCEPT_MS
    else:
        d_motor = INTERCEPT_MS

    return {
        "tasksense_ms":   round(k * i_cog + d_motor, 2),
        "cognitive_type": cog,
        "motor_ms":       round(d_motor, 2),
    }


# ── HTTP fetch with retry ─────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (research)"

def fetch(url: str, as_json=False, retries=2, timeout=12):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json() if as_json else r.content
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return None


def fetch_n_elements(ax_url: str) -> int:
    """Fetch AX tree, count interactive elements, return integer. Discard tree."""
    if not is_url(ax_url):
        return 0
    raw = fetch(ax_url)
    if not raw:
        return 0
    try:
        tree = json.loads(raw)
        if isinstance(tree, list):
            return sum(count_elements(n) for n in tree)
        return count_elements(tree)
    except Exception:
        return 0


# ── Dwell time calculation ────────────────────────────────────────────────────
def build_dwell_map(steps: list) -> dict:
    """
    Build step_index → dwell_seconds using ALL steps with createdTime.

    Hash-image steps and URL-image steps both contribute timestamps.
    launchApp steps (no createdTime) are skipped.

    Returns dict: {step_index: dwell_seconds}
    """
    # Collect (index, createdTime) for all steps that have a timestamp
    timed = [
        (i, step["createdTime"])
        for i, step in enumerate(steps)
        if step.get("createdTime") is not None
    ]

    dwells = {}
    for pos, (idx, t_curr) in enumerate(timed):
        if pos + 1 < len(timed):
            _, t_next  = timed[pos + 1]
            dwell_ms   = t_next - t_curr
            if dwell_ms > 0:
                dwells[idx] = round(dwell_ms / 1000.0, 3)
    return dwells


# ── Process one trajectory ────────────────────────────────────────────────────
def process_trajectory(traj: dict) -> list[dict]:
    traj_id     = traj.get("id") or traj.get("trajectory_id", "")
    steps       = traj.get("steps", [])
    instruction = traj.get("title") or traj.get("instruction", "")
    split       = traj_split(traj_id)
    total_steps = len(steps)

    if total_steps < 2:
        return []

    # Compute dwell for all steps using all timestamps
    dwell_map = build_dwell_map(steps)

    if len(dwell_map) < 2:
        return []   # need at least 2 dwells for z-score

    # Identify steps that can become training rows:
    # must have a real https:// image URL AND a dwell time
    candidate_indices = [
        i for i, step in enumerate(steps)
        if is_url(step.get("img", "")) and i in dwell_map
    ]

    if not candidate_indices:
        return []

    # ── Parallel fetch: images + AX element counts ────────────────────────────
    fetch_tasks = {}
    for i in candidate_indices:
        step = steps[i]
        fetch_tasks[f"img_{i}"] = (step["img"], "bytes")
        ax_url = step.get("axTree", "")
        if is_url(ax_url):
            fetch_tasks[f"ax_{i}"] = (ax_url, "count")

    fetched = {}
    if fetch_tasks:
        with ThreadPoolExecutor(max_workers=min(len(fetch_tasks), 16)) as ex:
            futures = {}
            for key, (url, mode) in fetch_tasks.items():
                if mode == "bytes":
                    futures[ex.submit(fetch, url)] = key
                else:
                    futures[ex.submit(fetch_n_elements, url)] = key
            for fut in as_completed(futures):
                fetched[futures[fut]] = fut.result()

    # ── Z-score log1p(dwell) within this trajectory ───────────────────────────
    all_dwells  = list(dwell_map.values())
    log_vals    = np.log1p(all_dwells)
    mu          = float(log_vals.mean())
    sigma       = float(log_vals.std()) or 1.0

    # ── Build training rows ───────────────────────────────────────────────────
    rows = []
    for i in candidate_indices:
        img_bytes = fetched.get(f"img_{i}")
        if img_bytes is None:
            continue   # fetch failed

        step      = steps[i]
        dwell     = dwell_map[i]
        prev_step = steps[i - 1] if i > 0 else None
        n_el      = fetched.get(f"ax_{i}", 0) or 0
        action    = step.get("type", "")

        ts         = compute_tasksense(
            action    = action,
            n_el      = n_el,
            rect      = step.get("rect"),
            prev_rect = prev_step.get("rect") if prev_step else None,
            step_idx  = i,
        )

        log_dwell        = math.log1p(dwell)
        difficulty_score = (log_dwell - mu) / sigma

        rows.append({
            # ── Identifiers ──────────────────────────────────────────────────
            "trajectory_id":    traj_id,
            "step_idx":         i,
            "step_frac":        round(i / max(total_steps - 1, 1), 3),
            "total_steps":      total_steps,
            "split":            split,
            # ── Model inputs ─────────────────────────────────────────────────
            "image":            img_bytes,       # PNG bytes from img URL
            "instruction":      instruction,     # task goal from trajectory.title
            "action_type":      action,          # click / type / scroll etc.
            "element_value":    step.get("value", ""),   # text of element clicked
            # ── Regression target ────────────────────────────────────────────
            # difficulty_score is z-scored within trajectory:
            #   positive = harder than avg step in this task
            #   negative = easier than avg step in this task  ← normal, not a bug
            "dwell_seconds":    dwell,
            "difficulty_score": round(difficulty_score, 4),  # ← what model predicts
            # ── TaskSense features (auxiliary training signal) ────────────────
            "tasksense_ms":     ts["tasksense_ms"],   # K_type * I_cog + D_motor
            "cognitive_type":   ts["cognitive_type"], # Find/Create/Extract/Orient…
            "n_elements":       n_el,            # count from AX tree (not the tree)
            "motor_ms":         ts["motor_ms"],  # Fitts' Law component
            # ── Geometry from JSON (no fetch needed) ─────────────────────────
            "target_w":         (step.get("rect") or {}).get("width",  0),
            "target_h":         (step.get("rect") or {}).get("height", 0),
            "viewport_w":       (step.get("viewport") or {}).get("width",  0),
            "viewport_h":       (step.get("viewport") or {}).get("height", 0),
            # ── Trajectory-level stats for denormalization at inference ───────
            "traj_mu":          round(mu, 4),
            "traj_sigma":       round(sigma, 4),
        })

    return rows


# ── Load trajectories from JSON files ────────────────────────────────────────
def iter_trajectories(json_dir: Path):
    files = sorted(json_dir.glob("**/*.json")) + \
            sorted(json_dir.glob("**/*.jsonl"))
    log.info(f"Found {len(files)} JSON files")

    for f in files:
        with open(f, "r") as fh:
            try:
                for line in fh:
                    if line.strip():
                        t = json.loads(line)
                        if t.get("steps"):
                            yield t
            except json.JSONDecodeError:
                fh.seek(0)
                try:
                    data  = json.load(fh)
                    items = data if isinstance(data, list) else [data]
                    for t in items:
                        if t.get("steps"):
                            yield t
                except json.JSONDecodeError:
                    log.warning(f"Skipping: {f.name}")


# ── Save chunk as compressed parquet ─────────────────────────────────────────
def save_chunk(rows: list, out_dir: Path, split: str, idx: int):
    path = out_dir / split / f"chunk_{idx:05d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, compression="zstd", index=False)
    mb = path.stat().st_size / 1024 ** 2
    log.info(f"  → {split}/chunk_{idx:05d}.parquet ({len(rows)} rows, {mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir",   help="Directory with JSON trajectory files")
    ap.add_argument("output_dir", help="Output directory for parquet files")
    ap.add_argument("--workers",  type=int, default=12)
    ap.add_argument("--chunk",    type=int, default=2000)
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    ckpt = out_dir / "completed.txt"
    done = set(ckpt.read_text().splitlines()) if ckpt.exists() else set()
    if done:
        log.info(f"Resuming — {len(done)} trajectories already done")

    buffers = defaultdict(list)
    totals  = defaultdict(int)
    stats   = dict(ok=0, skipped=0, failed=0, rows=0)

    # Initialize chunk counters from files already on disk so resuming
    # never overwrites chunks written in a previous run.
    chunks = defaultdict(int)
    for sp in ["train", "val", "test"]:
        existing = list((out_dir / sp).glob("chunk_*.parquet"))
        if existing:
            chunks[sp] = len(existing)
            log.info(f"  {sp}: found {len(existing)} existing chunks, "
                     f"next will be chunk_{len(existing):05d}.parquet")
    t0      = time.time()

    all_trajs = [t for t in iter_trajectories(json_dir)
                 if (t.get("id") or "") not in done]
    log.info(f"Trajectories to process: {len(all_trajs)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(process_trajectory, t): t.get("id","")
                      for t in all_trajs}

        for fut in as_completed(future_map):
            tid = future_map[fut]
            try:
                rows = fut.result()
                if rows:
                    sp = rows[0]["split"]
                    buffers[sp].extend(rows)
                    totals[sp]    += len(rows)
                    stats["rows"] += len(rows)
                    stats["ok"]   += 1
                    while len(buffers[sp]) >= args.chunk:
                        save_chunk(buffers[sp][:args.chunk], out_dir, sp, chunks[sp])
                        buffers[sp] = buffers[sp][args.chunk:]
                        chunks[sp] += 1
                else:
                    stats["skipped"] += 1

                with open(ckpt, "a") as f:
                    f.write(tid + "\n")

                n_done = stats["ok"] + stats["skipped"]
                if n_done % 200 == 0:
                    rate = n_done / max(time.time() - t0, 1)
                    eta  = (len(all_trajs) - n_done) / rate / 60
                    log.info(f"{n_done}/{len(all_trajs)} | "
                             f"{stats['rows']} rows | "
                             f"{rate:.1f}/s | ETA {eta:.0f}min")

            except Exception as e:
                log.error(f"Trajectory {tid} failed: {e}")
                stats["failed"] += 1

    for sp, rows in buffers.items():
        if rows:
            save_chunk(rows, out_dir, sp, chunks[sp])

    elapsed = (time.time() - t0) / 60
    log.info(f"\n{'='*50}")
    log.info(f"Done in {elapsed:.1f} min")
    log.info(f"  Total rows:  {stats['rows']}")
    log.info(f"  Trajectories processed: {stats['ok']}")
    log.info(f"  Skipped (no URL images or <2 dwells): {stats['skipped']}")
    log.info(f"  Failed: {stats['failed']}")
    for sp in ["train", "val", "test"]:
        log.info(f"  {sp}: {totals[sp]} rows")

    manifest = {
        "splits":     {s: int(totals[s]) for s in ["train","val","test"]},
        "total_rows": stats["rows"],
        "target":     "difficulty_score = zscore(log1p(dwell_seconds)) per trajectory. Negative = easier than avg, positive = harder.",
        "tasksense":  "K_type * log(n_elements+1) + Fitts_Law(rect)",
        "columns": [
            "trajectory_id", "step_idx", "step_frac", "total_steps", "split",
            "image", "instruction", "action_type", "element_value",
            "dwell_seconds", "difficulty_score",
            "tasksense_ms", "cognitive_type", "n_elements", "motor_ms",
            "target_w", "target_h", "viewport_w", "viewport_h",
            "traj_mu", "traj_sigma",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest written → {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
