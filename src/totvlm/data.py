"""
totvlm/data.py
==============
Data layer for Time-on-Task prediction. Three things live here:

1. Typed, tolerant views over the raw WebChain trajectory JSON
   (`Step`, `Trajectory`, `iter_raw_trajectories`) + the raw-data audit.
2. The Path-A SFT chat-example builder for the VLM
   (`build_vlm_examples`): screenshot → messages whose assistant turn is
   exactly `dwell_seconds: X.X` (winsorized dwell, 1 decimal).
3. The matching output parser (`parse_dwell_output`) used by the val
   decode hook and at evaluation time.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from totvlm.images import load_image

# ── Raw trajectory schema ─────────────────────────────────────────────────────
# Typed, tolerant views over the raw WebChain JSON. Every field is optional —
# the raw data is inconsistent (launchApp steps have no createdTime, images are
# sometimes URLs and sometimes opaque hashes, axTree may be null, ...).

def _as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Step:
    """One raw step. Field names are snake_case views of the JSON keys."""
    id: str | None = None
    type: str | None = None            # launchApp | click | type | select | ...
    tab_id: int | None = None          # JSON: tabId
    host: str | None = None
    href: str | None = None
    img: str | None = None             # https URL or opaque hash
    html: str | None = None
    ax_tree: str | None = None         # JSON: axTree (URL or null)
    created_time: int | None = None    # JSON: createdTime, epoch ms; no launchApp
    rect: dict | None = None
    viewport: dict | None = None
    scroll_x: float | None = None      # JSON: scrollX
    scroll_y: float | None = None      # JSON: scrollY
    value: str | None = None
    title: str | None = None
    node: dict | None = None           # JSON: attributes.data.node

    @classmethod
    def from_dict(cls, d: dict) -> Step:
        attrs = d.get("attributes")
        attrs_data = attrs.get("data") if isinstance(attrs, dict) else None
        node = attrs_data.get("node") if isinstance(attrs_data, dict) else None
        return cls(
            id=d.get("id"),
            type=d.get("type"),
            tab_id=_as_int(d.get("tabId")),
            host=d.get("host"),
            href=d.get("href"),
            img=d.get("img"),
            html=d.get("html"),
            ax_tree=d.get("axTree"),
            created_time=_as_int(d.get("createdTime")),
            rect=d.get("rect") if isinstance(d.get("rect"), dict) else None,
            viewport=(
                d.get("viewport") if isinstance(d.get("viewport"), dict) else None
            ),
            scroll_x=d.get("scrollX"),
            scroll_y=d.get("scrollY"),
            value=d.get("value"),
            title=d.get("title"),
            node=node if isinstance(node, dict) else None,
        )

    @property
    def is_launch_app(self) -> bool:
        return (self.type or "").lower() == "launchapp"

    @property
    def img_is_url(self) -> bool:
        return isinstance(self.img, str) and self.img.startswith(
            ("https://", "http://")
        )


@dataclass
class Trajectory:
    id: str | None = None
    title: str | None = None
    created_at: int | None = None      # JSON: createdAt
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Trajectory:
        return cls(
            id=d.get("id") or d.get("trajectory_id"),
            title=d.get("title") or d.get("instruction"),
            created_at=_as_int(d.get("createdAt")),
            steps=[Step.from_dict(s) for s in d.get("steps") or []
                   if isinstance(s, dict)],
        )


def _iter_json_objects(path: Path) -> Iterator[dict]:
    """Yield dicts from one file: JSONL, a JSON array, or a single object."""
    with open(path) as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":                      # one JSON array
            try:
                for obj in json.load(fh):
                    if isinstance(obj, dict):
                        yield obj
            except json.JSONDecodeError:
                return
            return
        try:                                  # JSONL (one object per line)
            for line in fh:
                if line.strip():
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
            return
        except json.JSONDecodeError:
            fh.seek(0)
        try:                                  # single JSON object
            obj = json.load(fh)
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            return


def iter_raw_trajectories(path: str | Path) -> Iterator[Trajectory]:
    """
    Stream typed trajectories from `path`: either a single JSON/JSONL file or a
    directory containing many of them (searched recursively, sorted for
    deterministic order). Objects without any steps are skipped.
    """
    path = Path(path)
    files = (
        sorted(path.glob("**/*.json")) + sorted(path.glob("**/*.jsonl"))
        if path.is_dir() else [path]
    )
    for f in files:
        for obj in _iter_json_objects(f):
            if obj.get("steps"):
                yield Trajectory.from_dict(obj)


# ── Raw-data audit ────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def audit_raw(trajectories: Iterable[Trajectory]) -> dict:
    """
    Compute descriptive stats over raw trajectories BEFORE trusting the data.
    Returns a JSON-serializable dict (see write_raw_audit for the markdown).
    """
    n_traj = 0
    n_steps = 0
    type_counts: Counter[str] = Counter()
    n_with_created = 0
    n_img_url = 0
    n_img_hash = 0
    n_img_missing = 0
    n_axtree = 0
    steps_per_traj: list[int] = []
    ident_runs = 0          # consecutive same-img+same-tab runs (len >= 2)
    ident_run_steps = 0     # actionable steps absorbed into those runs

    for traj in trajectories:
        n_traj += 1
        steps_per_traj.append(len(traj.steps))
        for s in traj.steps:
            n_steps += 1
            type_counts[s.type or "<missing>"] += 1
            if s.created_time is not None:
                n_with_created += 1
            if s.img_is_url:
                n_img_url += 1
            elif s.img:
                n_img_hash += 1
            else:
                n_img_missing += 1
            if s.ax_tree:
                n_axtree += 1

        # merge-eligible runs among actionable (non-launchApp) steps
        actionable = [s for s in traj.steps if not s.is_launch_app]
        run_len = 1
        for prev, cur in zip(actionable, actionable[1:]):
            if cur.img and cur.img == prev.img and cur.tab_id == prev.tab_id:
                run_len += 1
            else:
                if run_len >= 2:
                    ident_runs += 1
                    ident_run_steps += run_len
                run_len = 1
        if run_len >= 2:
            ident_runs += 1
            ident_run_steps += run_len

    spt = np.array(steps_per_traj) if steps_per_traj else np.array([0])
    return {
        "n_trajectories": n_traj,
        "n_steps": n_steps,
        "step_type_counts": dict(type_counts.most_common()),
        "pct_steps_with_created_time": _pct(n_with_created, n_steps),
        "img": {
            "pct_http_url": _pct(n_img_url, n_steps),
            "pct_opaque_hash": _pct(n_img_hash, n_steps),
            "pct_missing": _pct(n_img_missing, n_steps),
        },
        "pct_axtree_present": _pct(n_axtree, n_steps),
        "steps_per_trajectory": {
            "min": int(spt.min()),
            "p25": float(np.percentile(spt, 25)),
            "median": float(np.median(spt)),
            "p90": float(np.percentile(spt, 90)),
            "p99": float(np.percentile(spt, 99)),
            "max": int(spt.max()),
            "mean": round(float(spt.mean()), 2),
        },
        "consecutive_identical_img_runs": {
            "n_runs": ident_runs,
            "n_steps_in_runs": ident_run_steps,
        },
    }


def write_raw_audit(stats: dict, out_path: str | Path) -> None:
    """Render audit stats as markdown (plus the raw dict as fenced JSON)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img, spt = stats["img"], stats["steps_per_trajectory"]
    runs = stats["consecutive_identical_img_runs"]
    lines = [
        "# Raw WebChain audit",
        "",
        f"- Trajectories: **{stats['n_trajectories']}**",
        f"- Steps: **{stats['n_steps']}**",
        f"- Steps with `createdTime`: **{stats['pct_steps_with_created_time']}%**",
        f"- `img` is http(s) URL: **{img['pct_http_url']}%** · opaque hash: "
        f"**{img['pct_opaque_hash']}%** · missing: **{img['pct_missing']}%**",
        f"- `axTree` present: **{stats['pct_axtree_present']}%**",
        f"- Steps/trajectory: min {spt['min']} · p25 {spt['p25']} · median "
        f"{spt['median']} · p90 {spt['p90']} · p99 {spt['p99']} · max {spt['max']}"
        f" · mean {spt['mean']}",
        f"- Consecutive identical-img (same tab) runs ≥2: **{runs['n_runs']}** "
        f"covering {runs['n_steps_in_runs']} steps",
        "",
        "## Step type counts",
        "",
    ]
    lines += [
        f"- `{t}`: {c}" for t, c in stats["step_type_counts"].items()
    ]
    lines += ["", "## Raw stats (JSON)", "", "```json",
              json.dumps(stats, indent=2), "```", ""]
    out_path.write_text("\n".join(lines))

# ── Path-A SFT chat examples (VLM emits `dwell_seconds: X.X`) ─────────────────
# Prompt text is part of the label SPEC (SPEC.md), not a tunable knob — it
# stays here. Everything numeric (pixel bounds, flags) comes from configs/*.yaml.

SYSTEM_PROMPT = (
    "You estimate how many seconds a user spends on this screen before their "
    "next action. Reply with exactly: dwell_seconds: <number to 1 dp>"
)

# Exactly what the model is trained to emit — anything else is a parse failure.
DWELL_OUTPUT_RE = re.compile(r"^\s*dwell_seconds:\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def format_dwell_target(dwell_s: float, winsor_cap: float) -> str:
    """Assistant turn: dwell clipped to the winsor range [0, cap], 1 decimal.
    `dwell_s` in rows_final is already winsorized — the clip is a guarantee,
    not a second winsorization."""
    return f"dwell_seconds: {min(max(dwell_s, 0.0), winsor_cap):.1f}"


def parse_dwell_output(text: str | None) -> float | None:
    """Inverse of format_dwell_target. None ⇒ the output did not parse."""
    m = DWELL_OUTPUT_RE.match(text or "")
    return float(m.group(1)) if m else None


# Lenient tiers for EVALUATION decoding (training hooks stay strict):
# a real prediction buried in sloppy formatting should count as a prediction,
# not silently become an imputed median.
_DWELL_LABELED_RE = re.compile(
    r"dwell[_\s]?seconds?\s*[:=]?\s*(-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")

PARSE_TIERS = ("strict", "labeled", "bare_number", "fail")


def parse_dwell_output_lenient(text: str | None) -> tuple[float | None, str]:
    """
    Robust eval-time parser. Returns (seconds | None, tier):

      strict      — exactly the trained format (`dwell_seconds: X.X`)
      labeled     — a `dwell_seconds`-labeled number anywhere in the text
      bare_number — no label; first number in the text
      fail        — nothing usable (incl. negative values: dwell is never < 0)

    Tier counts are reported by the eval entrypoint — failures are imputed
    (train median) and NEVER hidden.
    """
    v = parse_dwell_output(text)
    if v is not None:
        return v, "strict"
    for regex, tier in ((_DWELL_LABELED_RE, "labeled"),
                        (_NUMBER_RE, "bare_number")):
        m = regex.search(text or "")
        if m:
            v = float(m.group(1) if regex.groups else m.group(0))
            return (v, tier) if v >= 0 else (None, "fail")
    return None, "fail"


def _vlm_messages(
    image: Image.Image,
    task_title: str | None,
    target_text: str,
    include_task_title: bool,
) -> list[dict]:
    """One chat example in the format UnslothVisionDataCollator expects."""
    user_content: list[dict] = [{"type": "image", "image": image}]
    if include_task_title and task_title:
        user_content.append({"type": "text", "text": f"Task: {task_title}"})
    return [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
        {"role": "assistant",
         "content": [{"type": "text", "text": target_text}]},
    ]


def build_vlm_examples(
    df: pd.DataFrame,
    *,
    winsor_cap: float,
    max_side: int,
    min_pixels: int,
    max_pixels: int,
    include_task_title: bool = False,
) -> list[dict]:
    """
    Rows → list of {"messages": [...], "dwell_s": float, "is_navigation": bool}.

    - Only rows with a locally resolved screenshot are used (img_resolved).
    - Images are loaded eagerly as downscaled PIL (bounded visual tokens).
    - Built with a LIST COMPREHENSION, not datasets.map(): keeping PIL images
      in plain Python dicts avoids Arrow image-type conversion issues.
    - DEFAULT is screen-only (no task title) — the research question is how
      much dwell is recoverable from the screen alone. include_task_title=True
      is the ablation.

    The extra keys (dwell_s, is_navigation) are ignored by the collator and
    used by the val decode hook / evaluation.
    """
    rows = df[df["img_resolved"] & df["img_path"].notna()]
    return [
        {
            "messages": _vlm_messages(
                load_image(
                    r.img_path,
                    max_side=max_side,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                ),
                getattr(r, "task_title", None),
                format_dwell_target(float(r.dwell_s), winsor_cap),
                include_task_title,
            ),
            "dwell_s": float(r.dwell_s),
            "is_navigation": bool(r.is_navigation),
        }
        for r in rows.itertuples(index=False)
    ]


def build_inference_examples(
    img_paths: list[str],
    *,
    max_side: int,
    min_pixels: int,
    max_pixels: int,
    task_title: str | None = None,
) -> list[dict]:
    """
    Prompt-only chat examples (no gold turn) for arbitrary screenshots —
    e.g. zero-shot external validation or scripts/predict.py. Same system
    prompt and user-turn format as training, so the model sees exactly the
    format it was tuned on. `task_title` (applied to every image) matches
    the screen+task condition; None matches screen-only.
    """
    return [
        {
            "messages": _vlm_messages(
                load_image(
                    p,
                    max_side=max_side,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                ),
                task_title, "", include_task_title=task_title is not None,
            )[:-1],   # drop the (empty) assistant turn: prompt only
        }
        for p in img_paths
    ]
