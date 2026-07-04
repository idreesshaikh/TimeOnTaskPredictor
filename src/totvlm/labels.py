"""Per-screen dwell labels — implements EXACTLY the label spec in SPEC.md.
Tolerances for messy raw data: steps merge only on a non-empty `img`; units
with no createdTime are removed from their tab's chain (counted); all unit
metadata comes from the FIRST merged step. Winsorization happens later,
train-split-only (splits.py)."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import tldextract

from totvlm.data import Step, Trajectory

# hard filters (SPEC.md §7) — overridable via build_rows kwargs / CLI
MIN_DWELL_S = 0.05
MAX_DWELL_S = 600.0

# bundled public-suffix snapshot: no network at import/call
_extract = tldextract.TLDExtract(suffix_list_urls=())

ROW_COLUMNS = [
    "trajectory_id", "task_title", "tab_id", "unit_index", "img_ref", "href",
    "prev_href", "axtree_ref", "dwell_s_raw", "is_navigation", "host",
    "registrable_domain", "viewport_w", "viewport_h", "action_type",
    "target_w", "target_h", "n_steps_merged",
]


def registrable_domain(host: str | None, href: str | None = None) -> str | None:
    """eTLD+1 via tldextract (offline snapshot). Falls back from host to href."""
    for candidate in (host, href):
        if candidate:
            reg = _extract(candidate).top_domain_under_public_suffix
            if reg:
                return reg
    return None


@dataclass
class ScreenUnit:
    """One merged screen unit (spec §3). unit_index counts units within the
    trajectory BEFORE any unit is dropped, so rows stay traceable to raw steps."""
    unit_index: int
    tab_id: int | None
    created_time: int | None   # LAST merged step's createdTime
    img_ref: str | None        # FIRST merged step's img
    href: str | None           # FIRST merged step's href
    axtree_ref: str | None     # FIRST merged step's axTree
    host: str | None
    action_type: str | None
    viewport_w: int | float | None
    viewport_h: int | float | None
    target_w: int | float | None    # FIRST step's rect.width (click target)
    target_h: int | float | None
    n_steps_merged: int


def merge_screen_units(steps: list[Step]) -> list[ScreenUnit]:
    """Spec §1–§3: drop launchApp, then merge consecutive identical-img,
    same-tab actionable steps into screen units, preserving file order."""
    actionable = [s for s in steps if not s.is_launch_app]

    groups: list[list[Step]] = []
    for s in actionable:
        prev = groups[-1][-1] if groups else None
        if (
            prev is not None
            and s.img and prev.img and s.img == prev.img
            and s.tab_id == prev.tab_id
        ):
            groups[-1].append(s)
        else:
            groups.append([s])

    units = []
    for i, g in enumerate(groups):
        first = g[0]
        # last step with a createdTime (None if none — handled by caller)
        created = next(
            (s.created_time for s in reversed(g) if s.created_time is not None),
            None,
        )
        vp = first.viewport or {}
        rect = first.rect or {}
        units.append(ScreenUnit(
            unit_index=i,
            tab_id=first.tab_id,
            created_time=created,
            img_ref=first.img,
            href=first.href,
            axtree_ref=first.ax_tree,
            host=first.host,
            action_type=first.type,
            viewport_w=vp.get("width"),
            viewport_h=vp.get("height"),
            target_w=rect.get("width"),
            target_h=rect.get("height"),
            n_steps_merged=len(g),
        ))
    return units


def rows_for_trajectory(
    traj: Trajectory,
    min_dwell_s: float = MIN_DWELL_S,
    max_dwell_s: float = MAX_DWELL_S,
) -> list[dict]:
    """Spec §4–§7: per-tab predecessor chains → dwell rows, hard-filtered."""
    units = merge_screen_units(traj.steps)

    rows = []
    prev_in_tab: dict[int | None, ScreenUnit] = {}
    for unit in units:
        if unit.created_time is None:
            continue   # cannot anchor a dwell on either side
        prev = prev_in_tab.get(unit.tab_id)
        prev_in_tab[unit.tab_id] = unit
        if prev is None:
            continue   # first unit in tab: predecessor only, not a target

        dwell_s = (unit.created_time - prev.created_time) / 1000.0
        if not (min_dwell_s < dwell_s <= max_dwell_s):
            continue

        rows.append({
            "trajectory_id": traj.id,
            "task_title": traj.title,
            "tab_id": unit.tab_id,
            "unit_index": unit.unit_index,
            "img_ref": unit.img_ref,
            "href": unit.href,
            "prev_href": prev.href,
            "axtree_ref": unit.axtree_ref,
            "dwell_s_raw": dwell_s,
            "is_navigation": unit.href != prev.href,
            "host": unit.host,
            "registrable_domain": registrable_domain(unit.host, unit.href),
            "viewport_w": unit.viewport_w,
            "viewport_h": unit.viewport_h,
            "action_type": unit.action_type,
            "target_w": unit.target_w,
            "target_h": unit.target_h,
            "n_steps_merged": unit.n_steps_merged,
        })
    return rows


def build_rows(
    trajectories: Iterable[Trajectory],
    min_dwell_s: float = MIN_DWELL_S,
    max_dwell_s: float = MAX_DWELL_S,
) -> pd.DataFrame:
    """Build the full label DataFrame (columns = ROW_COLUMNS, fixed order)."""
    all_rows: list[dict] = []
    for traj in trajectories:
        all_rows.extend(rows_for_trajectory(traj, min_dwell_s, max_dwell_s))
    return pd.DataFrame(all_rows, columns=ROW_COLUMNS)
