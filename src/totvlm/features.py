"""
totvlm/features.py
==================
Interpretable no-image features for the LightGBM baseline.

Two sources:
1. AX tree (needs one fetch per row): element/interactive/link/button/input
   counts + visible-text length. Trees average ~1.2 MB, and the corpus has
   ~182k unique tree URLs (~210 GB) — far too large to mirror. So we fetch a
   tree, extract the tiny feature vector, CACHE ONLY THE FEATURES
   (data/axtree_features_cache/<sha1(url)>.json), and discard the tree.
   Reruns are cheap and resumable; failures are logged, not cached, so a
   rerun retries them.
2. Row columns already on disk (free): click-target area (rect w*h),
   action_type one-hot, is_navigation, unit_index.

Rows without an axTree URL — or whose fetch/parse fails — are EXCLUDED from
the baseline; the entrypoint reports how many (never silently).
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import httpx
import pandas as pd

from totvlm.fetch import fetch_all, fetch_bytes

log = logging.getLogger(__name__)

# ── AX tree vocabulary (roles are ARIA-ish; html_tag disambiguates) ──────────
INTERACTIVE_ROLES = {
    "button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio",
    "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "option",
    "spinbutton", "switch", "slider", "treeitem", "listbox",
}
LINK_ROLES = {"link"}
LINK_TAGS = {"a"}
BUTTON_ROLES = {"button"}
BUTTON_TAGS = {"button"}
INPUT_ROLES = {
    "textbox", "searchbox", "combobox", "checkbox", "radio", "spinbutton",
    "switch", "slider", "listbox",
}
INPUT_TAGS = {"input", "textarea", "select"}

AXTREE_FEATURE_NAMES = (
    "ax_n_nodes", "ax_n_interactive", "ax_n_links", "ax_n_buttons",
    "ax_n_inputs", "ax_text_len",
)


def extract_axtree_features(tree, max_depth: int = 80) -> dict[str, int]:
    """Walk one tree (dict, or list of root dicts) and count what we keep.
    Visible-text length = total `name` length over LEAF nodes only — container
    names repeat their children's text, so counting every node double-counts."""
    counts = dict.fromkeys(AXTREE_FEATURE_NAMES, 0)
    roots = tree if isinstance(tree, list) else [tree]
    # Iterative DFS with explicit depth guard (trees can be deep and cyclic-ish
    # data would otherwise recurse forever).
    stack = [(n, 0) for n in roots if isinstance(n, dict)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            continue
        counts["ax_n_nodes"] += 1
        role = (node.get("role") or "").lower()
        tag = ((node.get("attributes") or {}).get("html_tag") or "").lower()
        if role in INTERACTIVE_ROLES or tag in (
            LINK_TAGS | BUTTON_TAGS | INPUT_TAGS
        ):
            counts["ax_n_interactive"] += 1
        if role in LINK_ROLES or tag in LINK_TAGS:
            counts["ax_n_links"] += 1
        if role in BUTTON_ROLES or tag in BUTTON_TAGS:
            counts["ax_n_buttons"] += 1
        if role in INPUT_ROLES or tag in INPUT_TAGS:
            counts["ax_n_inputs"] += 1
        children = node.get("children") or []
        if not children:
            counts["ax_text_len"] += len(node.get("name") or "")
        stack.extend(
            (c, depth + 1) for c in children if isinstance(c, dict)
        )
    return counts


# ── Fetch + feature cache (features only — never the tree) ───────────────────

def feature_cache_path(url: str, cache_dir: str | Path) -> Path:
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{key}.json"


def _features_for_url(
    client: httpx.Client, url: str, cache_dir: Path, max_depth: int
) -> dict[str, int] | None:
    """Cached features, or fetch tree → extract → cache. None on failure."""
    dest = feature_cache_path(url, cache_dir)
    if dest.exists():
        try:
            return json.loads(dest.read_text())
        except json.JSONDecodeError:
            dest.unlink()  # half-written cache entry: refetch
    try:
        tree = json.loads(fetch_bytes(client, url))
        feats = extract_axtree_features(tree, max_depth=max_depth)
    except Exception as e:
        log.warning(f"axtree fetch/parse failed: {url} ({e})")
        return None
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(feats))
    tmp.rename(dest)
    return feats


def fetch_axtree_features(
    urls: Iterable[str],
    cache_dir: str | Path,
    concurrency: int = 16,
    timeout_s: float = 20.0,
    max_depth: int = 80,
    client: httpx.Client | None = None,
) -> dict[str, dict[str, int] | None]:
    """Resolve unique axTree URLs → feature dict (or None). Resumable via the
    on-disk feature cache. Pass `client` to inject a mock transport in tests."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return fetch_all(
        urls,
        lambda c, u: _features_for_url(c, u, cache_dir, max_depth),
        concurrency=concurrency,
        timeout_s=timeout_s,
        client=client,
        progress_label="axtree features",
    )


# ── Feature frame assembly ────────────────────────────────────────────────────

def build_feature_frame(
    df: pd.DataFrame,
    axtree_features: dict[str, dict[str, int] | None],
    action_vocab: list[str],
) -> pd.DataFrame:
    """Assemble the baseline design matrix for rows whose axTree resolved.
    Returns df rows (same index) + feature columns; caller handles exclusions
    by checking `ax_resolved`."""
    df = df.copy()
    feats = df["axtree_ref"].map(
        lambda u: axtree_features.get(u) if isinstance(u, str) else None
    )
    df["ax_resolved"] = feats.notna()
    for name in AXTREE_FEATURE_NAMES:
        df[name] = feats.map(lambda f, n=name: f[n] if f else None)

    df["click_target_area"] = df["target_w"] * df["target_h"]
    df["f_is_navigation"] = df["is_navigation"].astype(int)
    df["f_unit_index"] = df["unit_index"]

    action = df["action_type"].fillna("").str.lower()
    for a in action_vocab:
        df[f"action_{a}"] = (action == a).astype(int)
    df["action_other"] = (~action.isin(action_vocab)).astype(int)
    return df


def feature_columns(action_vocab: list[str]) -> list[str]:
    """Column order of the design matrix (single source of truth)."""
    return (
        list(AXTREE_FEATURE_NAMES)
        + ["click_target_area", "f_is_navigation", "f_unit_index"]
        + [f"action_{a}" for a in action_vocab]
        + ["action_other"]
    )
