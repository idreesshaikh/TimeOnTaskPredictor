"""
Disk/time-bounded screenshot resolution: the trajectory-budget sampler
(totvlm.splits.sample_trajectory_rows) and store-time JPEG recompression
(totvlm.images resolve path). Both keep the full download (~182k images,
~122 GB) off the table while guaranteeing sampled tasks have no missing
screens.
"""
from __future__ import annotations

import io

import httpx
import pandas as pd
from PIL import Image

from totvlm.images import cache_key, cache_path, find_cached, resolve_refs
from totvlm.splits import sample_trajectory_rows

# ── Trajectory-budget sampler ─────────────────────────────────────────────────

def _rows(spec: dict[str, dict[str, int]]) -> pd.DataFrame:
    """spec: split → {trajectory_id: n_rows}."""
    records = [
        {"split": split, "trajectory_id": tid, "step": i}
        for split, trajs in spec.items()
        for tid, n in trajs.items()
        for i in range(n)
    ]
    return pd.DataFrame(records)


def test_takes_whole_trajectories_until_budget():
    df = _rows({"train": {"t1": 4, "t2": 4, "t3": 4}})
    mask = sample_trajectory_rows(df, {"train": 6}, seed=42)
    picked = df[mask].groupby("trajectory_id").size()
    # budget 6 with 4-row tasks → exactly two complete trajectories (overshoot
    # to finish the last one, never a partial task)
    assert sorted(picked.tolist()) == [4, 4]
    assert set(picked.index) <= {"t1", "t2", "t3"}


def test_prefix_stable_growing_budget_only_adds():
    df = _rows({"train": {f"t{i}": 3 for i in range(20)}})
    small = set(df[sample_trajectory_rows(df, {"train": 9}, seed=42)]
                ["trajectory_id"])
    large = set(df[sample_trajectory_rows(df, {"train": 21}, seed=42)]
                ["trajectory_id"])
    assert small < large            # strict superset: cache always reused


def test_splits_sampled_independently_and_unbudgeted_ignored():
    df = _rows({"train": {"a": 2}, "val": {"b": 2}, "test": {"c": 2}})
    mask = sample_trajectory_rows(df, {"train": 2, "test": 2}, seed=42)
    assert set(df[mask]["split"]) == {"train", "test"}   # no val budget → none


def test_straddling_trajectory_sampled_per_split():
    # the same trajectory id appearing in two splits = two independent units
    df = _rows({"train": {"x": 3}, "test": {"x": 2}})
    mask = sample_trajectory_rows(df, {"train": 3}, seed=42)
    assert df[mask]["split"].unique().tolist() == ["train"]
    assert mask.sum() == 3          # test-side rows of "x" not dragged in


# ── Store-time recompression ──────────────────────────────────────────────────

URL = "https://data.imean.tech/uploads/big.png"


def _png_bytes(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _client(payload: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, content=payload))
    )


def test_store_downscales_and_saves_jpeg(tmp_path):
    out = resolve_refs([URL], cache_dir=tmp_path, client=_client(
        _png_bytes(2000, 1200)), store={"max_side": 1280, "jpeg_quality": 85})
    path = out[URL]
    assert path is not None and path.endswith(".jpg")
    img = Image.open(path)
    assert img.format == "JPEG"
    assert max(img.size) <= 1280
    assert find_cached(URL, tmp_path).suffix == ".jpg"


def test_existing_original_cache_is_reused_not_recompressed(tmp_path):
    original = cache_path(URL, tmp_path)     # legacy .png entry
    original.write_bytes(_png_bytes(50, 40))
    out = resolve_refs([URL], cache_dir=tmp_path, client=_client(b"unused"),
                       store={"max_side": 1280, "jpeg_quality": 85})
    assert out[URL] == str(original)         # cache hit, no re-download
    assert find_cached(URL, tmp_path) == original


def test_no_store_keeps_original_bytes(tmp_path):
    payload = _png_bytes(30, 30)
    out = resolve_refs([URL], cache_dir=tmp_path, client=_client(payload))
    path = out[URL]
    assert path.endswith(".png")
    assert cache_key(URL) in path
    assert open(path, "rb").read() == payload
