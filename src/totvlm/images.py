"""Resolve row screenshots (`img_ref`) to cached local files + model-ready
loader. URL refs download once to data/images_cache/<sha1(url)>; hash refs
resolve under a local images root. Unresolved rows are kept and flagged
(`img_resolved=False`) — excluded at train time, not build time."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from collections.abc import Iterable
from pathlib import Path

import httpx
from PIL import Image

from totvlm.fetch import fetch_all, fetch_bytes

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/images_cache")
IMAGES_ROOT = Path("data/images_root")   # where opaque-hash images would live
MAX_SIDE = 1280                          # long-side bound for training images
PATCH = 28            # Qwen3-VL effective grid: 14 px patches, 2x2 token merge
TIMEOUT_S = 15.0
CONCURRENCY = 8

HASH_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def is_http_ref(ref: str | None) -> bool:
    return isinstance(ref, str) and ref.startswith(("https://", "http://"))


def cache_key(url: str) -> str:
    """Deterministic cache key: sha1 of the exact URL string."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def cache_path(url: str, cache_dir: str | Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{cache_key(url)}.png"


CACHE_EXTS = (".png", ".jpg")   # .png = original bytes; .jpg = recompressed


def find_cached(url: str, cache_dir: str | Path = CACHE_DIR) -> Path | None:
    """Cached file for a URL in either storage format, or None."""
    stem = Path(cache_dir) / cache_key(url)
    for ext in CACHE_EXTS:
        p = stem.with_suffix(ext)
        if p.exists():
            return p
    return None


def _download_one(
    client: httpx.Client, url: str, cache_dir: Path,
    store: dict | None = None,
) -> Path | None:
    """Fetch one URL into the cache; None on failure. `store` (configs/
    data.yaml resolve.store) recompresses to JPEG at download time (~3× disk
    savings — training downscales anyway)."""
    dest = find_cached(url, cache_dir)
    if dest:
        return dest
    try:
        data = fetch_bytes(client, url)
        if store:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((store["max_side"], store["max_side"]))
            dest = cache_path(url, cache_dir).with_suffix(".jpg")
            tmp = dest.with_suffix(".tmp")
            img.save(tmp, format="JPEG", quality=store["jpeg_quality"])
        else:
            dest = cache_path(url, cache_dir)
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
    except Exception as e:
        log.warning(f"image fetch failed: {url} ({e})")
        return None
    tmp.rename(dest)          # never leave half-written files
    return dest


def find_hash_image(ref: str, images_root: str | Path = IMAGES_ROOT) -> Path | None:
    root = Path(images_root)
    for ext in ("",) + HASH_EXTS:
        p = root / f"{ref}{ext}"
        if p.is_file():
            return p
    return None


def resolve_refs(
    refs: Iterable[str],
    cache_dir: str | Path = CACHE_DIR,
    images_root: str | Path = IMAGES_ROOT,
    concurrency: int = CONCURRENCY,
    timeout_s: float = TIMEOUT_S,
    client: httpx.Client | None = None,
    store: dict | None = None,
) -> dict[str, str | None]:
    """Resolve unique refs → local path (str) or None. Never raises on a bad
    ref. Pass `client` to inject a mock transport in tests."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    refs = [r for r in dict.fromkeys(refs) if isinstance(r, str) and r]

    out: dict[str, str | None] = {}
    url_refs = []
    for r in refs:
        if is_http_ref(r):
            url_refs.append(r)
        else:
            p = find_hash_image(r, images_root)
            out[r] = str(p) if p else None

    downloaded = fetch_all(
        url_refs,
        lambda c, u: _download_one(c, u, cache_dir, store),
        concurrency=concurrency,
        timeout_s=timeout_s,
        client=client,
        progress_label="resolved",
    )
    out.update({u: str(p) if p else None for u, p in downloaded.items()})
    return out


def fit_size(
    width: int,
    height: int,
    max_side: int = MAX_SIDE,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """Target (w, h): long side ≤ max_side, total pixels clamped into
    [min_pixels, max_pixels]. Aspect preserved; never returns < 1."""
    scale = min(1.0, max_side / max(width, height))
    if max_pixels is not None and (width * scale) * (height * scale) > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / (width * height)))
    if min_pixels is not None and (width * scale) * (height * scale) < min_pixels:
        scale = max(scale, math.sqrt(min_pixels / (width * height)))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    if max_pixels is not None and w * h > max_pixels:
        # max_pixels is a hard VRAM budget — floor when rounding overshoots
        w, h = max(1, math.floor(width * scale)), max(1, math.floor(height * scale))
    return w, h


def estimate_visual_tokens(width: int, height: int, patch: int = PATCH) -> int:
    """Qwen3-VL visual tokens ≈ cells of the effective 28 px grid
    (14 px patches merged 2×2)."""
    return math.ceil(width / patch) * math.ceil(height / patch)


def load_image(
    path: str | Path,
    max_side: int = MAX_SIDE,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Image.Image:
    """Open → RGB → resize per fit_size. LANCZOS for downscale quality."""
    img = Image.open(path).convert("RGB")
    target = fit_size(img.width, img.height, max_side, min_pixels, max_pixels)
    if target != (img.width, img.height):
        img = img.resize(target, Image.LANCZOS)
    return img


def write_resolution_report(stats: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image resolution report",
        "",
        f"- Rows: **{stats['n_rows']}** · resolved (trainable): "
        f"**{stats['n_rows_resolved']}** (**{stats['pct_rows_resolved']}%**)",
        f"- Unique refs: {stats['n_refs']} (URL {stats['n_url_refs']} / "
        f"hash {stats['n_hash_refs']})",
        f"- URL refs resolved: {stats['n_url_resolved']} · "
        f"hash refs found locally: {stats['n_hash_resolved']}",
    ]
    if stats.get("note"):
        lines += ["", f"> {stats['note']}"]
    lines += ["", "```json", json.dumps(stats, indent=2), "```", ""]
    out_path.write_text("\n".join(lines))
