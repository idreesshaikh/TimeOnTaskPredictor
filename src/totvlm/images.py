"""
totvlm/images.py
================
Resolve row screenshots (`img_ref`) to cached local files + model-ready loader.

Resolution rules
----------------
- http(s) ref  → download once to data/images_cache/<sha1(url)>.png
  (httpx + tenacity backoff, bounded concurrency, failures logged NOT fatal,
  skip when already cached — reruns are cheap and resumable).
- opaque hash  → look for <hash>.<ext> under a local images root (a directory
  you synced separately); absent → unresolved.
- Rows whose ref does not resolve are kept and flagged (`img_resolved=False`);
  they are excluded at TRAIN time, not at build time.

Loader
------
`load_image()` opens → RGB → downscales so the long side ≤ `max_side`
(default 1280) and optionally clamps total pixels into [min_pixels,
max_pixels], bounding Qwen3-VL visual tokens (28 px effective patch grid
after 2×2 merge — see `estimate_visual_tokens`).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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


# ── Cache keys ────────────────────────────────────────────────────────────────

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


# ── Download one URL (retry w/ backoff; caller treats failure as non-fatal) ──

class _HTTPStatusError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    retry=retry_if_exception_type((httpx.TransportError, _HTTPStatusError)),
    reraise=True,
)
def _fetch_bytes(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url)
    if resp.status_code != 200:
        raise _HTTPStatusError(f"HTTP {resp.status_code} for {url}")
    return resp.content


def _download_one(
    client: httpx.Client, url: str, cache_dir: Path,
    store: dict | None = None,
) -> Path | None:
    """Fetch one URL into the cache. Returns the path, or None on failure.

    `store` (optional, from configs/data.yaml `resolve.store`): downscale to
    `max_side` and re-encode as JPEG `jpeg_quality` at download time — the
    training pipeline downscales anyway, so full-resolution PNGs only burn
    disk (~3× savings). Already-cached files are used as-is either way.
    """
    dest = find_cached(url, cache_dir)
    if dest:
        return dest
    try:
        data = _fetch_bytes(client, url)
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
    tmp.rename(dest)          # atomic-ish: never leave half-written files
    return dest


# ── Hash refs: look under a local images root ────────────────────────────────

def find_hash_image(ref: str, images_root: str | Path = IMAGES_ROOT) -> Path | None:
    root = Path(images_root)
    for ext in ("",) + HASH_EXTS:
        p = root / f"{ref}{ext}"
        if p.is_file():
            return p
    return None


# ── Bulk resolver ─────────────────────────────────────────────────────────────

def resolve_refs(
    refs: Iterable[str],
    cache_dir: str | Path = CACHE_DIR,
    images_root: str | Path = IMAGES_ROOT,
    concurrency: int = CONCURRENCY,
    timeout_s: float = TIMEOUT_S,
    client: httpx.Client | None = None,
    store: dict | None = None,
) -> dict[str, str | None]:
    """
    Resolve unique refs → local path (str) or None. Never raises on a bad ref.
    Pass `client` to inject a mock transport in tests.
    """
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

    own_client = client is None
    if own_client:
        client = httpx.Client(
            timeout=timeout_s, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (research)"},
        )
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_download_one, client, u, cache_dir, store): u
                for u in url_refs
            }
            done = 0
            for fut in as_completed(futures):
                url = futures[fut]
                path = fut.result()
                out[url] = str(path) if path else None
                done += 1
                if done % 1000 == 0:
                    log.info(f"resolved {done}/{len(url_refs)} URLs ...")
    finally:
        if own_client:
            client.close()
    return out


# ── Model-ready loader ───────────────────────────────────────────────────────

def fit_size(
    width: int,
    height: int,
    max_side: int = MAX_SIDE,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """
    Target (width, height) after bounding: long side ≤ max_side, then total
    pixels clamped into [min_pixels, max_pixels]. Aspect ratio preserved;
    never returns a dimension < 1.
    """
    scale = min(1.0, max_side / max(width, height))
    if max_pixels is not None and (width * scale) * (height * scale) > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / (width * height)))
    if min_pixels is not None and (width * scale) * (height * scale) < min_pixels:
        scale = max(scale, math.sqrt(min_pixels / (width * height)))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    if max_pixels is not None and w * h > max_pixels:
        # rounding can push a few pixels past the cap — max_pixels is a hard
        # budget (visual tokens / VRAM), so floor instead
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


# ── Resolution report ────────────────────────────────────────────────────────

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
