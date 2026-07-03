"""
Unit tests for totvlm.images: cache-key determinism, downscale/token math,
and the resolver with a mocked network (httpx.MockTransport — no real fetch).
"""
import httpx
import pytest
import tenacity
from PIL import Image

from totvlm import images
from totvlm.images import (
    cache_key,
    cache_path,
    estimate_visual_tokens,
    fit_size,
    load_image,
    resolve_refs,
)

URL_OK = "https://data.imean.tech/uploads/ok.png"
URL_BAD = "https://data.imean.tech/uploads/missing.png"


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """Keep retry/backoff logic but drop the sleeps so tests stay fast."""
    monkeypatch.setattr(images._fetch_bytes.retry, "wait", tenacity.wait_fixed(0))


# ── Cache keys ────────────────────────────────────────────────────────────────

def test_cache_key_deterministic_and_distinct(tmp_path):
    assert cache_key(URL_OK) == cache_key(URL_OK)
    assert cache_key(URL_OK) != cache_key(URL_BAD)
    p = cache_path(URL_OK, tmp_path)
    assert p.parent == tmp_path and p.suffix == ".png"
    assert p.stem == cache_key(URL_OK)
    int(cache_key(URL_OK), 16)   # sha1 hex, 40 chars
    assert len(cache_key(URL_OK)) == 40


# ── Downscale / token math ────────────────────────────────────────────────────

def test_fit_size_bounds_long_side_and_keeps_aspect():
    assert fit_size(2560, 1440, max_side=1280) == (1280, 720)
    assert fit_size(1440, 2560, max_side=1280) == (720, 1280)
    # already small → untouched (no upscaling without min_pixels)
    assert fit_size(800, 600, max_side=1280) == (800, 600)


def test_fit_size_pixel_clamps():
    w, h = fit_size(2000, 1000, max_side=4096, max_pixels=500_000)
    assert w * h <= 500_000 + w + h          # rounding slack of < one row/col
    assert abs(w / h - 2.0) < 0.01
    w, h = fit_size(100, 50, max_side=4096, min_pixels=80_000)
    assert w * h >= 80_000 - (w + h)
    assert fit_size(1, 1, max_side=1280) == (1, 1)


def test_estimate_visual_tokens():
    # 28 px effective grid: 1280x960 → ceil(1280/28)*ceil(960/28) = 46*35
    assert estimate_visual_tokens(1280, 960) == 46 * 35
    assert estimate_visual_tokens(28, 28) == 1
    assert estimate_visual_tokens(29, 28) == 2
    # bounding the long side to 1280 keeps tokens comfortably bounded
    w, h = fit_size(3840, 2160, max_side=1280)
    assert estimate_visual_tokens(w, h) <= 46 * 46


def test_load_image_downscales(tmp_path):
    src = tmp_path / "big.png"
    Image.new("RGB", (2000, 1500), "red").save(src)
    img = load_image(src, max_side=1280)
    assert (img.width, img.height) == (1280, 960)
    assert img.mode == "RGB"
    img2 = load_image(src, max_side=4096, max_pixels=280 * 210)
    assert img2.width * img2.height <= 280 * 210 + img2.width + img2.height


# ── Resolver with mocked network ─────────────────────────────────────────────

def _png_bytes() -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(buf, format="PNG")
    return buf.getvalue()


def _mock_client(calls: list) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url) == URL_OK:
            return httpx.Response(200, content=_png_bytes())
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_resolve_refs_mixed(tmp_path):
    cache = tmp_path / "cache"
    root = tmp_path / "root"
    root.mkdir()
    (root / "abc123hash.png").write_bytes(_png_bytes())

    calls: list = []
    out = resolve_refs(
        [URL_OK, URL_BAD, "abc123hash", "missinghash", None, ""],
        cache_dir=cache, images_root=root, client=_mock_client(calls),
    )
    assert out[URL_OK] == str(cache_path(URL_OK, cache))
    assert cache_path(URL_OK, cache).read_bytes() == _png_bytes()
    assert out[URL_BAD] is None            # failure logged, not fatal
    assert calls.count(URL_BAD) == 3       # retried with backoff
    assert out["abc123hash"] == str(root / "abc123hash.png")
    assert out["missinghash"] is None
    assert None not in out and "" not in out


def test_resolve_refs_skips_cached(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_path(URL_OK, cache).write_bytes(b"already-here")

    calls: list = []
    out = resolve_refs([URL_OK], cache_dir=cache, client=_mock_client(calls))
    assert calls == []                     # no network hit for cached file
    assert out[URL_OK] == str(cache_path(URL_OK, cache))
    assert cache_path(URL_OK, cache).read_bytes() == b"already-here"
