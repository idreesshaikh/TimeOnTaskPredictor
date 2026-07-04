"""Shared HTTP fetch scaffolding for the resumable caches (images.py,
features.py): one retry policy, one client recipe, one bounded fan-out."""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


class _HTTPStatusError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    retry=retry_if_exception_type((httpx.TransportError, _HTTPStatusError)),
    reraise=True,
)
def fetch_bytes(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url)
    if resp.status_code != 200:
        raise _HTTPStatusError(f"HTTP {resp.status_code} for {url}")
    return resp.content


def fetch_all[T](
    urls: Iterable[str],
    fetch_one: Callable[[httpx.Client, str], T],
    *,
    concurrency: int,
    timeout_s: float,
    client: httpx.Client | None = None,
    progress_label: str = "fetched",
) -> dict[str, T]:
    """Fan `fetch_one` out over unique urls with a shared client; returns
    {url: result}. Pass `client` to inject a mock transport in tests."""
    urls = [u for u in dict.fromkeys(urls) if isinstance(u, str) and u]
    own_client = client is None
    if own_client:
        client = httpx.Client(
            timeout=timeout_s, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (research)"},
        )
    out: dict[str, T] = {}
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(fetch_one, client, u): u for u in urls}
            done = 0
            for fut in as_completed(futures):
                out[futures[fut]] = fut.result()
                done += 1
                if done % 1000 == 0:
                    log.info(f"{progress_label} {done}/{len(urls)} ...")
    finally:
        if own_client:
            client.close()
    return out
