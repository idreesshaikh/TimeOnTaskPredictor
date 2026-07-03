"""axTree feature extraction + baseline design-matrix assembly."""
import json

import httpx
import pandas as pd

from totvlm.features import (
    AXTREE_FEATURE_NAMES,
    build_feature_frame,
    extract_axtree_features,
    feature_cache_path,
    feature_columns,
    fetch_axtree_features,
)

# Mirrors the real shape: ARIA-ish roles + attributes.html_tag + children.
TREE = {
    "role": "generic", "name": "", "attributes": {"html_tag": "html"},
    "children": [
        {"role": "link", "name": "Home", "attributes": {"html_tag": "a"},
         "children": []},
        {"role": "button", "name": "Buy now",
         "attributes": {"html_tag": "button"}, "children": []},
        {"role": "generic", "name": "", "attributes": {"html_tag": "div"},
         "children": [
             {"role": "textbox", "name": "Search",
              "attributes": {"html_tag": "input"}, "children": []},
             {"role": "paragraph", "name": "hello world",
              "attributes": {"html_tag": "p"}, "children": []},
         ]},
    ],
}


def test_extract_counts():
    f = extract_axtree_features(TREE)
    assert f["ax_n_nodes"] == 6
    assert f["ax_n_links"] == 1
    assert f["ax_n_buttons"] == 1
    assert f["ax_n_inputs"] == 1
    assert f["ax_n_interactive"] == 3
    # leaf names only: "Home" + "Buy now" + "Search" + "hello world"
    assert f["ax_text_len"] == len("Home") + len("Buy now") + len("Search") \
        + len("hello world")


def test_extract_handles_list_root_and_depth_guard():
    f = extract_axtree_features([TREE, TREE])
    assert f["ax_n_nodes"] == 12
    deep = {"role": "generic", "name": "x", "children": []}
    for _ in range(30):
        deep = {"role": "generic", "name": "", "children": [deep]}
    shallow = extract_axtree_features(deep, max_depth=5)
    assert shallow["ax_n_nodes"] == 6          # root depth 0 .. depth 5


def test_fetch_caches_features_only_and_skips_failures(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if "bad" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, content=json.dumps(TREE).encode())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    urls = ["https://x.test/ok.json", "https://x.test/bad.json"]
    out = fetch_axtree_features(urls, cache_dir=tmp_path, client=client)
    assert out["https://x.test/ok.json"]["ax_n_nodes"] == 6
    assert out["https://x.test/bad.json"] is None
    # success cached (features only, tiny), failure NOT cached
    ok_cache = feature_cache_path("https://x.test/ok.json", tmp_path)
    assert ok_cache.exists() and ok_cache.stat().st_size < 200
    assert not feature_cache_path("https://x.test/bad.json", tmp_path).exists()
    # second call: served from cache — no new request for the ok URL
    calls_before = calls["n"]
    out2 = fetch_axtree_features(urls[:1], cache_dir=tmp_path, client=client)
    assert out2 == {urls[0]: out[urls[0]]}
    assert calls["n"] == calls_before


def test_build_feature_frame_and_columns():
    vocab = ["click", "type"]
    df = pd.DataFrame({
        "axtree_ref": ["u1", "u2", None],
        "target_w": [10.0, 20.0, 5.0],
        "target_h": [4.0, 2.0, 1.0],
        "is_navigation": [True, False, True],
        "unit_index": [1, 2, 3],
        "action_type": ["click", "hover", "type"],
    })
    feats = {"u1": extract_axtree_features(TREE), "u2": None}
    out = build_feature_frame(df, feats, vocab)
    assert list(out["ax_resolved"]) == [True, False, False]
    assert out.loc[0, "click_target_area"] == 40.0
    assert list(out["f_is_navigation"]) == [1, 0, 1]
    assert list(out["action_click"]) == [1, 0, 0]
    assert list(out["action_other"]) == [0, 1, 0]   # hover not in vocab
    cols = feature_columns(vocab)
    assert set(AXTREE_FEATURE_NAMES) <= set(cols)
    assert set(cols) <= set(out.columns)
