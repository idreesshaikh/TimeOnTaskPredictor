"""Domain-disjoint split assignment + train-only winsorized target."""
import json

import numpy as np
import pandas as pd
import pytest

from totvlm.splits import (
    add_split_column,
    add_targets,
    assert_disjoint,
    assign_domain_splits,
    dataset_card_stats,
    load_or_create_splits,
    train_winsor_cap,
)

DOMAINS = [f"site{i}.com" for i in range(100)]


def test_assignment_is_deterministic_and_disjoint():
    a = assign_domain_splits(DOMAINS, seed=42)
    b = assign_domain_splits(list(reversed(DOMAINS)), seed=42)  # order-free
    assert a == b
    assert set(a) == set(DOMAINS)
    assert_disjoint(a)


def test_fractions_by_domain_count():
    a = assign_domain_splits(DOMAINS, seed=42)
    counts = pd.Series(a).value_counts()
    assert counts["train"] == 70 and counts["val"] == 15 and counts["test"] == 15


def test_different_seed_changes_assignment():
    assert assign_domain_splits(DOMAINS, seed=42) != assign_domain_splits(
        DOMAINS, seed=43
    )


def test_assert_disjoint_rejects_unknown_split():
    with pytest.raises(AssertionError):
        assert_disjoint({"a.com": "train", "b.com": "holdout"})


def test_saved_assignment_is_reused_not_recomputed(tmp_path):
    path = tmp_path / "splits.json"
    first = load_or_create_splits(DOMAINS, path=path, seed=42)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["seed"] == 42
    assert payload["n_domains"] == len(DOMAINS)
    # different domain list, same file → saved assignment wins
    again = load_or_create_splits(["other.org"], path=path, seed=42)
    assert again == first


def test_add_split_column_maps_and_flags_missing():
    df = pd.DataFrame({"registrable_domain": ["a.com", "b.com", None, "x.org"]})
    out = add_split_column(df, {"a.com": "train", "b.com": "test"})
    assert list(out["split"][:2]) == ["train", "test"]
    assert out["split"][2:].isna().all()


def _target_df():
    return pd.DataFrame({
        "trajectory_id": ["t"] * 8,
        "registrable_domain": ["d.com"] * 8,
        "is_navigation": [True, False] * 4,
        "split": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
        # train dwells: 1..4 → p95 close to 4; test has a 100s outlier
        "dwell_s_raw": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 2.0],
    })


def test_winsor_cap_uses_train_split_only():
    df = _target_df()
    cap = train_winsor_cap(df, percentile=0.95)
    assert cap == pytest.approx(3.85)          # p95 of [1,2,3,4], not of all rows
    with pytest.raises(ValueError):
        train_winsor_cap(df[df["split"] != "train"], percentile=0.95)


def test_add_targets_caps_all_splits_and_keeps_raw():
    df = add_targets(_target_df(), cap=3.85)
    assert df["dwell_s"].max() == pytest.approx(3.85)   # 100s outlier capped
    assert df["dwell_s_raw"].max() == pytest.approx(100.0)  # raw untouched
    assert np.allclose(df["y"], np.log1p(df["dwell_s"]))


def test_dataset_card_stats_shape():
    df = add_targets(_target_df(), cap=3.85)
    stats = dataset_card_stats(df, cap=3.85, percentile=0.95)
    assert stats["n_rows"] == 8
    assert stats["per_split"]["train"]["rows"] == 4
    assert stats["per_split"]["test"]["pct_capped_by_winsor"] == 50.0
    assert stats["winsor"]["cap_dwell_s"] == 3.85
