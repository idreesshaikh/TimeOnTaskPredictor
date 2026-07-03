"""
Parser unit tests: strict (training hooks) and lenient (evaluation) decoding
of `dwell_seconds: X.X` on clean, noisy, and malformed model outputs.
The lenient tiers and the never-hide-failures contract are what
scripts/evaluate.py relies on.
"""
from __future__ import annotations

import pytest

from totvlm.data import (
    PARSE_TIERS,
    parse_dwell_output,
    parse_dwell_output_lenient,
)

# ── Strict parser (train-time val hook: exact trained format only) ───────────

@pytest.mark.parametrize("text,expected", [
    ("dwell_seconds: 12.3", 12.3),
    ("dwell_seconds: 0.1", 0.1),
    ("  dwell_seconds:  0.8  ", 0.8),      # whitespace tolerated
    ("dwell_seconds: 12", 12.0),           # integer still parses
    ("dwell_seconds: 12.34", 12.34),       # 2 dp still parses
])
def test_strict_accepts_trained_format(text, expected):
    assert parse_dwell_output(text) == expected


@pytest.mark.parametrize("text", [
    "dwell seconds: 12.3",                 # wrong key
    "dwell_seconds: -3.0",                 # negative never emitted
    "dwell_seconds: 12.3 seconds",         # trailing junk
    "I think dwell_seconds: 12.3",         # leading junk
    "dwell_seconds:",                      # no number
    "12.3",                                # bare number
    "",
    None,
])
def test_strict_rejects_everything_else(text):
    assert parse_dwell_output(text) is None


# ── Lenient parser (eval decoding: tiered, failures visible) ─────────────────

@pytest.mark.parametrize("text,value,tier", [
    # clean → strict tier
    ("dwell_seconds: 12.3", 12.3, "strict"),
    # noisy but labeled → labeled tier
    ("dwell_seconds: 12.3 seconds", 12.3, "labeled"),
    ("Sure! dwell_seconds: 4.7", 4.7, "labeled"),
    ("The answer is dwell_seconds: 8.0.", 8.0, "labeled"),
    ("DWELL_SECONDS: 3.2", 3.2, "labeled"),        # case-insensitive
    ("dwell seconds: 5.5", 5.5, "labeled"),        # space for underscore
    ("dwell_seconds = 2.0", 2.0, "labeled"),       # = for :
    ("dwell_second: 9.1", 9.1, "labeled"),         # singular
    # no label at all → first bare number
    ("12.3", 12.3, "bare_number"),
    ("about 7 seconds", 7.0, "bare_number"),
    ("I estimate the user spends 4.5s here", 4.5, "bare_number"),
    # malformed → fail
    ("dwell_seconds: -3.0", None, "fail"),         # negative dwell
    ("no idea", None, "fail"),
    ("", None, "fail"),
    (None, None, "fail"),
])
def test_lenient_tiers(text, value, tier):
    assert parse_dwell_output_lenient(text) == (value, tier)


def test_lenient_labeled_wins_over_earlier_bare_number():
    # the labeled value is the prediction, not incidental numbers around it
    v, tier = parse_dwell_output_lenient(
        "screen 3 of 10 -> dwell_seconds: 6.4")
    assert (v, tier) == (6.4, "labeled")


def test_lenient_never_returns_value_on_fail():
    for text in ("dwell_seconds: -1", "n/a", ""):
        v, tier = parse_dwell_output_lenient(text)
        assert v is None and tier == "fail"


def test_tier_vocabulary_is_closed():
    # evaluate.py counts by these exact names — keep them in sync
    assert PARSE_TIERS == ("strict", "labeled", "bare_number", "fail")
    for text in ("dwell_seconds: 1.0", "x dwell_seconds: 1.0", "1.0", "zzz"):
        assert parse_dwell_output_lenient(text)[1] in PARSE_TIERS
