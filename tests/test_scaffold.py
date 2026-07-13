"""v3 scaffold tests (CPU-only): the `ui:` stats line of the training target
— format, last-line parsing, prompt switching, per-row fallback, and the
fixed λ grid configs. This is the part of the redesign that must be correct
BEFORE burning GPU time."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from totvlm.config import load_config
from totvlm.data import (
    SCAFFOLD_FIELDS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SCAFFOLD,
    build_vlm_examples,
    format_scaffold_target,
    parse_dwell_output,
    parse_dwell_output_lenient,
)

CAP = 24.954
STATS = {"ax_n_nodes": 812, "ax_n_interactive": 56, "ax_n_links": 34,
         "ax_n_buttons": 12, "ax_n_inputs": 3, "ax_text_len": 5400}
UI_LINE = "ui: nodes=812 interactive=56 links=34 buttons=12 inputs=3 text=5400"


# ── target format ─────────────────────────────────────────────────────────────

def test_scaffold_target_is_two_lines_in_field_order():
    out = format_scaffold_target(STATS, 6.52, CAP)
    assert out == f"{UI_LINE}\ndwell_seconds: 6.5"


def test_scaffold_target_without_stats_falls_back_to_dwell_only():
    assert format_scaffold_target(None, 6.52, CAP) == "dwell_seconds: 6.5"


def test_scaffold_target_still_clips_to_winsor():
    assert format_scaffold_target(STATS, 600.0, CAP).splitlines()[-1] == \
        "dwell_seconds: 25.0"


# ── parsing: dwell read from the LAST line, both formats ─────────────────────

def test_strict_parse_reads_scaffolded_output():
    assert parse_dwell_output(f"{UI_LINE}\ndwell_seconds: 6.5") == 6.5


def test_strict_parse_still_reads_plain_output():
    assert parse_dwell_output("dwell_seconds: 6.5") == 6.5


def test_strict_parse_rejects_scaffold_without_dwell_line():
    assert parse_dwell_output(UI_LINE) is None


def test_strict_parse_rejects_dwell_before_scaffold():
    # dwell must be the LAST line — trained order is ui first
    assert parse_dwell_output(f"dwell_seconds: 6.5\n{UI_LINE}") is None


def test_lenient_parse_scores_scaffolded_output_as_strict():
    v, tier = parse_dwell_output_lenient(f"{UI_LINE}\ndwell_seconds: 6.5")
    assert (v, tier) == (6.5, "strict")


def test_lenient_parse_never_reads_ui_numbers_when_labeled_dwell_exists():
    v, tier = parse_dwell_output_lenient(
        f"{UI_LINE}\nthe dwell_seconds: 6.5 I think")
    assert v == 6.5 and tier == "labeled"


def test_roundtrip_scaffold():
    for v in (0.1, 4.7, 24.954):
        out = format_scaffold_target(STATS, v, CAP)
        assert parse_dwell_output(out) == round(v, 1)


# ── example builder integration ───────────────────────────────────────────────

@pytest.fixture
def rows(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"shot_{i}.png"
        Image.new("RGB", (800, 600), (10 * i, 20, 30)).save(p)
        paths.append(str(p))
    base = pd.DataFrame({
        "img_path": paths,
        "img_resolved": [True, True],
        "task_title": ["Find a flight", "Book a room"],
        "dwell_s": [7.697, 3.0],
        "is_navigation": [False, True],
    })
    # row 0 has stats, row 1 does not (NaN → dwell-only fallback)
    for _, col in SCAFFOLD_FIELDS:
        base[col] = [float(STATS[col]), float("nan")]
    return base


BUILD_KW = dict(winsor_cap=CAP, max_side=1024, min_pixels=100352,
                max_pixels=602112)


def test_builder_scaffold_targets_and_prompt(rows):
    examples = build_vlm_examples(rows, scaffold=True, **BUILD_KW)
    msgs0, msgs1 = examples[0]["messages"], examples[1]["messages"]
    assert msgs0[0]["content"][0]["text"] == SYSTEM_PROMPT_SCAFFOLD
    assert msgs0[2]["content"][0]["text"] == f"{UI_LINE}\ndwell_seconds: 7.7"
    # missing stats → dwell-only target, same scaffold prompt
    assert msgs1[2]["content"][0]["text"] == "dwell_seconds: 3.0"


def test_builder_without_scaffold_ignores_stat_columns(rows):
    examples = build_vlm_examples(rows, scaffold=False, **BUILD_KW)
    msgs = examples[0]["messages"]
    assert msgs[0]["content"][0]["text"] == SYSTEM_PROMPT
    assert msgs[2]["content"][0]["text"] == "dwell_seconds: 7.7"


def test_builder_scaffold_without_stat_columns_falls_back(rows):
    stripped = rows.drop(columns=[c for _, c in SCAFFOLD_FIELDS])
    examples = build_vlm_examples(stripped, scaffold=True, **BUILD_KW)
    assert examples[0]["messages"][2]["content"][0]["text"] == \
        "dwell_seconds: 7.7"


# ── the fixed λ grid ──────────────────────────────────────────────────────────

def test_grid_is_exactly_five_lambdas():
    cfgs = sorted(Path("configs/sweeps").glob("*.yaml"))
    assert [p.name for p in cfgs] == [
        "lam000.yaml", "lam025.yaml", "lam050.yaml",
        "lam075.yaml", "lam100.yaml",
    ]
    lams, out_dirs = [], []
    base = load_config("configs/vlm.yaml")
    for p in cfgs:
        cfg = load_config(p)
        lams.append(cfg["lupi"]["lambda"])
        out_dirs.append(cfg["paths"]["output_dir"])
        # full fidelity: inherits the base recipe, no train-row cap
        assert cfg["model"] == base["model"]
        assert "max_train_rows" not in cfg["train"]
        assert cfg["data"]["scaffold"] is True
    assert lams == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(set(out_dirs)) == 5     # no two runs share a checkpoint dir


def test_base_config_has_scaffold_enabled():
    cfg = load_config("configs/vlm.yaml")
    assert cfg["data"]["scaffold"] is True
    assert cfg["data"]["scaffold_stats"].endswith("scaffold_stats.parquet")
    assert cfg["eval"]["max_new_tokens"] >= 40   # room for the ui line
