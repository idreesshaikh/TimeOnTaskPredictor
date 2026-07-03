# Architecture
### How the Time-on-Task Predictor is built, file by file

Companion to [`APPROACH.md`](APPROACH.md) (the *why* behind each choice) and
[`PROPOSAL.tex`](PROPOSAL.tex) (the research framing). This document is the
*what*: the system's shape, what every file does, and the invariants that hold
it together. `CLAUDE.md` at the repo root is the binding spec — where this
document and that one disagree, `CLAUDE.md` wins.

---

## 1. What this project is, in one paragraph

We ask how much of **per-screen Time-on-Task** — the seconds a user dwells on
a web UI screen before their next action — is predictable **from the
screenshot alone**, independent of the user's goal or history. To answer it we
(1) derive dwell labels from real interaction traces (WebChain) with a
verified, documented procedure; (2) fine-tune a compact vision-language model
(Qwen3-VL-4B, 4-bit QLoRA) to *emit* the dwell as text from the screenshot,
and compare it against a no-image LightGBM baseline built on interpretable
accessibility-tree features — the bar pixels must clear; and (3) validate the
frozen model zero-shot, exactly once, against an independent human-timing
dataset. The contribution is the methodology and the honest measurement, not a
leaderboard entry.

**The three objectives** (each maps to concrete pipeline stages below):

| objective | question | answered by |
|---|---|---|
| **O1 — Labels** | can faithful per-screen dwell labels be extracted from traces, confounds documented? | stages 1a–1d |
| **O2 — Recoverable signal** | does the screenshot model beat the interpretable structural baseline? | stages 2–4 |
| **O3 — Generalization** | does the frozen model rank screens like independently measured humans? | stage 5 |

---

## 2. System at a glance

```
                 raw WebChain trajectory JSON  (data/raw/)
                                │
             scripts/build_dataset.py  (4 idempotent stages)
                                │
   --audit ──────► artifacts/raw_audit.md        (trust the data first)
   --labels ─────► data/processed/rows.parquet   (dwell spec: labels.py)
   --resolve-images ► data/processed/rows_resolved.parquet
                      + data/images_cache/       (resumable downloads)
   --splits ─────► data/processed/rows_final.parquet   ◄── THE dataset
                   + artifacts/splits.json  (domain-disjoint, reused forever)
                   + artifacts/dataset_card.md  (winsor cap lives here)
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
  scripts/train_baseline.py            python -m totvlm.train        (GPU)
  LightGBM on axTree features          Qwen3-VL-4B QLoRA SFT, Path A:
  (no image — the bar to beat)         emit "dwell_seconds: X.X"
  → baseline_report.md                 → artifacts/vlm_ckpt/ + vlm_train_card.md
              └─────────────────┬──────────────────┘
                                ▼
                    scripts/evaluate.py            (GPU decode, CPU report)
                    TEST head-to-head on identical rows:
                    floors vs LightGBM vs VLM · nav/in-page split ·
                    calibration plot · qualitative screenshots
                    → artifacts/eval_report.md                        (O2)
                                │
                                ▼
        scripts/prepare_external.py  →  data/external/  (READ-ONLY)
        scripts/validate_external.py  (zero-shot, EXACTLY ONCE, guarded)
        → artifacts/external_report.md + scatter                      (O3)
```

Two hard boundaries run through the diagram:

- **The split boundary.** Train/val/test are disjoint by *registrable domain*
  (eTLD+1). Assignments live in `artifacts/splits.json` and are loaded, never
  recomputed. No model is ever tested on a website it trained on.
- **The external boundary.** `data/external/` is chmod'd read-only, evaluated
  exactly once by the one script allowed to touch it, and no training/tuning
  file may even *mention* the path (`tests/test_external_guard.py` fails the
  suite if one does).

---

## 3. Pipeline stages (inputs → outputs → card)

Every stage is seeded (42), idempotent, takes its knobs from `configs/*.yaml`
(no magic numbers in code), and writes a small stats card to `artifacts/` so
each step's decisions are on the record.

### Stage 1 — Dataset (`scripts/build_dataset.py`, CPU)

| substage | does | writes |
|---|---|---|
| `--audit` | descriptive stats over raw JSON *before* trusting it (step types, timestamp coverage, img/axTree availability, merge-eligible runs) | `artifacts/raw_audit.md` |
| `--labels` | the load-bearing dwell spec (see §4, `labels.py`) | `rows.parquet`, `artifacts/rows_card.md` |
| `--resolve-images` | download/cache screenshots (sha1-keyed, resumable, failures flagged not fatal) | `rows_resolved.parquet`, `artifacts/image_resolution.md` |
| `--splits` | domain-disjoint split assignment + train-only p95 winsor cap + target `y = log1p(dwell_s)` | **`rows_final.parquet`**, `artifacts/splits.json`, `artifacts/dataset_card.md` |

Result: **206,925 rows** (≈32k trajectories, 193 domains; train 135 / val 29 /
test 29 domains), winsor cap **24.954 s**, ~32% navigation steps.

### Stage 2 — No-image baseline (`scripts/train_baseline.py`, CPU + network)

Fetches axTrees for a seeded *prefix-stable* sample of rows (20k; smaller
samples are prefixes of larger ones so the on-disk feature cache always
reuses), extracts a tiny interpretable feature vector per tree (element/
link/button/input counts, text length, click-target area, action one-hot,
`is_navigation`, step index) and never stores the ~1.2 MB trees themselves.
Trains LightGBM on TRAIN with early stopping on VAL, evaluates once on TEST.
Current result: MAE(log) **0.5084**, Spearman **0.4813** overall — clearly
above the constant floors (0.578–0.582) → `artifacts/baseline_report.md`.

### Stage 3 — VLM fine-tuning (`python -m totvlm.train`, CUDA GPU)

Path A SFT on Unsloth's official Qwen3-VL vision pipeline: the model learns to
answer with exactly `dwell_seconds: X.X` (dwell winsorized, 1 decimal) given
the system prompt and the downscaled screenshot — **no task title by default**
(the research question is screen-only; `include_task_title: true` is the
ablation). 4-bit QLoRA, vision layers frozen, LoRA r=16/α=16 on language
attention+MLP. Per-epoch val hook decodes a seeded sample and reports
parse-success rate. `--dry-run` (50 examples, same batch/pixels) is the
mandatory memory smoke test before any full run. Checkpoints →
`artifacts/vlm_ckpt/`, card → `artifacts/vlm_train_card.md`.

### Stage 4 — Internal evaluation (`scripts/evaluate.py`) — answers O2

Two sub-stages so the GPU is never re-run to tweak a report:
**predict** (GPU) batch-decodes every TEST row with a resolved screenshot into
a raw-output cache; **report** (CPU) parses with lenient tiers (failure rate
stated; failures imputed with the TRAIN median — never hidden), re-scores the
LightGBM baseline *on the same rows* (axTree features fetched via the cache),
and writes the head-to-head table (floors / LightGBM / VLM × overall /
navigation / in-page), calibration deciles + `calibration.png`, an
auto-computed paragraph on whether the VLM's edge concentrates in in-page
steps (cognitive signal) vs navigation (page-load latency), and six annotated
qualitative screenshots → `artifacts/eval_report.md`.

### Stage 5 — External validation (`prepare_external.py` + `validate_external.py`) — answers O3

`prepare_external.py` builds the read-only set. First choice TaskSense
(arXiv 2511.09309) has **no public data** (checked 2026-07-03); the documented
fallback **VSGUI10K** (Putkonen et al. 2025, OSF `hmg9b`) is used: 894 GUI
screens with per-screen *median visual-search time* over target-present
trials (caveat stated everywhere: search time is a *component* of ToT — a
weaker but fully independent check). Because the training corpus is web-only,
the **web category (298 screens) is the pre-registered in-domain primary**;
desktop/mobile are kept as labeled out-of-domain transfer rows. The set also
carries `aim_metrics.csv` (per-screen Aalto Interface Metrics shipped with
VSGUI10K) for the post-hoc theory-grounding analysis. Decision + stats →
`artifacts/external_card.md`; files locked read-only.

`validate_external.py` runs the frozen checkpoint zero-shot (predict stage
refuses if predictions exist; report stage refuses if the report exists;
`--allow-rerun` overrides but stamps a RERUN warning into the report), then
reports Spearman ρ with a seeded 10k-resample bootstrap 95% CI — overall and
per category (web = in-domain primary) — plus a scatter plot. Parse failures
are *excluded* here, not imputed: a constant would inject fake rank
information into a rank statistic.

---

## 4. Module inventory — `src/totvlm/`

| module | responsibility | key exports |
|---|---|---|
| `config.py` | tiny YAML loader; every entrypoint reads knobs through it — the mechanism behind "no magic numbers in code" | `load_config` |
| `data.py` | three things: (1) typed, tolerant views over raw WebChain JSON + the raw audit; (2) the Path-A chat-example builders (list comprehension, never `datasets.map()` — Arrow fights PIL images); (3) the output parsers | `Step`, `Trajectory`, `iter_raw_trajectories`, `audit_raw`, `build_vlm_examples`, `build_inference_examples`, `format_dwell_target`, `parse_dwell_output` (strict), `parse_dwell_output_lenient` (tiered eval parser) |
| `labels.py` | the load-bearing dwell spec, exactly as written in `CLAUDE.md`: launchApp dropped, same-img+same-tab steps merged into screen units, dwell = createdTime delta to the same-tab predecessor, `is_navigation` flag, hard filters. Verified against two hand-checked trajectory oracles pinned in `tests/test_labels.py` | `build_rows`, `MIN_DWELL_S`, `MAX_DWELL_S` |
| `images.py` | screenshot resolution (URL → sha1-keyed cache; opaque hash → local root), resumable + failure-tolerant; and the model-ready loader that bounds pixels so Qwen3-VL visual tokens stay within budget (`max_pixels` is a hard cap) | `resolve_refs`, `load_image`, `fit_size`, `estimate_visual_tokens` |
| `splits.py` | domain-disjoint split assignment (70/15/15 *by domain count*), saved once to `splits.json` and reused forever; train-only winsor cap; dataset card writer | `assign_splits`, winsor/card helpers |
| `features.py` | interpretable axTree features for the baseline; fetches a tree, extracts the tiny feature vector, caches **features only** (trees are ~1.2 MB × ~182k ≈ 210 GB — infeasible to mirror) | `fetch_axtree_features`, `extract_axtree_features`, `build_feature_frame`, `feature_columns` |
| `scoring.py` | the single shared metrics module — every model (floors, LightGBM, VLM) is scored through it so numbers are comparable. Log-space convention: `y = log1p(dwell_s)`; seconds via `expm1`; Spearman computed once (rank-based, invariant to the monotone transform) | `regression_metrics`, `metrics_by_navigation`, `calibration_table` |
| `model.py` | both models. VLM: `load_vlm` (4-bit Qwen3-VL + LoRA per the backbone rules), `load_vlm_for_inference` (adapter dir → inference mode), `predict_dwell_batch` (left-padded batched greedy decode returning *raw strings* — parsing stays in the caller so failures stay visible). Baseline: `train_lgbm_baseline`, `lgbm_feature_importances`. Unsloth imports are lazy so the module works on CPU-only machines | see left |
| `train.py` | the SFT entrypoint: TRL `SFTTrainer` + `UnslothVisionDataCollator` (nothing hand-rolled), version-tolerant TRL shims, per-epoch `ValDecodeCallback` (parse rate + decoded MAE), `--dry-run` acceptance gate (loss must drop, outputs must parse), train-card writer with measured peak GPU memory. **`unsloth` must be imported before transformers/trl** — guarded by an `# isort: off` block | `main` |

### Scripts — `scripts/`

| script | stage | needs |
|---|---|---|
| `download_minimal.sh` | fetch raw WebChain JSON | network, HF access |
| `build_dataset.py` | stage 1 (audit/labels/images/splits) | CPU |
| `train_baseline.py` | stage 2 (LightGBM baseline) | CPU + network |
| `evaluate.py` | stage 4 (TEST head-to-head, O2) | GPU for predict, CPU for report |
| `prepare_external.py` | stage 5a (build read-only external set) | CPU + network |
| `validate_external.py` | stage 5b (zero-shot O3, evaluate-once) | GPU for predict, CPU for report |
| `analyze_aim.py` | post-hoc theory grounding: regress human time and cached predictions on AIM clutter metrics (APPROACH §5.1) → `artifacts/aim_analysis.md` | CPU, after 5b |

### Configs — `configs/` (every tunable number lives here)

| config | governs |
|---|---|
| `data.yaml` | split fractions, dwell filters, winsor percentile, paths |
| `baseline.yaml` | axTree sampling/caching/vocab, LightGBM params, early stopping |
| `vlm.yaml` | checkpoint, LoRA r/α, image pixel budget, lr/epochs/batch/grad-accum, dry-run knobs, `include_task_title` ablation flag |
| `eval.yaml` | TEST eval: batch size, parse fallback policy, calibration bins, qualitative count |
| `external.yaml` | source choice (tasksense/vsgui10k), aggregation, bootstrap CI knobs — the **only** yaml allowed to reference `data/external` |

### Tests — `tests/`

| test file | protects |
|---|---|
| `test_labels.py` | the dwell spec against two hand-verified trajectory oracles (±0.01 s) — the ground truth of O1 |
| `test_splits.py` | domain-disjointness, determinism, splits.json reuse |
| `test_images.py` | resolution/caching, pixel-budget hard cap |
| `test_features.py` | axTree feature extraction + cache behavior |
| `test_scoring.py` | metric definitions shared by every model |
| `test_vlm_data.py` | chat-example structure, winsor clipping, screen-only default, ablation flag |
| `test_parse.py` | strict + lenient parsers on clean/noisy/malformed outputs |
| `test_external_guard.py` | the read-only rule: no module/script/config outside the external pair may reference `data/external` (includes a planted-leak self-test) |

### Artifacts — `artifacts/` (the paper trail)

`raw_audit.md` → `rows_card.md` → `image_resolution.md` → `dataset_card.md`
(winsor cap) → `baseline_report.md` → `vlm_train_card.md` → `eval_report.md`
(+ `calibration.png`, `qualitative/`) → `external_card.md` →
`external_report.md` (+ scatter). Plus `splits.json` (the frozen split
assignment) and model files (`baseline_lgbm.txt`, `vlm_ckpt/`).

---

## 5. Cross-cutting invariants (the rules that make results defensible)

1. **Seed 42 everywhere** — Python, NumPy, torch, split assignment, subsample
   selection, bootstrap.
2. **All tunables in `configs/*.yaml`** — code contains no magic numbers;
   every report records which config produced it.
3. **Domain-disjoint splits, frozen once** — `splits.json` is loaded, never
   recomputed; leakage through recurring websites is the biggest threat to
   validity and this is the defense.
4. **Train-only winsorization** — the p95 cap (24.954 s) is fitted on TRAIN
   and applied everywhere; raw values kept for reporting; cap logged in the
   dataset card. Target = `log1p(dwell_s_winsorized)`.
5. **One scoring module** — floors, baseline, and VLM all report MAE/RMSE in
   log *and* seconds plus Spearman, overall *and* split by `is_navigation`
   (navigation dwells bundle page-load latency; in-page is the cleaner
   cognitive signal).
6. **Failures are reported, never hidden** — unresolved images, failed axTree
   fetches, unparseable model outputs: all counted in cards/reports, with the
   fallback policy stated (internal eval imputes the train median; external
   validation excludes, because imputing a constant would fabricate rank
   information).
7. **The external set is read-only and evaluated once** — filesystem
   permissions, evaluate-once guards in the script, and a leak-guard test.
8. **Every stage writes a card** — decisions are archived next to results.
9. **GPU work is cached, reports are cheap** — both eval scripts split into a
   predict stage (GPU, writes raw outputs) and a report stage (CPU,
   re-runnable) so report tweaks never re-run inference.

---

## 6. Compute topology & dependencies

- **CPU (any machine):** dataset build, baseline, all report stages, the full
  test suite. `uv sync`.
- **CUDA GPU (~40 GB class):** VLM training and the two predict stages.
  `uv sync --extra vlm` installs the Unsloth stack (Linux-only marker; unsloth
  pins its own compatible TRL/transformers range — see the comment block in
  `pyproject.toml` before touching those pins).
- Version drift across TRL releases is absorbed by small shims in `train.py`
  (`max_seq_length`→`max_length`, `tokenizer`→`processing_class`) rather than
  by pinning the whole stack.

## 7. Current status

| stage | state |
|---|---|
| 1. Dataset (audit → labels → images → splits) | ✅ built — 206,925 rows, cards written |
| 2. LightGBM baseline | ✅ trained + reported (MAE log 0.5084, ρ 0.4813 on TEST) |
| 3. VLM QLoRA SFT | code + tests ready; ⏳ awaiting GPU run (`--dry-run` first) |
| 4. TEST head-to-head (O2) | report path verified end-to-end; ⏳ awaiting VLM predictions |
| 5a. External set | ✅ built read-only (VSGUI10K, 894 screens + AIM metrics) + card |
| 5b. Zero-shot external (O3) | guards verified; ⏳ awaiting the single GPU pass |
| 5c. AIM theory-grounding analysis | ✅ implemented + smoke-tested; ⏳ runs post-hoc on 5b's cached predictions |
