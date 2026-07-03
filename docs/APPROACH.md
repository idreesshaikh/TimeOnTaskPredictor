# Implementation & Approach Guide
### Screenshot-Grounded Prediction of Per-Screen Time-on-Task

This is the working companion to the proposal. The proposal states *what* is being asked and *why* it matters; this document states *how* the single research question is answered, with the scientific reasoning behind each engineering choice. Implementation specifics live here, deliberately, so they stay out of the research question. For the file-by-file map of the implemented system — which module does what, the pipeline stages, and the invariants — see [`ARCHITECTURE.md`](ARCHITECTURE.md); the binding spec is `CLAUDE.md` at the repo root.

---

## 0. The question, and the three objectives that operationalize it

**Research question (v2).** *How accurately can per-screen — and, by aggregation, whole-task — Time-on-Task be predicted from rendered user interface screens, and how much of that predictability resides in the screen alone versus the user's stated task?*

This is one construct-level question. It is answered through three objectives, none of which is a separate research question — they are the means of answering the one:

- **O1 — Labels.** Establish a faithful procedure for deriving per-screen Time-on-Task from real interaction traces, with documented treatment of confounds (system-response latency, repeated-exposure habituation).
- **O2 — Recoverable signal.** Determine whether screen-based learned models improve on an interpretable structural baseline, and decompose the predictable share: two otherwise-identical conditions — *screen-only* (the stimulus-driven share) and *screen+task* (its gap to screen-only = the goal-driven share) — evaluated per screen **and rolled up to whole tasks** (per-trajectory sums of predictions vs. summed actual time — the data-driven analogue of KLM's operator sum, and the deliverable that makes "Time-on-Task Predictor" literally true).
- **O3 — Generalization.** Test whether the screen-only predictor agrees with human times measured independently, zero-shot, separating a genuine relationship from a training-source artifact. (The external set has no task descriptions, so it validates the perceptual core.)

Everything below serves O1–O3. If a piece of engineering serves none of them, it is a stretch goal.

---

## 1. Scientific grounding

The project sits on a well-established lineage, which is what makes a screenshot-to-time model a scientifically motivated step rather than an arbitrary one.

- **Usability theory.** ISO 9241-11:2018 defines *efficiency* as the resources — explicitly including time — expended to achieve a goal. Time-on-Task is the canonical objective measure of that dimension, used in industrial usability testing since the 1980s and codified via the MUSiC project.
- **The predictive tradition (the intellectual parent).** The Keystroke-Level Model (Card, Moran & Newell, 1980; *The Psychology of Human-Computer Interaction*, 1983) predicts expert, error-free task time by decomposing a task into primitive operators (keystroke, point, home, mental, response) and summing empirically calibrated durations — accurate to roughly 20%. Its known limitations are precisely the openings this project exploits: it needs an analyst to hand-code the operator sequence for every task, it assumes a skilled user and a known method, and it ignores visual layout and readability. The interpretable structural baseline in this project (Section 4) is a deliberate, learned, automatically-extracted echo of KLM's operator decomposition — element counts and action types stand in for operators — which is why beating it with pixels is the scientifically interesting result.
- **Learned interface-performance models.** The gradient-descent UI-optimization line (Duan et al., CHI 2020; "Deep Menu" lineage) shows task time is learnable with recurrent models that capture the within-session learning effect — but on constrained interfaces and structured UI descriptions, not rendered screens.
- **Visual search on GUIs (the nearest neighbor).** Eye-tracking datasets of GUI visual search (Putkonen et al., IJHCS 2025) and scanpath-predicting VLMs (SeekUI; Guo et al., CHI 2026) predict *search time* from a screenshot plus a textual cue of the target to find. This project differs on two axes that matter: it is **target-free** (no named element to find) and it predicts **whole per-screen dwell**, of which visual search is only one component.

**The gap, stated once:** no prior approach predicts target-free, per-screen Time-on-Task directly from a rendered screen, learned from heterogeneous real-world traces. That is the contribution.

---

## 2. O1 — Label extraction (the load-bearing methodology)

### 2.1 What the timestamp is (verified, not assumed)

In the interaction traces, each non-`launchApp` step carries epoch-millisecond artifacts whose ordering was checked directly against raw rows and is consistent everywhere:

| Artifact | Marks | Offset vs. `createdTime` |
|---|---|---|
| `fullScreenshot: [hash-TIMESTAMP]` | pre-action screen capture | earlier (2–20s) |
| **`createdTime`** | **the action being committed** | **t = 0 (reference)** |
| `html` / `img` / `axTree` upload filenames | post-action artifacts uploaded | shortly after |

Example (Help click, trajectory `yjeXPEBxd5...`): `fullScreenshot …835229` → `createdTime 840681` (+5.45s) → `html …842857` → `img …843259` → `axTree …845601`. This capture → action → upload ordering, plus monotonic human-plausible gaps within a tab, confirms **`createdTime` is the wall-clock moment the action was committed.**

### 2.2 Label definition

```
dwell(step_i) = createdTime(step_i) - createdTime(step_{i-1})      # within the same tabId
```

### 2.3 Extraction rules (apply in order)

1. Drop `launchApp` rows (no timestamp; not a perceived screen).
2. Drop the first real step of each tab-chain (no valid predecessor).
3. Flag `is_navigation = (href[i] != href[i-1])`. This is the variable O1's confound analysis is built on — not optional.
4. Winsorize dwell at a train-split-fitted percentile (p95; **fitted cap: 24.954 s**). **Log the cutoff in the dataset card** — it is a reportable choice, not a hidden knob.
5. Target = `log(1 + dwell_seconds)` (raw seconds are heavily right-skewed; verified range in spot-checks 0.83s–74s).
6. Structural-baseline features come from the accessibility tree; that baseline is restricted to steps where `axTree` is present — report the coverage fraction.
7. **Split by registrable domain (eTLD+1 via `tldextract`)**, never by trajectory, step, or raw `host` (subdomains of one site must not straddle splits). The same sites recur across traces; a random split leaks and inflates everything. This is the single most important anti-leakage decision. Assignments are frozen once to `artifacts/splits.json` and reused by every stage.

### 2.4 Worked sanity check (from two verified trajectories)

| Transition | nav? | dwell | reading |
|---|---|---|---|
| menu hover → hover | no | 1.0s | fast expert motion |
| → category (URL change) | yes | 8.4s | load-confounded |
| → filter checkbox | no | 22.5s | real scanning |
| → click a search result | no | 54.0s | reading 13 results — large real signal |
| pager → click "2" | no | 0.83s | already rendered |

Sub-1s to 70s+ is the range a screen model must recover, and the nav/in-page contrast is exactly what separates cognitive time from system latency.

### 2.5 One early empirical check

Whether the `fullScreenshot` "before" timestamp is reliably close to the *previous* step's `createdTime` — if so, it can sharpen the binary navigation flag into a continuous load-time-subtracted estimate. Spot-check 50–100 steps once the full data is loaded. Nice-to-have, not a blocker.

### 2.6 What the pipeline produced (O1, done)

The implemented pipeline (`scripts/build_dataset.py`, spec in `totvlm/labels.py`, oracles pinned in `tests/test_labels.py`) yields **206,925 screen-level rows** from ≈32k trajectories across **193 registrable domains** (train 135 / val 29 / test 29), ~32% navigation steps, winsor cap 24.954 s. Full accounting — including every exclusion count — lives in `artifacts/dataset_card.md`.

### 2.7 Screenshot download strategy (disk/time bounded, trajectory-complete)

Two facts make "download everything" wrong: ~182k unique screenshots ≈ 122 GB at observed sizes, and a missing image never corrupts a *label* (labels come from raw-JSON timestamps; unresolved rows are just flagged and excluded) — but it *would* punch holes in the task-level rollup. The implemented strategy (`configs/data.yaml → resolve:`):

- **Sampling unit = a trajectory's rows within one split** (whole tasks; ~1.6% of trajectories cross domains/splits and are sampled per split part). No sampled task has missing screens.
- **Seeded, prefix-stable order** per split with row budgets (default train 6000 / val 1200 / test 3000 ≈ 1,540 complete tasks): growing a budget only *adds* trajectories, so the cache is always reused; the run is resumable — start and stop it any time. This deliberately mirrors the baseline's prefix-stable axTree sampling.
- **Store-time recompression**: downscale to 1280 px + JPEG q85 at download (~670 KB PNG → ~107 KB measured, 6×), since training reads images at ≤1024 px anyway. Default budgets ≈ **1.1 GB** on disk.
- Continuous download-while-training was considered and rejected: it couples shared-GPU time to network reliability for no statistical gain — a resumable, budgeted pre-download achieves the same "grow it over time" property offline.

---

## 3. O2 / O3 — Model and compute

### 3.1 Backbone: a compact open VLM, fine-tuned parameter-efficiently

The proposal stays model-agnostic on purpose. In practice: **Qwen3-VL-4B-Instruct, fine-tuned with Unsloth (QLoRA, 4-bit)**, vision encoder frozen, LoRA on the language layers.

**Why this, on a shared 40GB A100.** Unsloth officially supports Qwen3-VL fine-tuning across 2B/4B/8B/32B and reports ~1.7× faster training and ~60% less VRAM. Community QLoRA figures (third-party, config-dependent — not vendor specs) put **4B ≈ 16GB** and **8B ≈ 24GB**, so both *fit* — which corrects the earlier assumption that 8B was impossible. 4B is still the right primary choice, but for a methodology reason rather than a memory one: the study needs many runs (baseline, screen model, the independent-timing pass, debugging iterations) on a *shared, queued* slot, and 4B's larger headroom buys batch size, iteration speed, and OOM safety. Reserve one optional 8B run as a confirmatory check that a bigger backbone doesn't change the qualitative conclusion. Keep 2B as a fallback only if wall-time, not memory, binds — and verify its small-text/OCR robustness on cluttered web screenshots before trusting it.

**Step zero of the build:** one real forward+backward pass on real data at the intended image resolution and batch size, then read `nvidia-smi`. The VRAM figures above are someone else's config; confirm yours before planning a full run.

### 3.2 Setup (implemented)

```bash
uv sync --extra vlm      # Unsloth stack; Linux + CUDA only
uv run python -m totvlm.train --config configs/vlm.yaml --dry-run   # then the full run
```
The loader (`totvlm/model.py::load_vlm`) does exactly the backbone spec:
```python
model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-4B-Instruct", load_in_4bit=True, ...)
model = FastVisionModel.get_peft_model(
    model, finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True, r=16, lora_alpha=16)
```
The training loop is Unsloth's official Qwen3-VL SFT pipeline (TRL `SFTTrainer` + `UnslothVisionDataCollator`), not hand-written — it handles vision-token packing and chat templating, which are easy to get subtly wrong. Every knob (LR, epochs, batch/grad-accum, image pixel budget, LoRA rank) lives in `configs/vlm.yaml`.

### 3.3 Output formulation — the choice that shapes engineering effort

**Path A (implemented, the primary): structured text-output regression.** Train the model via standard SFT to answer with the pinned format **`dwell_seconds: X.X`** (winsorized dwell, one decimal; system prompt fixed in `totvlm/data.py`). Pure next-token SFT, runs through Unsloth's stock trainer unmodified, no custom head or loss to debug. At evaluation the number is parsed back with a tiered parser (exact format → labeled anywhere → bare number → fail) and scored as regression; parse-failure rates are always reported, with a documented fallback (internal eval imputes the train median; external validation excludes failures — a constant would fabricate rank information). Lowest engineering risk on a fixed time budget.

**Two conditions, one flag (RQ v2).** The model is trained twice with identical recipes: `configs/vlm.yaml` (screen-only — the science core) and `configs/vlm_task.yaml` (`include_task_title: true` — the prompt also carries the task, making it the deployable *predictor*). The controlled gap between them is itself a result: the goal-driven share of dwell, per screen and at task level.

**Path B (stretch): scalar regression head.** A small MLP on pooled hidden states, Huber loss on `log(1+dwell)`. More principled (true regression loss, no parsing) and there is precedent — scalar heads on causal-LM backbones are exactly how text reward models are built. The catch is that public examples are mostly text-only; wiring a head onto multimodal Qwen3-VL means manual glue (pulling pre-LM-head hidden states, a custom trainer) and less reference material. Worth it only after A works, as an architecture ablation — not required to answer the research question.

### 3.4 Multi-image / history (optional)

If adding prior-screen context: build the multi-image dataset with a plain list comprehension, not `dataset.map(...)` — Unsloth's docs flag that `.map()` triggers Arrow type-standardization that fights variable-length image lists.

---

## 4. The interpretable baseline (the bar pixels must clear)

No images. Gradient-boosted trees (LightGBM) on accessibility-tree features. This baseline is not an arbitrary competitor — it is the course's classical performance-model canon, rendered as a learnable model. Each feature operationalizes a named construct from the predictive-modeling literature:

| classical construct | operationalized as (feature) |
|---|---|
| KLM operator class — K/P/M differ in cost (Card, Moran & Newell, 1980) | `action_type` one-hot (click / type / select / …) |
| KLM's R operator — system response time | `is_navigation` flag (page load bundled into the dwell) |
| Hick–Hyman law — choice time grows with the number of alternatives (Hick, 1952; Hyman, 1953) | `ax_n_interactive` and per-kind counts (links / buttons / inputs) |
| Fitts' law — motor time depends on target size (Fitts, 1954) | `click_target_area` (`rect.width × rect.height`) |
| reading / scanning load | `ax_text_len`, `ax_n_nodes` |
| practice & task position (power law of practice) | `f_unit_index` (step index within the trajectory) |

Two deliberate modeling choices: features enter as *raw* quantities and the tree ensemble learns each construct's functional form (trees are invariant to monotone transforms, so Hick's logarithmic shape is learnable rather than imposed); and the constructs are extracted *automatically* from the accessibility tree — removing exactly the manual operator-coding step that limits classical KLM/GOMS practice.

Trains in minutes, no GPU, gives a citable result in week one. The screen model's entire justification (O2) is beating this — which is why the baseline being theory-shaped matters: **whatever margin pixels add on top of it is, by construction, signal the classical constructs do not capture.** Build it first.

**Status: built and reported** (`scripts/train_baseline.py` → `artifacts/baseline_report.md`). On held-out TEST domains (20k-row seeded feature sample): MAE(log) **0.5084**, Spearman **0.4813** overall — clearly above the constant floors (MAE(log) 0.578–0.582). The nav/in-page breakdown already tells a story: within-subset rank correlation drops to ≈0.21, so much of the baseline's overall ρ comes from separating navigation from in-page steps rather than from ranking screens *within* a kind. That is precisely the room the screen model has to demonstrate real visual signal.

---

## 5. Training and evaluation

**Training.** Huber loss on `log(1+dwell)` (Path B) or SFT cross-entropy (Path A). All hyperparameter tuning happens on a domain-disjoint held-out slice of the *training* corpus — never on the independent-timing set. Report navigation vs. in-page diagnostics during training, not only at the end.

**Evaluation (maps to objectives).**
- **O2 (per screen):** MAE/RMSE in log-seconds (plus back-transformed seconds) and Spearman rank correlation on held-out domains: floors < LightGBM < VLM(screen) < VLM(screen+task), all on identical rows. Relative ranking matters as much as absolute error for a screening tool.
- **O2 (task level):** per-screen predictions summed within each trajectory and scored against the summed actual — the KLM-style operator-sum, learned. Totals cover only evaluated steps, identically for targets and every model, so the comparison stays apples-to-apples; the trajectory-complete image sampling (§2.7) exists precisely so sampled tasks have no missing screens.
- **O1 confound check:** all metrics broken down by `is_navigation`, isolating the screen-predictable cognitive component from system-latency-driven intervals; calibration (decile reliability plot) to catch systematic bias.
- **O3 — the headline generalization test:** run the fully-trained model zero-shot on a small, independently collected dataset of cleanly measured human times; report Spearman rank correlation **with a seeded bootstrap 95% CI**. **Touch this set exactly once.** Using it earlier — even to pick features — would convert it from an independent check into a tuning set and destroy the one thing that makes the generalization claim credible. (This is also why these labels are *comparative output*, never *training input*.) The exactly-once rule is enforced mechanically, not by discipline alone: the set is filesystem-read-only, the validation script refuses to re-run once predictions or the report exist, and a guard test fails the suite if any training/tuning file so much as references the path.

**Comparability discipline.** Every model — constant floors, LightGBM, VLM — is scored through the single shared metrics module (`totvlm/scoring.py`), and both evaluation scripts split into a cached GPU *predict* stage and a re-runnable CPU *report* stage, so polishing a report never silently re-runs inference.

### 5.1 Theory-grounding analysis: what does the model's "time sense" consist of? (AIM complexity)

The external set already ships, per screen, the **Aalto Interface Metrics** (AIM) — established computational models of perceptual load: Rosenholtz's feature congestion and subband entropy, contour congestion/density, figure-ground contrast, color-cluster counts. This enables a near-free analysis that turns the black-box predictor back into computational-modeling terms:

1. Regress **measured human time** on the AIM metrics → how much of human per-screen time is explained by quantified visual clutter alone (a replication-flavored anchor to the clutter literature).
2. Regress the **VLM's predictions** on the same metrics → how much of the model's "time sense" is reducible to classical clutter, and how much is residual.
3. The interesting cell: does the VLM's *residual* (the part clutter cannot explain) still correlate with human time? If yes, the model learned something about interfaces beyond established complexity models — stated in effect sizes, not vibes. If no, the honest conclusion is that a screenshot's predictable time cost ≈ its clutter, which is itself a clean, citable finding.

This analysis reads the model *through* the course's theory canon (clutter models playing the role the KLM constructs play for the baseline), costs no GPU time, and both of its outcomes are informative. It runs on the external set only after the one zero-shot pass — it consumes the already-cached predictions and public AIM values, so it respects the evaluate-once rule (nothing feeds back into training or tuning).

**Status: implemented** (`scripts/analyze_aim.py` → `artifacts/aim_analysis.md`; the AIM table ships inside the read-only external set as `aim_metrics.csv`). The human-side anchor regression already ran on real data during verification: quantified clutter explains **R² ≈ 0.13 of human search time on web screens** (≈0.19 across all categories) — the modest-but-real share the literature would predict, and the yardstick the model's estimates will be read against. Note the domain framing: since the training corpus is web-only, the **web subset is the pre-registered in-domain primary** everywhere on the external set; desktop/mobile results are reported as out-of-domain transfer, never as the headline.

**A demoted-but-kept comparison.** If time permits, one additional row in the O2 table: a general-purpose frontier VLM prompted zero-shot with the same format — framed strictly as *"a model with no exposure to human timing data"* vs. *"a model fit to human traces"*. One row and one paragraph answering "does predicting human time require learning from humans?"; not a headline, and not a benchmark paper.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Training timing reflects fast, task-assigned annotators, not naive users | Frame claims as *relative* difficulty, validated by the independent-timing rank correlation (O3); state explicitly. |
| Domain leakage inflates accuracy | Domain-disjoint splits from the first script, not retrofitted. |
| Navigation-step dwell is mostly load latency | Separate reporting (O1); optionally report a headline figure on in-page steps with the full figure alongside. |
| Partial accessibility-tree coverage | Report coverage; if sparse, add a screenshot-derived-feature baseline (CV/OCR heuristics) for a fair comparison. |
| GPU-memory estimates don't hold for the real config | Empirical smoke test before any full run — implemented as `totvlm.train --dry-run` (same batch/pixels, 50 examples), which also gates on loss dropping and outputs parsing. |
| Outlier/idle dwell dominates the loss | Winsorize + Huber + logged cutoff. |

---

## 7. Build order

1. ✅ Label pipeline + dataset card (coverage, winsorization cutoff, navigation fraction) — **O1**.
2. ✅ Interpretable baseline, trained and reported — **O2 floor**.
3. ✅ External set prepared, read-only, decision recorded (`artifacts/external_card.md`).
4. ⏳ Budgeted image resolution (§2.7; mechanism built + probe-verified — run the full budgets, then `--splits` to refresh `rows_final`).
5. ⏳ Memory smoke test on the real GPU (`--dry-run`; code + acceptance gate ready).
6. ⏳ Qwen3-VL-4B + Unsloth QLoRA, Path A — **both conditions**: `configs/vlm.yaml` (screen-only), then `configs/vlm_task.yaml` (screen+task).
7. ⏳ Held-out-domain evaluation: per-screen AND task-level tables, nav/in-page breakdown, goal-increment paragraph, calibration — **O2** (multi-condition report path built and verified end-to-end; awaits VLM predictions).
8. ⏳ Independent-timing zero-shot pass — once, mechanically guarded — **O3**; then the post-hoc AIM analysis (§5.1).
9. Write-up. If time remains: Path B head ablation, optional 8B confirmatory run, history variant.

---

## 8. On the TaskSense direction (input vs. comparative output) — decided

**Comparative output, used once for validation — not training input.** Two reasons. First, mechanically: its tasks are fixed, specific items whose measured times cannot be merged as additional training rows for an entirely different set of screens; the only real "input" use would be letting its findings shape baseline features. Second, and decisively: it is small. The moment any feature or hyperparameter is tuned against it, it can no longer serve as an independent check — and that independent check (O3) is the strongest evidence the project can offer that the model learned a real property of interfaces rather than an artifact of its training source. Spend the small, clean dataset on the thing only it can do.

**Resolution (2026-07-03).** TaskSense released no dataset, code, screenshots, or per-item human times — its tables carry only fitted difficulty constants, so there is no screenshot→time set to transcribe. The documented fallback is in place instead: **VSGUI10K** (Putkonen et al. 2025, OSF `hmg9b`, public) — 894 GUI screens with per-screen median visual-search time over target-present trials (299 web / 298 desktop / 297 mobile). The standing caveat is stated in every report: visual search time is a *component* of Time-on-Task, so this is a weaker but fully independent rank-agreement check, with the web subset as the in-domain primary. Decision record: `artifacts/external_card.md`.

---

## 9. Scope discipline

One research question. Three objectives serving it. Everything else — the specific backbone, Path B, the 8B run, multi-image history — is an appendix or a stretch goal. That discipline is what keeps this a methodology contribution rather than a model-leaderboard entry.