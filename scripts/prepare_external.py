"""
prepare_external.py
===================
Build the READ-ONLY external validation set under data/external/ (SPEC.md:
evaluated exactly once, zero-shot; NEVER used to pick features,
hyperparameters, thresholds, or the winsorization cap; never referenced from
training/tuning code — enforced by tests/test_external_guard.py).

    uv run python scripts/prepare_external.py [--config configs/external.yaml]
                                              [--zip /path/to/VSGUI10K.zip]

Source decision (recorded in artifacts/external_card.md):

1. FIRST CHOICE — TaskSense (Yin et al., arXiv 2511.09309): measured human
   completion times over GUI tasks. As of 2026-07-03 the paper releases NO
   dataset, code, screenshots, or per-item human times (checked arxiv.org
   listing + web search; only fitted difficulty constants appear in the
   paper, without the underlying screenshots — so there is nothing to
   transcribe into a screenshot→time set). If that changes, drop the
   transcription at data/external/tasksense/tasksense.csv
   [item_id, screenshot_path, human_time_s] and run with `source: tasksense`.

2. DOCUMENTED FALLBACK — VSGUI10K (Putkonen et al. 2025, OSF osf.io/hmg9b,
   public): 10,282 visual-search trials over 894 everyday GUI screenshots
   (web/desktop/mobile) with eye-tracking. Per-trial search time = last
   fixation timestamp within the trial's visual-search phase (img_type=2);
   per-screen human_time_s = the aggregate (median) over target-present
   trials. CAVEAT (also stated in every report): this measures visual
   SEARCH time — one component of Time-on-Task — so it is a weaker but
   fully independent check of whether the model ranks screens like humans.

Output layout (chmod'd read-only at the end):
    data/external/vsgui10k/items.csv   [item_id, screenshot_path,
                                        human_time_s, n_trials, category]
    data/external/vsgui10k/images/     only the referenced screenshots
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import stat
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from totvlm.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ITEM_COLUMNS = ["item_id", "screenshot_path", "human_time_s", "n_trials",
                "category"]


# ── TaskSense (first choice — manual transcription, if it ever exists) ────────

def prepare_tasksense(cfg: dict) -> tuple[Path, pd.DataFrame, dict]:
    csv = Path(cfg["tasksense"]["csv"])
    if not csv.exists():
        sys.exit(
            f"source=tasksense but {csv} does not exist.\n"
            "TaskSense (arXiv 2511.09309) has no public data release as of "
            "2026-07-03 — there are no screenshots or per-item human times "
            "to transcribe. Use the documented fallback instead:\n"
            "    set `source: vsgui10k` in configs/external.yaml"
        )
    df = pd.read_csv(csv)
    missing = {"item_id", "screenshot_path", "human_time_s"} - set(df.columns)
    if missing:
        sys.exit(f"{csv} is missing columns: {sorted(missing)}")
    df["n_trials"] = df.get("n_trials", 1)
    df["category"] = df.get("category", "gui")
    stats = {"source": "tasksense", "n_items": len(df)}
    return csv.parent, df[ITEM_COLUMNS], stats


# ── VSGUI10K (documented fallback — public, OSF) ──────────────────────────────

def _fetch_zip(url: str, dest: Path) -> Path:
    log.info(f"downloading {url} → {dest} (~295 MB)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
    return dest


def prepare_vsgui10k(cfg: dict, zip_path: str | None
                     ) -> tuple[Path, pd.DataFrame, dict]:
    vcfg = cfg["vsgui10k"]
    root = Path(cfg["paths"]["external_root"]) / "vsgui10k"
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    zp = Path(zip_path) if zip_path else _fetch_zip(
        vcfg["zip_url"], root / "VSGUI10K.zip")

    with zipfile.ZipFile(zp) as zf:
        with zf.open("data/vsgui10k_fixations.csv") as fh:
            fx = pd.read_csv(
                io.TextIOWrapper(fh, encoding="utf-8"),
                usecols=["pid", "new_img_name", "img_name", "img_type",
                         "TIME", "absent", "category"],
            )
        log.info(f"fixation rows: {len(fx)}")

        # Per-trial search time: last fixation timestamp of the trial's
        # visual-search phase (img_type=2). Trial key = (pid, new_img_name)
        # — new_img_name uniquely names a trial's stimulus per the readme.
        vs = fx[fx["img_type"] == 2]
        trials = vs.groupby(["pid", "new_img_name"]).agg(
            search_time_s=("TIME", "max"),
            img_name=("img_name", "first"),
            absent=("absent", "first"),
            category=("category", "first"),
        ).reset_index()
        n_trials_all = len(trials)
        if vcfg["target_present_only"]:
            trials = trials[~trials["absent"]]

        # Per-screen aggregate over trials
        per_img = trials.groupby("img_name").agg(
            human_time_s=("search_time_s", vcfg["aggregate"]),
            n_trials=("search_time_s", "size"),
            category=("category", "first"),
        ).reset_index()
        per_img = per_img[per_img["n_trials"] >= vcfg["min_trials_per_screen"]]

        # Extract only the referenced screenshots
        names = set(zf.namelist())
        kept, missing_img = [], 0
        for r in per_img.itertuples(index=False):
            member = f"data/vsgui10k-images/{r.img_name}"
            if member not in names:
                missing_img += 1
                continue
            dest = img_dir / r.img_name
            if not dest.exists():
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
            kept.append({
                "item_id": r.img_name,
                "screenshot_path": str(dest),
                "human_time_s": float(r.human_time_s),
                "n_trials": int(r.n_trials),
                "category": r.category,
            })

    items = pd.DataFrame(kept, columns=ITEM_COLUMNS)
    stats = {
        "source": "vsgui10k",
        "osf": "https://osf.io/hmg9b",
        "trials_total": n_trials_all,
        "trials_target_present": int(len(trials)),
        "screens_min_trials": int(vcfg["min_trials_per_screen"]),
        "screens_missing_image": missing_img,
        "n_items": len(items),
        "per_category": items["category"].value_counts().to_dict(),
        "human_time_s_quantiles": {
            f"p{int(q * 100)}": round(float(items["human_time_s"].quantile(q)), 2)
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "aggregate": vcfg["aggregate"],
        "measure": "visual search time (last fixation timestamp of the "
                   "search phase), target-present trials only",
    }
    return root, items, stats


# ── Card + read-only lockdown ─────────────────────────────────────────────────

def write_card(cfg: dict, stats: dict, items: pd.DataFrame) -> None:
    card = Path(cfg["paths"]["card"])
    card.parent.mkdir(parents=True, exist_ok=True)
    src = stats["source"]
    lines = [
        "# External validation set card (READ-ONLY)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"config `configs/external.yaml` · seed {cfg['seed']}_",
        "",
        "## Source decision",
        "",
        "- **First choice — TaskSense (arXiv 2511.09309)**: NOT available. "
        "As of 2026-07-03 the paper releases no dataset, code, screenshots, "
        "or per-item human times (checked the arXiv listing and web search); "
        "the paper's tables contain only fitted difficulty constants without "
        "the underlying screenshots, so no screenshot→time transcription is "
        "possible.",
        "- **Used: " + (
            "TaskSense transcription** (`data/external/tasksense/`)."
            if src == "tasksense" else
            "documented fallback VSGUI10K** (Putkonen et al. 2025, "
            "https://osf.io/hmg9b, public). ⚠️ It measures visual SEARCH "
            "time — one component of Time-on-Task — so rank agreement "
            "against it is a weaker but fully independent check."),
        "",
        "## Contract (SPEC.md)",
        "",
        "- Lives under `data/external/` and is chmod'd **read-only**.",
        "- Evaluated **exactly once, zero-shot** by "
        "`scripts/validate_external.py`.",
        "- NEVER used to pick features, hyperparameters, thresholds, or the "
        "winsorization cap; no training/tuning code may reference the path "
        "(enforced by `tests/test_external_guard.py`).",
        "",
        "## Stats",
        "",
        f"- Items (screens): **{stats['n_items']}**",
        f"- human_time_s: median "
        f"{float(items['human_time_s'].median()):.2f} s",
        "",
        "```json",
        json.dumps(stats, indent=2),
        "```",
        "",
    ]
    card.write_text("\n".join(lines))
    log.info(f"card → {card}")


def make_read_only(root: Path) -> None:
    ro_file = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    for p in sorted(root.rglob("*"), reverse=True):
        p.chmod(ro_file | (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                           if p.is_dir() else 0))
    root.chmod(ro_file | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log.info(f"{root} locked read-only")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--zip", default=None,
                    help="use an already-downloaded VSGUI10K.zip")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if items.csv already exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = cfg["source"]
    root = Path(cfg["paths"]["external_root"]) / source
    if (root / "items.csv").exists() and not args.force:
        sys.exit(f"{root}/items.csv already exists (READ-ONLY set — rebuild "
                 f"only with --force, which re-derives it from the same "
                 f"public source).")

    if source == "tasksense":
        root, items, stats = prepare_tasksense(cfg)
    elif source == "vsgui10k":
        root, items, stats = prepare_vsgui10k(cfg, args.zip)
    else:
        sys.exit(f"unknown source {source!r} (tasksense | vsgui10k)")

    if items.empty:
        sys.exit("external set came out empty — refusing to write")
    items_csv = root / "items.csv"
    if items_csv.exists():          # --force on a locked set
        items_csv.chmod(0o644)
    items.to_csv(items_csv, index=False)
    log.info(f"{len(items)} items → {items_csv}")

    write_card(cfg, stats, items)
    make_read_only(root)


if __name__ == "__main__":
    main()
