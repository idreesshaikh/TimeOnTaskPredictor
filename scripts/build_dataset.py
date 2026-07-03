"""
build_dataset.py
================
Dataset pipeline: raw WebChain trajectory JSON → training-ready rows.
Stages are independent, idempotent, and each writes a stats card to
artifacts/ (SPEC.md rule). Run them in order:

  # 1. Audit the raw data BEFORE trusting it → artifacts/raw_audit.md
  python scripts/build_dataset.py <json_dir> --audit

  # 2. Per-screen dwell labels (SPEC.md spec, no fetching)
  #    → data/processed/rows.parquet + artifacts/rows_card.md
  python scripts/build_dataset.py <json_dir> --labels

  # 3. Resolve screenshots into a local cache (resumable)
  #    → data/processed/rows_resolved.parquet + artifacts/image_resolution.md
  python scripts/build_dataset.py --resolve-images [--resolve-limit N]

  # 4. Domain-disjoint splits + winsorized target (configs/data.yaml)
  #    → data/processed/rows_final.parquet + artifacts/splits.json
  #      + artifacts/dataset_card.md
  python scripts/build_dataset.py --splits
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

from totvlm.data import audit_raw, iter_raw_trajectories, write_raw_audit
from totvlm.labels import MAX_DWELL_S, MIN_DWELL_S, build_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Stage 1: raw-data audit ───────────────────────────────────────────────────

def run_audit(json_dir: Path, out_path: Path) -> None:
    log.info(f"Auditing raw trajectories in {json_dir} ...")
    stats = audit_raw(iter_raw_trajectories(json_dir))
    write_raw_audit(stats, out_path)
    log.info(f"Audit written → {out_path}")
    log.info(json.dumps(stats, indent=2))


# ── Stage 2: dwell label rows (SPEC.md "Label definition") ──────────────────

def run_labels(json_dir: Path, rows_out: Path,
               min_dwell: float, max_dwell: float) -> None:
    log.info(f"Building dwell label rows from {json_dir} ...")
    df = build_rows(
        iter_raw_trajectories(json_dir),
        min_dwell_s=min_dwell,
        max_dwell_s=max_dwell,
    )
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(rows_out, compression="zstd", index=False)
    log.info(f"{len(df)} rows → {rows_out}")

    d = df["dwell_s_raw"]
    nav = int(df["is_navigation"].sum())
    summary = {
        "n_rows": int(len(df)),
        "n_trajectories": int(df["trajectory_id"].nunique()),
        "n_registrable_domains": int(df["registrable_domain"].nunique()),
        "dwell_s_raw": {
            "median": round(float(d.median()), 3),
            "p90": round(float(d.quantile(0.90)), 3),
            "p95": round(float(d.quantile(0.95)), 3),
            "max": round(float(d.max()), 3),
        },
        "is_navigation": {"nav": nav, "in_page": int(len(df) - nav)},
        "filters": {"min_dwell_s": min_dwell, "max_dwell_s": max_dwell},
        "source": str(json_dir),
        "output": str(rows_out),
    }
    card = Path("artifacts/rows_card.md")
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(
        "# Dwell label rows card\n\n```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n"
    )
    log.info(f"nav {nav} / in-page {len(df) - nav} — card → {card}")


# ── Stage 3: screenshot resolution ────────────────────────────────────────────

def run_resolve_images(
    rows_in: Path,
    rows_out: Path,
    cache_dir: Path,
    images_root: Path,
    limit: int | None,
) -> None:
    """
    Resolve each row's img_ref to a cached local file. Rows are kept and
    flagged (`img_resolved`), never dropped — exclusion happens at train time.
    Idempotent: cached files are skipped, so reruns resume where they stopped.
    `--resolve-limit N` bounds this run to N un-cached URL fetches (sampling /
    incremental runs); the report then also extrapolates full coverage.
    """
    from totvlm.images import (
        cache_path,
        is_http_ref,
        resolve_refs,
        write_resolution_report,
    )

    df = pd.read_parquet(rows_in)
    refs = [r for r in df["img_ref"].dropna().unique() if r]
    url_refs = [r for r in refs if is_http_ref(r)]
    hash_refs = [r for r in refs if not is_http_ref(r)]

    to_fetch = list(hash_refs)   # hash lookup is local and free — always do it
    uncached = [u for u in url_refs if not cache_path(u, cache_dir).exists()]
    cached = [u for u in url_refs if cache_path(u, cache_dir).exists()]
    if limit is None:
        attempted = uncached
    else:
        # random sample (seed 42) → representative success-rate/size estimates
        attempted = random.Random(42).sample(uncached, min(limit, len(uncached)))
    to_fetch += cached + attempted
    log.info(
        f"{len(refs)} unique refs: {len(url_refs)} URL ({len(cached)} already "
        f"cached, attempting {len(attempted)}), {len(hash_refs)} hash"
    )

    resolved = resolve_refs(to_fetch, cache_dir=cache_dir, images_root=images_root)

    df["img_path"] = df["img_ref"].map(lambda r: resolved.get(r))
    df["img_resolved"] = df["img_path"].notna()
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(rows_out, compression="zstd", index=False)
    log.info(f"{len(df)} rows → {rows_out}")

    n_url_ok = sum(1 for u in url_refs if resolved.get(u))
    n_hash_ok = sum(1 for h in hash_refs if resolved.get(h))
    n_attempt_ok = sum(1 for u in attempted if resolved.get(u))
    stats = {
        "n_rows": int(len(df)),
        "n_rows_resolved": int(df["img_resolved"].sum()),
        "pct_rows_resolved": round(100 * float(df["img_resolved"].mean()), 2),
        "n_refs": len(refs),
        "n_url_refs": len(url_refs),
        "n_hash_refs": len(hash_refs),
        "n_url_resolved": n_url_ok,
        "n_hash_resolved": n_hash_ok,
        "fetch_attempted_this_run": len(attempted),
        "fetch_succeeded_this_run": n_attempt_ok,
    }
    if attempted:
        rate = n_attempt_ok / len(attempted)
        sizes = [
            cache_path(u, cache_dir).stat().st_size
            for u in attempted if resolved.get(u)
        ]
        mean_kb = (sum(sizes) / len(sizes) / 1024) if sizes else 0.0
        stats["url_success_rate_this_run"] = round(rate, 4)
        stats["mean_image_kb_this_run"] = round(mean_kb, 1)
        if limit is not None:
            url_rows = int(df["img_ref"].map(is_http_ref).sum())
            hash_rows = int(df["img_ref"].notna().sum()) - url_rows
            hash_rate = (n_hash_ok / len(hash_refs)) if hash_refs else 0.0
            est = 100 * (url_rows * rate + hash_rows * hash_rate) / len(df)
            proj_gb = mean_kb * len(url_refs) / 1024**2
            stats["estimated_pct_rows_trainable_full_run"] = round(est, 2)
            stats["projected_full_cache_gb"] = round(proj_gb, 1)
            stats["note"] = (
                f"Bounded run (--resolve-limit {limit}): img_resolved reflects "
                "files actually on disk NOW. Rerun without --resolve-limit to "
                "populate the full cache (resumable; cached files are skipped) "
                "and refresh this report."
            )
    write_resolution_report(stats, Path("artifacts/image_resolution.md"))
    log.info(
        f"resolved rows: {stats['n_rows_resolved']}/{stats['n_rows']} "
        f"({stats['pct_rows_resolved']}%) — report → artifacts/image_resolution.md"
    )


# ── Stage 4: splits + winsorized target (SPEC.md §8–§9) ─────────────────────

def run_splits(config_path: Path) -> None:
    """Domain-disjoint 70/15/15 splits (seed from config) + train-only winsor
    cap + y = log1p(dwell_s). Writes rows_final.parquet, splits.json and the
    dataset card. splits.json is the source of truth: if it exists it is
    REUSED, never recomputed."""
    from totvlm.config import load_config
    from totvlm.splits import (
        add_split_column,
        add_targets,
        assert_disjoint,
        dataset_card_stats,
        load_or_create_splits,
        train_winsor_cap,
        write_dataset_card,
    )

    cfg = load_config(config_path)
    paths = cfg["paths"]

    df = pd.read_parquet(paths["rows_resolved"])
    assignment = load_or_create_splits(
        df["registrable_domain"].dropna(),
        path=paths["splits_json"],
        seed=cfg["seed"],
    )
    assert_disjoint(assignment)   # zero cross-split domain overlap, always
    df = add_split_column(df, assignment)

    # Belt-and-braces: assert disjointness at the ROW level too.
    doms = {
        s: set(df.loc[df["split"] == s, "registrable_domain"])
        for s in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = doms[a] & doms[b]
        assert not overlap, f"{a}/{b} rows share domains: {sorted(overlap)[:5]}"

    cap = train_winsor_cap(df, cfg["winsor"]["percentile"])
    df = add_targets(df, cap)

    out = Path(paths["rows_final"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd", index=False)
    log.info(f"{len(df)} rows → {out}")

    stats = dataset_card_stats(df, cap, cfg["winsor"]["percentile"])
    write_dataset_card(stats, paths["dataset_card"])
    log.info(
        f"winsor cap (train p{int(cfg['winsor']['percentile'] * 100)}): "
        f"{cap:.3f}s — card → {paths['dataset_card']}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("json_dir", nargs="?", default="data/raw/all_json_files",
                    help="Directory with raw trajectory JSON files")
    ap.add_argument("--audit", action="store_true",
                    help="Write artifacts/raw_audit.md (no fetching)")
    ap.add_argument("--audit-out", default="artifacts/raw_audit.md")
    ap.add_argument("--labels", action="store_true",
                    help="Build dwell label rows (no fetching)")
    ap.add_argument("--rows-out", default="data/processed/rows.parquet")
    ap.add_argument("--min-dwell-s", type=float, default=MIN_DWELL_S)
    ap.add_argument("--max-dwell-s", type=float, default=MAX_DWELL_S)
    ap.add_argument("--resolve-images", action="store_true",
                    help="Resolve img_refs to a local cache; write rows_resolved")
    ap.add_argument("--resolved-out", default="data/processed/rows_resolved.parquet")
    ap.add_argument("--images-cache", default="data/images_cache")
    ap.add_argument("--images-root", default="data/images_root",
                    help="Local dir searched for opaque-hash image refs")
    ap.add_argument("--resolve-limit", type=int, default=None,
                    help="Bound this run to N un-cached URL fetches (resumable)")
    ap.add_argument("--splits", action="store_true",
                    help="Assign domain splits + winsorized target")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args()

    if not (args.audit or args.labels or args.resolve_images or args.splits):
        ap.error("pick at least one stage: "
                 "--audit / --labels / --resolve-images / --splits")

    if args.audit:
        run_audit(Path(args.json_dir), Path(args.audit_out))
    if args.labels:
        run_labels(Path(args.json_dir), Path(args.rows_out),
                   args.min_dwell_s, args.max_dwell_s)
    if args.resolve_images:
        run_resolve_images(Path(args.rows_out), Path(args.resolved_out),
                           Path(args.images_cache), Path(args.images_root),
                           args.resolve_limit)
    if args.splits:
        run_splits(Path(args.data_config))


if __name__ == "__main__":
    main()
