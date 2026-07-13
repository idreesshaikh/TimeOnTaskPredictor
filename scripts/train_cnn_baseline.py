"""Pixels-only CNN baseline: ImageNet-pretrained ResNet-50 regressing
log1p(winsorized dwell) from the screenshot alone.

    uv run python scripts/train_cnn_baseline.py [--config configs/cnn.yaml]
                                                [--dry-run]

Why it exists (v3): the study ladder needs an image-only reference between
the constant floors and the VLM conditions — LightGBM answers "what do the
metadata features buy?", this answers "what do generic visual features buy?".
No LUPI blend, no scaffold: raw targets, so the row is a clean pixels-only
control. Train on TRAIN, early-stop on VAL MAE(log), evaluate ONCE on TEST
(same discipline as the LightGBM baseline); writes cnn_test_preds.parquet
(consumed by scripts/evaluate.py) + cnn_baseline_report.md."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from totvlm.config import load_config
from totvlm.data import ROW_KEY
from totvlm.scoring import metrics_by_navigation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


class ScreenDataset(Dataset):
    """Lazy screenshot → (tensor, y) pairs; images load per item, never all
    at once (same reasoning as totvlm.data.VlmExamples)."""

    def __init__(self, frame: pd.DataFrame, size: int):
        self.paths = frame["img_path"].tolist()
        self.y = frame["y"].to_numpy(dtype=np.float32)
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.y[i]


def _split_rows(df: pd.DataFrame, split: str, limit: int | None, seed: int):
    part = df[(df["split"] == split) & df["img_resolved"]
              & df["img_path"].notna()].sort_values(
        ROW_KEY, kind="mergesort").reset_index(drop=True)
    if limit is not None and len(part) > limit:
        part = part.sample(n=limit, random_state=seed).reset_index(drop=True)
    return part


def build_model(mcfg: dict) -> nn.Module:
    if mcfg["arch"] != "resnet50":
        raise SystemExit(f"unsupported arch {mcfg['arch']!r} — configs/cnn."
                         f"yaml supports resnet50")
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if mcfg["pretrained"] \
        else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    if mcfg["freeze_backbone"]:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return model


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device)).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cnn.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny data, 1 epoch — CPU-able smoke test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tcfg = dict(cfg["train"])
    n_train = n_val = None
    report_path = Path(cfg["paths"]["report"])
    model_out = Path(cfg["paths"]["model_out"])
    preds_out = Path(cfg["paths"]["test_preds"])
    if args.dry_run:
        d = cfg["dry_run"]
        n_train, n_val = d["n_train"], d["n_val"]
        tcfg["epochs"] = d["epochs"]
        tcfg["batch_size"] = d["batch_size"]
        tcfg["num_workers"] = d["num_workers"]
        report_path = report_path.with_name(
            report_path.stem + "-dryrun" + report_path.suffix)
        model_out = model_out.with_name(model_out.stem + "-dryrun.pt")
        preds_out = preds_out.with_name(preds_out.stem + "-dryrun.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.dry_run:
        sys.exit("full CNN training needs a CUDA GPU — use --dry-run on CPU")

    df = pd.read_parquet(cfg["paths"]["rows_final"])
    size = cfg["image"]["size"]
    train_rows = _split_rows(df, "train", n_train, seed)
    val_rows = _split_rows(df, "val", n_val, seed)
    log.info(f"train {len(train_rows)} · val {len(val_rows)} resolved screens")
    if train_rows.empty or val_rows.empty:
        sys.exit("no resolved-image rows — run build_dataset.py "
                 "--resolve-images first")

    loader = lambda rows, shuffle: DataLoader(  # noqa: E731
        ScreenDataset(rows, size),
        batch_size=tcfg["batch_size"],
        shuffle=shuffle,
        num_workers=tcfg["num_workers"],
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    train_dl = loader(train_rows, True)
    val_dl = loader(val_rows, False)

    model = build_model(cfg["model"]).to(device)
    head_params = [p for n, p in model.named_parameters()
                   if n.startswith("fc.") and p.requires_grad]
    body_params = [p for n, p in model.named_parameters()
                   if not n.startswith("fc.") and p.requires_grad]
    lr = float(tcfg["learning_rate"])
    opt = torch.optim.AdamW(
        [{"params": body_params, "lr": lr},
         {"params": head_params, "lr": lr * 10}],
        weight_decay=tcfg["weight_decay"],
    )
    loss_fn = nn.L1Loss() if tcfg["loss"] == "l1" else nn.MSELoss()

    best = {"epoch": -1, "val_mae_log": float("inf"), "state": None}
    history = []
    epochs_no_improve = 0
    for epoch in range(tcfg["epochs"]):
        model.train()
        running, n_seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(y)
            n_seen += len(y)
        val_pred = predict(model, val_dl, device)
        val_mae = float(np.mean(np.abs(val_pred - val_rows["y"].to_numpy())))
        history.append({"epoch": epoch,
                        "train_loss": round(running / max(n_seen, 1), 4),
                        "val_mae_log": round(val_mae, 4)})
        log.info(f"epoch {epoch}: train {history[-1]['train_loss']} · "
                 f"val MAE(log) {val_mae:.4f}")
        if val_mae < best["val_mae_log"]:
            best = {"epoch": epoch, "val_mae_log": val_mae,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= tcfg["patience"]:
                log.info(f"early stop after epoch {epoch} "
                         f"(best epoch {best['epoch']})")
                break

    model.load_state_dict(best["state"])
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], model_out)

    # TEST — evaluated exactly once, with the best-by-val weights.
    # (A dry run caps TEST too: it writes -dryrun files, never the real cache.)
    test_rows = _split_rows(df, "test", n_val, seed)
    log.info(f"TEST: predicting {len(test_rows)} resolved screens (once)")
    test_pred = predict(model, loader(test_rows, False), device)
    out = test_rows[ROW_KEY].copy()
    out["cnn_pred_log"] = test_pred
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(preds_out, compression="zstd", index=False)

    test_metrics = metrics_by_navigation(
        test_rows["y"].to_numpy(),
        test_pred,
        test_rows["is_navigation"].to_numpy(),
    )

    lines = [
        "# Pixels-only CNN baseline report",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"config `{args.config}` · seed {seed}"
        f"{' · DRY-RUN' if args.dry_run else ''}_",
        "",
        f"ImageNet-pretrained `{cfg['model']['arch']}`, fc → 1 output, "
        f"input {size}×{size} (direct resize — aspect distortion accepted "
        f"to keep the full page visible). Target `y = log1p(dwell_s)` "
        f"(winsorized). No features, no distillation, no scaffold: this row "
        f"is the pixels-only control between the floors and the VLM "
        f"conditions. Trained on TRAIN, early-stopped on VAL MAE(log) "
        f"(best epoch {best['epoch']}), evaluated ONCE on TEST.",
        "",
        "## Epochs",
        "",
        "| epoch | train loss | val MAE (log) |",
        "|---|---|---|",
        *[f"| {h['epoch']} | {h['train_loss']} | {h['val_mae_log']} |"
          for h in history],
        "",
        "## Test metrics",
        "",
        "| subset | n | MAE (log) | MAE (s) | Spearman ρ |",
        "|---|---|---|---|---|",
        *[f"| {name} | {m['n']} | {m['mae_log']:.4f} | {m['mae_s']:.2f} "
          f"| {m['spearman_rho']:.4f} |"
          for name, m in test_metrics.items() if m.get("n")],
        "",
        "## Full stats (JSON)",
        "",
        "```json",
        json.dumps({"history": history, "best_epoch": best["epoch"],
                    "test_metrics": test_metrics, "dry_run": args.dry_run},
                   indent=2),
        "```",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    log.info(f"weights → {model_out} · test preds → {preds_out} · "
             f"report → {report_path}")


if __name__ == "__main__":
    main()
