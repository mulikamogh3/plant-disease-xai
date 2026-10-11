"""Step 3: score a trained model on every test and append one row to results/results.csv.

Tests: lab test (original), background-only, Mixed-Rand background swap, Grad-CAM LFS,
and PlantDoc field test (if data/plantdoc.csv exists). 95% CIs by bootstrap.

Usage:  python evaluate.py --run runs/resnet50_baseline
        python evaluate.py --all            (every run folder that has best.pt)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader

from config import FIELD_CLASSES, IMG_SIZE, NUM_CLASSES, SHORT_NAMES, SEED
from data import PlantDoc, PlantVillage, background_only, mixed_rand
from models import CamModel, grad_cam, lesion_focus_score


def boot_ci(correct, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, dtype=float)
    s = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def load(run, device):
    ck = torch.load(Path(run) / "best.pt", map_location=device)
    model = CamModel(ck["name"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def eval_lab(model, loader, device):
    """One pass: original, background-only, mixed-rand predictions + LFS on originals."""
    gen = torch.Generator().manual_seed(SEED)
    rec = {k: [] for k in ("y", "orig", "bgonly", "mixed", "lfs", "area")}
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        with torch.no_grad():
            rec["bgonly"].append(model(background_only(x, m)).argmax(1).cpu())
            rec["mixed"].append(model(mixed_rand(x, m, y.cpu(), gen)).argmax(1).cpu())
        with torch.enable_grad():
            out = model(x)
            pred = out.argmax(1)
            cam = grad_cam(model, out.float(), pred)
        lfs, _ = lesion_focus_score(cam, m, (IMG_SIZE, IMG_SIZE))
        rec["y"].append(y.cpu()); rec["orig"].append(pred.cpu())
        rec["lfs"].append(lfs.cpu()); rec["area"].append(m.mean(dim=(1, 2, 3)).cpu())
    return {k: torch.cat(v).numpy() for k, v in rec.items()}


@torch.no_grad()
def eval_field(model, loader, device):
    ys, strict, restricted = [], [], []
    keep = torch.full((NUM_CLASSES,), float("-inf"), device=device)
    keep[FIELD_CLASSES] = 0
    for x, y in loader:
        out = model(x.to(device)).float()
        ys.append(y); strict.append(out.argmax(1).cpu()); restricted.append((out + keep).argmax(1).cpu())
    return torch.cat(ys).numpy(), torch.cat(strict).numpy(), torch.cat(restricted).numpy()


def run_one(run, args, device):
    run = Path(run)
    model, ck = load(run, device)
    test = PlantVillage(args.csv, "test")
    tl = DataLoader(test, args.batch, shuffle=False, num_workers=args.workers)
    r = eval_lab(model, tl, device)
    y = r["y"]
    row = {"run": run.name, "model": ck["name"], "mode": ck["mode"], "best_epoch": ck["epoch"] + 1}
    acc = (r["orig"] == y)
    row["acc"] = acc.mean() * 100
    row["acc_lo"], row["acc_hi"] = [v * 100 for v in boot_ci(acc)]
    row["macro_f1"] = f1_score(y, r["orig"], average="macro") * 100
    row["lfs"] = float(r["lfs"].mean())
    row["lfs_random"] = float(r["area"].mean())              # what a uniform map would score
    row["bgonly_acc"] = (r["bgonly"] == y).mean() * 100
    row["mixed_acc"] = (r["mixed"] == y).mean() * 100
    row["shortcut_gap"] = row["acc"] - row["mixed_acc"]

    per_class = pd.DataFrame({"class": SHORT_NAMES,
                              "lfs": [r["lfs"][y == c].mean() for c in range(NUM_CLASSES)],
                              "acc": [(r["orig"][y == c] == c).mean() * 100 for c in range(NUM_CLASSES)],
                              "bgonly_acc": [(r["bgonly"][y == c] == c).mean() * 100 for c in range(NUM_CLASSES)]})
    per_class.to_csv(run / "per_class.csv", index=False)
    np.save(run / "test_lfs.npy", r["lfs"])

    fcsv = Path(args.field_csv)
    if fcsv.exists():
        fl = DataLoader(PlantDoc(fcsv), args.batch, shuffle=False, num_workers=args.workers)
        fy, fs, fr = eval_field(model, fl, device)
        row["field_n"] = len(fy)
        row["field_acc_strict"] = (fs == fy).mean() * 100
        row["field_acc"] = (fr == fy).mean() * 100
        row["field_acc_lo"], row["field_acc_hi"] = [v * 100 for v in boot_ci(fr == fy)]
        row["field_f1"] = f1_score(fy, fr, average="macro", labels=FIELD_CLASSES) * 100
        np.save(run / "field_confusion.npy", confusion_matrix(fy, fr, labels=FIELD_CLASSES))

    out = Path(args.results)
    out.parent.mkdir(exist_ok=True)
    df = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(df):
        df = df[df.run != row["run"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(out, index=False, float_format="%.3f")
    print(json.dumps({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, np.integer) else v))
                      for k, v in row.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--csv", default="data/pv_split.csv")
    ap.add_argument("--field-csv", default="data/plantdoc.csv")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = sorted(p for p in Path(args.runs_dir).iterdir() if (p / "best.pt").exists()) if args.all else [args.run]
    for r in runs:
        print("=" * 20, r)
        run_one(r, args, device)


if __name__ == "__main__":
    main()
