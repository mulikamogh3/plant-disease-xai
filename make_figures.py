"""Step 4: paper figures and tables from runs/ and results/results.csv.

  results/fig_gradcam_grid.png   same 6 test images x every run (Grad-CAM overlay)
  results/fig_lfs_vs_field.png   LFS vs PlantDoc accuracy, Pearson r
  results/fig_curves.png         validation accuracy / training loss per run
  results/table_A.md, table_B.md paper tables (Markdown, paste into the doc)

Usage:  python make_figures.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import IMG_SIZE, SHORT_NAMES, SEED
from data import PlantVillage, denormalize
from evaluate import load
from models import grad_cam, lesion_focus_score

ORDER_M = ["resnet50", "efficientnet_b0", "mobilenetv3", "vit_small"]
NICE_M = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "mobilenetv3": "MobileNetV3-L", "vit_small": "ViT-S/16"}
NICE_MODE = {"baseline": "Baseline", "bgsuppress": "BG-Suppress", "bgrand": "Randomisation only", "attloss": "Alignment loss only"}


def run_dirs(runs_dir):
    ds = [p for p in Path(runs_dir).iterdir() if (p / "best.pt").exists()]
    key = lambda p: (next((i for i, m in enumerate(ORDER_M) if p.name.startswith(m)), 9), p.name)
    return sorted(ds, key=key)


def gradcam_grid(runs, csv, out, n_img=6):
    ds = PlantVillage(csv, "test")
    rng = np.random.default_rng(SEED)
    # one image from each of n_img different classes, fixed by seed
    idx = [int(rng.choice(np.where(ds.df.label.values == c)[0])) for c in rng.choice(10, n_img, replace=False)]
    xs = torch.stack([ds[i][0] for i in idx]); ms = torch.stack([ds[i][1] for i in idx])
    ys = [ds[i][2] for i in idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = 1 + len(runs)
    fig, ax = plt.subplots(rows, n_img, figsize=(2.0 * n_img, 2.0 * rows))
    ax = np.atleast_2d(ax)
    imgs = denormalize(xs).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    for j in range(n_img):
        ax[0, j].imshow(imgs[j]); ax[0, j].set_title(SHORT_NAMES[ys[j]], fontsize=8)
    ax[0, 0].set_ylabel("Input", fontsize=8)
    for i, run in enumerate(runs, start=1):
        model, ck = load(run, device)
        x, m = xs.to(device), ms.to(device)
        with torch.enable_grad():
            o = model(x); cam = grad_cam(model, o.float(), o.argmax(1))
        lfs, c = lesion_focus_score(cam, m, (IMG_SIZE, IMG_SIZE))
        for j in range(n_img):
            ax[i, j].imshow(imgs[j]); ax[i, j].imshow(c[j].cpu(), cmap="jet", alpha=0.45)
            ax[i, j].set_title(f"LFS {lfs[j]:.2f}", fontsize=7)
        ax[i, 0].set_ylabel(f"{NICE_M.get(ck['name'], ck['name'])}\n{NICE_MODE.get(ck['mode'], ck['mode'])}", fontsize=7)
        del model
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def lfs_vs_field(res, out):
    if "field_acc" not in res:
        print("no field results yet; skipping scatter"); return
    fig, a = plt.subplots(figsize=(4.2, 3.4))
    for mode, mk in (("baseline", "o"), ("bgsuppress", "s"), ("bgrand", "^"), ("attloss", "v")):
        d = res[res["mode"] == mode]
        if len(d):
            a.scatter(d.lfs, d.field_acc, marker=mk, label=NICE_MODE[mode])
            for _, r in d.iterrows():
                a.annotate(NICE_M.get(r.model, r.model), (r.lfs, r.field_acc), fontsize=6, xytext=(3, 3), textcoords="offset points")
    if len(res) >= 3:
        r = np.corrcoef(res.lfs, res.field_acc)[0, 1]
        a.set_title(f"Pearson r = {r:.2f}  (n = {len(res)} runs)", fontsize=9)
    a.set_xlabel("Lesion Focus Score (lab test)"); a.set_ylabel("PlantDoc accuracy (%)")
    a.legend(fontsize=7); plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def curves(runs, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for run in runs:
        lg = pd.read_csv(run / "log.csv")
        a1.plot(lg.epoch + 1, lg.val_acc * 100, label=run.name, lw=1)
        a2.plot(lg.epoch + 1, lg.train_ce, label=run.name, lw=1)
    a1.set_xlabel("epoch"); a1.set_ylabel("validation accuracy (%)")
    a2.set_xlabel("epoch"); a2.set_ylabel("training cross-entropy")
    a1.legend(fontsize=6); plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def tables(res, outdir):
    cols = ["model", "mode", "acc", "macro_f1", "lfs", "bgonly_acc", "shortcut_gap"] + (["field_acc"] if "field_acc" in res else [])
    t = res[res["mode"].isin(["baseline", "bgsuppress"])][cols].copy()
    t["model"] = t.model.map(NICE_M); t["mode"] = t["mode"].map(NICE_MODE)
    t = t.round({"acc": 2, "macro_f1": 2, "lfs": 3, "bgonly_acc": 2, "shortcut_gap": 2, "field_acc": 2})
    (outdir / "table_A.md").write_text(t.to_markdown(index=False))
    b = res[res.model == "resnet50"][[c for c in ["mode", "acc", "lfs", "shortcut_gap", "field_acc"] if c in res]].copy()
    b["mode"] = b["mode"].map(NICE_MODE)
    (outdir / "table_B.md").write_text(b.round(3).to_markdown(index=False))
    print((outdir / "table_A.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--csv", default="data/pv_split.csv")
    ap.add_argument("--results", default="results/results.csv")
    a = ap.parse_args()
    out = Path(a.results).parent; out.mkdir(exist_ok=True)
    runs = run_dirs(a.runs_dir)
    curves(runs, out / "fig_curves.png")
    gradcam_grid(runs, a.csv, out / "fig_gradcam_grid.png")
    if Path(a.results).exists():
        res = pd.read_csv(a.results)
        lfs_vs_field(res, out / "fig_lfs_vs_field.png")
        tables(res, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
