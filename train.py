"""Step 2: train one model in one mode for N epochs.

Modes
  baseline     cross-entropy only
  bgsuppress   background randomisation + attention-alignment loss   (proposed)
  bgrand       background randomisation only                        (ablation)
  attloss      attention-alignment loss only                        (ablation)

Usage:  python train.py --model resnet50 --mode baseline --epochs 50
Resume after a crash / shutdown:  add --resume
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import MODELS, SEED
from data import PlantVillage
from models import CamModel, grad_cam, off_leaf_loss


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def amp_setup(device):
    if device.type != "cuda":
        return False, None
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


@torch.no_grad()
def evaluate(model, loader, device, use_amp, dtype, mf=torch.contiguous_format):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for x, m, y in loader:
        x, y = x.to(device, non_blocking=True, memory_format=mf), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            out = model(x)
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum").item()
        n += len(y)
        ys.append(y.cpu()); ps.append(out.argmax(1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return loss_sum / n, float((y == p).mean()), float(f1_score(y, p, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--mode", default="baseline", choices=["baseline", "bgsuppress", "bgrand", "attloss"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default comes from config.MODELS")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the alignment loss")
    ap.add_argument("--bg-p", type=float, default=0.8, help="probability of background randomisation")
    ap.add_argument("--bg-dir", default=None, help="optional folder of real soil/grass photos")
    ap.add_argument("--csv", default="data/pv_split.csv")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the run folder, e.g. lam0.5")
    a = ap.parse_args()

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, dtype = amp_setup(device)
    torch.backends.cudnn.benchmark = True
    use_bg = a.mode in ("bgsuppress", "bgrand")
    use_att = a.mode in ("bgsuppress", "attloss")
    lr = a.lr or MODELS[a.model]["lr"]

    run = Path(a.out) / f"{a.model}_{a.mode}{('_' + a.tag) if a.tag else ''}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps({**vars(a), "lr": lr, "device": str(device)}, indent=2))

    tr = PlantVillage(a.csv, "train", train=True, bg_rand_p=a.bg_p if use_bg else 0.0, bg_dir=a.bg_dir)
    va = PlantVillage(a.csv, "val")
    # pin_memory off + single-process validation: avoids Windows shared/pinned-memory errors (1455, already mapped)
    kw = dict(num_workers=a.workers, pin_memory=False, persistent_workers=a.workers > 0)
    tl = DataLoader(tr, a.batch, shuffle=True, drop_last=True, **kw)
    vl = DataLoader(va, 64, shuffle=False, num_workers=0, pin_memory=False)

    model = CamModel(a.model, pretrained=not a.no_pretrained).to(device)
    mf = torch.channels_last if (device.type == "cuda" and not a.model.startswith("vit")) else torch.contiguous_format
    model = model.to(memory_format=mf)   # faster tensor-core convolutions on RTX GPUs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.wd)
    steps_per_epoch = min(len(tl), a.max_steps) if a.max_steps else len(tl)
    total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)

    start, best_f1 = 0, -1.0
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best_f1 = ck["epoch"] + 1, ck["best_f1"]
        print(f"resumed from epoch {start}")

    log_path = run / "log.csv"
    if start == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_ce", "train_att", "train_acc",
                                    "val_loss", "val_acc", "val_f1", "lr", "sec"])

    print(f"{run.name}: {len(tr)} train / {len(va)} val images, lr {lr}, device {device}, amp {dtype if use_amp else 'off'}")
    for ep in range(start, a.epochs):
        model.train()
        t0 = time.time()
        s_loss = s_ce = s_att = s_acc = 0.0
        nb = 0
        for step, (x, m, y) in enumerate(tl):
            if a.max_steps and step >= a.max_steps:
                break
            x = x.to(device, non_blocking=True, memory_format=mf)
            m, y = m.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                out = model(x)
                ce = F.cross_entropy(out.float(), y, label_smoothing=0.1)
            if use_att:
                cam = grad_cam(model, out.float(), y, create_graph=True)
                att = off_leaf_loss(cam, m)
            else:
                att = torch.zeros((), device=device)
            loss = ce + a.lam * att
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            s_loss += loss.item(); s_ce += ce.item(); s_att += att.item()
            s_acc += (out.argmax(1) == y).float().mean().item(); nb += 1
            if ep == start and step == 20 and device.type == "cuda":
                per = (time.time() - t0) / 21
                print(f"  ~{per * steps_per_epoch:.0f}s/epoch train -> ~{per * steps_per_epoch * (a.epochs - start) / 60:.0f} min for the run")

        vloss, vacc, vf1 = evaluate(model, vl, device, use_amp, dtype, mf)
        sec = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, s_loss / nb, s_ce / nb, s_att / nb, s_acc / nb, vloss, vacc, vf1,
                                    sched.get_last_lr()[0], round(sec, 1)])
        flag = ""
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save({"model": model.state_dict(), "name": a.model, "mode": a.mode, "epoch": ep, "val_f1": vf1}, run / "best.pt")
            flag = "  *best"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_f1": best_f1}, run / "last.pt")
        print(f"ep {ep + 1:3d}/{a.epochs}  loss {s_loss / nb:.4f} (att {s_att / nb:.3f})  "
              f"train acc {s_acc / nb:.4f}  val acc {vacc:.4f}  val F1 {vf1:.4f}  {sec:.0f}s{flag}")

    print(f"done. best val macro-F1 {best_f1:.4f} -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
