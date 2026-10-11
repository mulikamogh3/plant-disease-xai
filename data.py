"""Datasets, paired image+mask augmentation, and background randomisation."""
import math
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def normalize(x):          # x: (...,3,H,W) in [0,1]
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def denormalize(x):
    return x * STD.to(x.device) + MEAN.to(x.device)


# ---------------------------------------------------------------- backgrounds
class BackgroundBank:
    """Random backgrounds in [0,1], shape (3,H,W).

    Procedural kinds (solid, gradient, smooth noise, 'soil/grass' noise) plus,
    optionally, crops from a folder of real photos (--bg-dir), e.g. soil and grass.
    """

    def __init__(self, size=IMG_SIZE, real_dir=None):
        self.size = size
        self.real = []
        if real_dir:
            self.real = sorted(p for p in Path(real_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def __call__(self):
        s = self.size
        kinds = ["solid", "gradient", "noise", "field"] + (["real", "real"] if self.real else [])
        k = random.choice(kinds)
        if k == "solid":
            return torch.rand(3, 1, 1).expand(3, s, s).clone()
        if k == "gradient":
            c1, c2 = torch.rand(3, 1, 1), torch.rand(3, 1, 1)
            ang = random.uniform(0, 2 * math.pi)
            yy, xx = torch.meshgrid(torch.linspace(0, 1, s), torch.linspace(0, 1, s), indexing="ij")
            t = (xx * math.cos(ang) + yy * math.sin(ang))
            t = (t - t.min()) / (t.max() - t.min() + 1e-6)
            return c1 * (1 - t) + c2 * t
        if k in ("noise", "field"):
            g = random.choice([4, 8, 16, 32])
            n = torch.rand(3, g, g)
            if k == "field":   # earthy / leafy palette
                base = random.choice([torch.tensor([0.45, 0.33, 0.22]), torch.tensor([0.25, 0.45, 0.18]),
                                      torch.tensor([0.55, 0.50, 0.40])]).view(3, 1, 1)
                n = (base + 0.25 * (n - 0.5)).clamp(0, 1)
            n = F.interpolate(n[None], size=(s, s), mode="bicubic", align_corners=False)[0]
            return (n + 0.05 * torch.randn(3, s, s)).clamp(0, 1)
        # real photo crop
        p = random.choice(self.real)
        img = Image.open(p).convert("RGB")
        t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                        v2.RandomResizedCrop(s, scale=(0.2, 1.0), antialias=True)])
        return t(img)


def composite(x01, m, bg):
    """x01 (3,H,W) in [0,1], m (1,H,W) in [0,1], bg (3,H,W)."""
    return x01 * m + bg * (1 - m)


# ---------------------------------------------------------------- datasets
class PlantVillage(Dataset):
    """Returns normalised image (3,224,224), mask (1,224,224) float, label."""

    def __init__(self, csv, split, train=False, bg_rand_p=0.0, bg_dir=None):
        df = pd.read_csv(csv)
        self.df = df[df.split == split].reset_index(drop=True)
        self.train = train
        self.bg_rand_p = bg_rand_p
        self.bank = BackgroundBank(real_dir=bg_dir) if bg_rand_p > 0 else None
        if train:
            self.geo = v2.Compose([
                v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
            ])
            self.color = v2.ColorJitter(0.2, 0.2, 0.2)
        else:
            self.geo = v2.CenterCrop(IMG_SIZE)
            self.color = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = tv_tensors.Image(v2.functional.pil_to_tensor(Image.open(r["image"]).convert("RGB")))
        msk = tv_tensors.Mask(v2.functional.pil_to_tensor(Image.open(r["mask"]).convert("L")))
        img, msk = self.geo(img, msk)
        x = img.float() / 255.0
        m = (msk.float() / 255.0 > 0.5).float()
        if self.color is not None:
            x = self.color(x)
        if self.bank is not None and random.random() < self.bg_rand_p:
            x = composite(x, m, self.bank())
        return normalize(x.as_subclass(torch.Tensor)), m.as_subclass(torch.Tensor), int(r["label"])


class PlantDoc(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv)
        self.t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                             v2.Resize(IMG_SIZE, antialias=True), v2.CenterCrop(IMG_SIZE)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return normalize(self.t(Image.open(r["image"]).convert("RGB"))), int(r["label"])


# ---------------------------------------------------------------- counterfactual test images
def background_only(xn, m):
    """Leaf replaced by the normalisation mean (= 0 after normalising)."""
    return xn * (1 - m)


def mixed_rand(xn, m, labels, generator=None):
    """Each leaf pasted onto the background of another test image from a DIFFERENT class.

    The donor's own leaf area is filled with the donor's mean background colour, so no
    foreign leaf pixels leak in. Mirrors Xiao et al. (ICLR 2021) 'Mixed-Rand'.
    """
    b = xn.shape[0]
    perm = torch.arange(b)
    for i in range(b):
        cands = (labels != labels[i]).nonzero().flatten()
        if len(cands):
            perm[i] = cands[torch.randint(len(cands), (1,), generator=generator)].item()
    donor_x, donor_m = xn[perm], m[perm]
    bg_area = (1 - donor_m).sum(dim=(2, 3), keepdim=True).clamp_min(1)
    bg_mean = (donor_x * (1 - donor_m)).sum(dim=(2, 3), keepdim=True) / bg_area
    donor_bg = donor_x * (1 - donor_m) + bg_mean * donor_m
    return xn * m + donor_bg * (1 - m)
