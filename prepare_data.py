"""Step 1: build the tomato subset with leaf masks, resized once to 256 px.

PlantVillage: point --pv at the folder that holds 'color' and 'segmented'
(Kaggle: abdallahalidev/plantvillage-dataset -> 'plantvillage dataset').
PlantDoc (optional): point --plantdoc at the cloned PlantDoc-Dataset folder
(github.com/pratikkayal/PlantDoc-Dataset, with 'train' and 'test' inside).

Writes:
  data/pv256/images/<class>/<id>.jpg, data/pv256/masks/<class>/<id>.png
  data/pv_split.csv          (image, mask, label, split)  70/15/15 stratified
  data/plantdoc256/...  and  data/plantdoc.csv  (image, label)

Usage:  python prepare_data.py --pv "D:/datasets/plantvillage dataset" --plantdoc "D:/datasets/PlantDoc-Dataset"
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy import ndimage
from sklearn.model_selection import train_test_split

from config import PV_CLASSES, PLANTDOC_MAP, STORE_SIZE, SEED

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def seg_key(stem: str) -> str:
    """Segmented files are named '<color stem>_final_masked'."""
    s = stem.lower()
    for suf in ("_final_masked", "_masked"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def mask_from_segmented(seg: Image.Image) -> Image.Image:
    a = np.asarray(seg.convert("RGB")).astype(np.int16)
    m = a.max(axis=2) > 12                       # background in the segmented copy is ~black
    m = ndimage.binary_closing(m, structure=np.ones((5, 5)), iterations=2)
    m = ndimage.binary_fill_holes(m)             # dark lesions inside the leaf stay leaf
    lab, n = ndimage.label(m)                    # keep the largest blob (the leaf)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return Image.fromarray((m * 255).astype(np.uint8))


def process_pv(args_tuple):
    color_p, seg_p, out_img, out_mask = args_tuple
    if out_img.exists() and out_mask.exists():
        return True
    try:
        img = Image.open(color_p).convert("RGB").resize((STORE_SIZE, STORE_SIZE), Image.BICUBIC)
        seg = Image.open(seg_p).convert("RGB").resize((STORE_SIZE, STORE_SIZE), Image.NEAREST)
        mask = mask_from_segmented(seg)
        if np.asarray(mask).mean() / 255 < 0.03:   # broken segmentation -> skip image
            return False
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_img, quality=95)
        mask.save(out_mask)
        return True
    except Exception as e:  # corrupt file
        print("skip", color_p, e)
        return False


def build_plantvillage(pv_root: Path, out_root: Path, workers: int):
    color_root, seg_root = pv_root / "color", pv_root / "segmented"
    assert color_root.is_dir() and seg_root.is_dir(), f"Expected 'color' and 'segmented' in {pv_root}"
    jobs, labels = [], []
    for ci, cname in enumerate(PV_CLASSES):
        cdir, sdir = color_root / cname, seg_root / cname
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cdir}")
        seg_index = {seg_key(p.stem): p for p in list_images(sdir)}
        n_miss = 0
        for p in list_images(cdir):
            sp = seg_index.get(p.stem.lower())
            if sp is None:
                n_miss += 1
                continue
            uid = p.stem.replace(" ", "_")
            jobs.append((p, sp, out_root / "images" / cname / f"{uid}.jpg", out_root / "masks" / cname / f"{uid}.png"))
            labels.append(ci)
        print(f"{cname:50s} {len(seg_index):5d} pairs found, {n_miss} without mask")
    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(process_pv, jobs))
    rows = [(str(j[2]), str(j[3]), l) for j, l, o in zip(jobs, labels, ok) if o]
    df = pd.DataFrame(rows, columns=["image", "mask", "label"])
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)
    df = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    return df


def build_plantdoc(pd_root: Path, out_root: Path):
    rows = []
    for sub in ("train", "test"):              # never trained on, so both halves are field test data
        d = pd_root / sub
        if not d.is_dir():
            continue
        for cdir in sorted(d.iterdir()):
            key = cdir.name.lower().strip()
            if not cdir.is_dir() or key not in PLANTDOC_MAP:
                continue
            for p in list_images(cdir):
                out = out_root / cdir.name.replace(" ", "_") / f"{sub}_{p.stem}.jpg"
                if not out.exists():
                    try:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        Image.open(p).convert("RGB").resize((STORE_SIZE, STORE_SIZE), Image.BICUBIC).save(out, quality=95)
                    except Exception as e:
                        print("skip", p, e)
                        continue
                rows.append((str(out), PLANTDOC_MAP[key]))
    return pd.DataFrame(rows, columns=["image", "label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv", required=True, help="folder containing 'color' and 'segmented'")
    ap.add_argument("--plantdoc", default=None, help="cloned PlantDoc-Dataset folder (optional)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    df = build_plantvillage(Path(a.pv), out / "pv256", a.workers)
    df.to_csv(out / "pv_split.csv", index=False)
    print("\nPlantVillage tomato:", len(df), "images")
    print(df.groupby(["split"]).size().to_string())
    print("mean leaf area fraction:", round(float(np.mean([np.asarray(Image.open(m)).mean() / 255 for m in df["mask"].sample(min(500, len(df)), random_state=0)])), 3))

    if a.plantdoc:
        pdf = build_plantdoc(Path(a.plantdoc), out / "plantdoc256")
        pdf.to_csv(out / "plantdoc.csv", index=False)
        print("\nPlantDoc tomato (field test):", len(pdf), "images")
        print(pdf.label.value_counts().sort_index().to_string())
        if len(pdf) == 0:
            print("WARNING: no PlantDoc tomato folders matched; check folder names against config.PLANTDOC_MAP")


if __name__ == "__main__":
    main()
