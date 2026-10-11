"""Streamlit dashboard: Explainability Audit of Plant Disease Classifiers.

Run from the project folder (with the venv active):
    streamlit run app.py

Needs the trained checkpoints in runs/<model>_<mode>/best.pt. The test-image picker and the
Lesion Focus Score also need data/pv_split.csv (from prepare_data.py); uploads work without it.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, SHORT_NAMES
from models import CamModel, grad_cam, lesion_focus_score

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
PV_CSV = ROOT / "data" / "pv_split.csv"
FIELD_CSV = ROOT / "data" / "plantdoc.csv"
MODEL_NAMES = {"ViT-S/16 (transformer)": "vit_small", "EfficientNet-B0 (light CNN)": "efficientnet_b0", "ResNet-50 (CNN)": "resnet50"}
MODES = {"baseline": "Normal training (baseline)", "bgsuppress": "BG-Suppress (our method)"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
JET = matplotlib.colormaps["jet"]

st.set_page_config(page_title="Plant Disease XAI", page_icon="🌿", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading model…")
def load_model(name, mode):
    path = RUNS / f"{name}_{mode}" / "best.pt"
    if not path.exists():
        return None, None
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = CamModel(name, pretrained=False).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, int(ck.get("epoch", -1)) + 1


@st.cache_data
def read_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for col in ("image", "mask"):
        if col in df:
            df[col] = df[col].map(lambda s: str(s) if Path(s).is_absolute() else str(ROOT / s))
    return df


# ------------------------------------------------------------------ image helpers
def center_224(img: Image.Image) -> Image.Image:
    """Resize the short side to 256 then centre-crop 224, matching the test pipeline."""
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    return img.crop((l, t, l + IMG_SIZE, t + IMG_SIZE))


def to_tensor01(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def to_mask(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy((np.asarray(img.convert("L"), dtype=np.float32) / 255.0 > 0.5).astype(np.float32))[None]


def swap_background(x01, m, donor01, donor_m):
    """Paste this leaf onto the donor's background; donor's own leaf area -> its mean background colour."""
    bg_pixels = donor01[:, donor_m[0] < 0.5]
    fill = bg_pixels.mean(dim=1).view(3, 1, 1) if bg_pixels.numel() else torch.full((3, 1, 1), 0.5)
    donor_bg = donor01 * (1 - donor_m) + fill * donor_m
    return x01 * m + donor_bg * (1 - m)


def overlay(x01, cam01, alpha=0.45):
    img = x01.permute(1, 2, 0).numpy()
    heat = JET(cam01)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def explain(model, x01, mask=None):
    """Prediction, top-3, Grad-CAM overlay and (if a mask is given) the Lesion Focus Score."""
    xn = ((x01 - MEAN) / STD)[None].to(DEVICE)
    with torch.enable_grad():
        logits = model(xn).float()
        probs = logits.softmax(1)[0].detach().cpu()
        pred = logits.argmax(1)
        cam = grad_cam(model, logits, pred)
    up = F.interpolate(cam[:, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    up = ((up - up.min()) / (up.max() - up.min() + 1e-8)).detach().cpu().numpy()
    lfs = None
    if mask is not None:
        lfs = float(lesion_focus_score(cam, mask[None].to(DEVICE), (IMG_SIZE, IMG_SIZE))[0][0])
    top = probs.topk(3)
    return {"pred": int(pred), "conf": float(probs.max()), "top": [(SHORT_NAMES[i], float(p)) for p, i in zip(top.values, top.indices)],
            "overlay": overlay(x01, up), "lfs": lfs}


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌿 Plant Disease XAI")
model_label = st.sidebar.selectbox("Model", list(MODEL_NAMES))
model_name = MODEL_NAMES[model_label]
source = st.sidebar.radio("Image source", ["Lab test image", "Upload your own", "Farm photo (PlantDoc)"])

pv = read_csv(PV_CSV)
field = read_csv(FIELD_CSV)
x01, mask, true_label, caption = None, None, None, ""

if source == "Lab test image":
    if pv is None:
        st.sidebar.warning("data/pv_split.csv not found. Run prepare_data.py, or upload an image instead.")
    else:
        test = pv[pv.split == "test"].reset_index(drop=True)
        cls = st.sidebar.selectbox("Disease class", SHORT_NAMES)
        pool = test[test.label == SHORT_NAMES.index(cls)].reset_index(drop=True)
        if "idx" not in st.session_state or st.session_state.get("idx_cls") != cls:
            st.session_state.idx, st.session_state.idx_cls = 0, cls
        if st.sidebar.button("🎲 Another image"):
            st.session_state.idx = int(np.random.randint(len(pool)))
        row = pool.iloc[st.session_state.idx % len(pool)]
        x01 = to_tensor01(center_224(Image.open(row["image"])))
        mask = to_mask(center_224(Image.open(row["mask"])))
        true_label, caption = int(row["label"]), Path(row["image"]).name
        test_mode = st.sidebar.radio("Test", ["Original", "Swap background (Mixed-Rand)", "Remove leaf (background only)"])
        if test_mode.startswith("Swap"):
            donors = test[test.label != true_label]
            d = donors.iloc[st.session_state.idx % len(donors) * 7919 % len(donors)]
            x01 = swap_background(x01, mask, to_tensor01(center_224(Image.open(d["image"]))), to_mask(center_224(Image.open(d["mask"]))))
            caption += f" · background from a {SHORT_NAMES[int(d['label'])]} image"
        elif test_mode.startswith("Remove"):
            x01 = x01 * (1 - mask) + MEAN * mask
            caption += " · leaf removed"
elif source == "Upload your own":
    up = st.sidebar.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
    if up is not None:
        x01 = to_tensor01(center_224(Image.open(up)))
        caption = up.name
        known = st.sidebar.selectbox("True class (optional)", ["Unknown"] + SHORT_NAMES)
        true_label = None if known == "Unknown" else SHORT_NAMES.index(known)
else:
    if field is None:
        st.sidebar.warning("data/plantdoc.csv not found.")
    else:
        if st.sidebar.button("🎲 Another farm photo") or "fidx" not in st.session_state:
            st.session_state.fidx = int(np.random.randint(len(field)))
        row = field.iloc[st.session_state.fidx]
        x01 = to_tensor01(center_224(Image.open(row["image"])))
        true_label, caption = int(row["label"]), Path(row["image"]).name

st.sidebar.markdown("---")
st.sidebar.caption("Amogh Narendra Mulik · Smruti Abhijit Shinde · Atharv Sanjay Chendake\n\nGuides: Prof. Prerna P. Nayakal · Prof. Ashvini Sandip Patil\n\nDept. of CSE (AIML), RIT Rajaramnagar")

# ------------------------------------------------------------------ main
st.title("Does the model look at the leaf, or the background?")
tab_pred, tab_res, tab_about = st.tabs(["🔍 Predict & explain", "📊 Results", "ℹ️ About"])

with tab_pred:
    if x01 is None:
        st.info("Choose an image in the sidebar.")
    else:
        cols = st.columns([1, 1.2, 1.2])
        with cols[0]:
            st.subheader("Input")
            st.image(x01.permute(1, 2, 0).numpy(), use_container_width=True, caption=caption)
            if true_label is not None:
                st.markdown(f"**True class:** {SHORT_NAMES[true_label]}")
        for col, mode in zip(cols[1:], MODES):
            with col:
                st.subheader(MODES[mode])
                model, epoch = load_model(model_name, mode)
                if model is None:
                    st.error(f"Missing runs/{model_name}_{mode}/best.pt")
                    continue
                r = explain(model, x01, mask)
                ok = "" if true_label is None else ("✅" if r["pred"] == true_label else "❌")
                st.markdown(f"### {ok} {SHORT_NAMES[r['pred']]}")
                st.metric("Confidence", f"{r['conf'] * 100:.1f}%")
                st.image(r["overlay"], use_container_width=True, caption="Grad-CAM: red = where the model looked")
                if r["lfs"] is not None:
                    st.metric("Lesion Focus Score", f"{r['lfs']:.2f}", help="Share of attention on the leaf. 1.0 = all on the leaf; about 0.52 = spread randomly.")
                st.dataframe(pd.DataFrame(r["top"], columns=["Top 3 classes", "Probability"]).style.format({"Probability": "{:.1%}"}),
                             hide_index=True, use_container_width=True)
        st.caption("Confidence is for this one image. Accuracy is measured over the 2,724-image test set (see the Results tab).")

with tab_res:
    res = read_csv(RESULTS / "results.csv")
    if res is None:
        st.info("results/results.csv not found. Run evaluate.py --all first.")
    else:
        nice = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "vit_small": "ViT-S/16"}
        res = res.assign(Model=res.model.map(nice).fillna(res.model), Training=res["mode"].map({"baseline": "Baseline", "bgsuppress": "BG-Suppress"}).fillna(res["mode"]))
        main = res[res["mode"].isin(["baseline", "bgsuppress"])]
        c = st.columns(4)
        c[0].metric("Avg. lab accuracy", f"{main.acc.mean():.2f}%")
        c[1].metric("Avg. farm accuracy", f"{main.field_acc.mean():.1f}%" if "field_acc" in main else "–")
        c[2].metric("Leaf focus, baseline → ours", f"{main[main['mode'] == 'baseline'].lfs.mean():.2f} → {main[main['mode'] == 'bgsuppress'].lfs.mean():.2f}")
        c[3].metric("Worst shortcut gap → ours", f"{main[main['mode'] == 'baseline'].shortcut_gap.max():.1f} → {main[main['mode'] == 'bgsuppress'].shortcut_gap.max():.1f} pts")
        st.subheader("All runs")
        show = {"Model": "Model", "Training": "Training", "acc": "Lab acc. (%)", "macro_f1": "Macro-F1 (%)", "lfs": "LFS",
                "bgonly_acc": "BG-only acc. (%)", "shortcut_gap": "Shortcut gap (pts)", "field_acc": "Farm acc. (%)", "field_f1": "Farm F1 (%)"}
        st.dataframe(main[[k for k in show if k in main]].rename(columns=show).round(3), hide_index=True, use_container_width=True)
        g1, g2 = st.columns(2)
        piv = lambda col: main.pivot(index="Model", columns="Training", values=col)
        with g1:
            st.markdown("**Lesion Focus Score** (higher = looks at the leaf)")
            st.bar_chart(piv("lfs"), stack=False)
        with g2:
            st.markdown("**Shortcut gap** (points lost when the background is swapped)")
            st.bar_chart(piv("shortcut_gap"), stack=False)
        for f, cap in [("fig_gradcam_grid.png", "Grad-CAM for the same test images under every model"), ("fig_curves.png", "Training curves")]:
            if (RESULTS / f).exists():
                st.image(str(RESULTS / f), caption=cap, use_container_width=True)

with tab_about:
    st.markdown("""
**Problem.** Plant disease models score 99%+ on PlantVillage lab photos but fail on real farm photos. Are they looking at the disease, or cheating with the background?

**What we did.**
1. Trained ResNet-50, EfficientNet-B0 and ViT-S/16 for 30 epochs on PlantVillage tomato (18,155 images, 10 classes).
2. Measured attention with Grad-CAM and the **Lesion Focus Score** (share of attention on the leaf).
3. Proved the shortcut with **background-only** and **background-swap (Mixed-Rand)** tests.
4. Proposed **BG-Suppress**: random backgrounds + a loss that penalises attention outside the leaf.
5. Tested every model on 744 real farm photos (PlantDoc).

**Key results.** Leaf focus rose to 0.98–0.99 and the background-swap loss fell to ≤ 0.3 points, at ≤ 0.37 points of lab accuracy. Farm accuracy stayed at 25–29%: background is only one of the lab-to-field differences.
""")
