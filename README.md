# Explainability Audit of Plant Disease Classifiers

**Does a plant disease model look at the leaf, or at the background?**

Deep learning models score 99%+ on PlantVillage lab photos but fail on real farm photos. This project uses **Grad-CAM** to measure where three models look, proves that they partly use the image **background** as a shortcut, and proposes **BG-Suppress**, a training strategy that moves attention back onto the leaf.

Mini project · Department of CSE (AIML) Engineering · Rajarambapu Institute of Technology, Rajaramnagar · 2026-27

| Team | Guides |
|---|---|
| Amogh Narendra Mulik (2417022) · Smruti Abhijit Shinde (2417029) · Atharv Sanjay Chendake (2417033) | Prof. Prerna P. Nayakal · Prof. Ashvini Sandip Patil |

## Method

1. **Data:** PlantVillage tomato, 10 classes, 18,155 images (70/15/15 split). Leaf masks come from the dataset's segmented copy. 744 PlantDoc farm photos are used for testing only.
2. **Models:** ResNet-50, EfficientNet-B0 and ViT-S/16 (ImageNet-pretrained, via `timm`), 30 epochs each.
3. **Audit:**
   - **Lesion Focus Score (LFS):** the share of Grad-CAM attention that falls on the leaf (1.0 = all on the leaf).
   - **Background-only test:** the leaf is removed, so the model sees only the background.
   - **Mixed-Rand test:** the leaf is pasted onto another class's background.
4. **BG-Suppress:** random background replacement (p = 0.8) plus an attention-alignment loss that penalises Grad-CAM attention outside the leaf.

## Results

| Model | Training | Lab acc. (%) | LFS | Background-only acc. (%) | Shortcut gap (pts) | Farm acc. (%) |
|---|---|---|---|---|---|---|
| ResNet-50 | Baseline | 99.23 | 0.905 | 22.21 | 3.27 | 28.63 |
| ResNet-50 | BG-Suppress | 99.19 | 0.994 | 13.55 | 0.07 | 28.63 |
| EfficientNet-B0 | Baseline | 99.82 | 0.839 | 24.05 | 12.30 | 25.27 |
| EfficientNet-B0 | BG-Suppress | 99.60 | 0.985 | 12.52 | 0.29 | 25.81 |
| ViT-S/16 | Baseline | 99.85 | 0.602 | 27.86 | 13.80 | 26.48 |
| ViT-S/16 | BG-Suppress | 99.49 | 0.981 | 5.98 | 0.29 | 28.50 |

- **The shortcut is real.** With the leaf removed, the baselines still score 22–28%, where chance is 10%.
- **BG-Suppress removes it.** LFS rises to 0.98–0.99 and the shortcut gap falls to ≤ 0.3 points, at a cost of ≤ 0.37 points of lab accuracy.
- **Farm accuracy stays at 25–29%.** Background is only one of the differences between lab and field photos.

## Setup (Windows, NVIDIA GPU)

```bat
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Data (not included in this repo)

- PlantVillage (with `color` and `segmented` folders): https://github.com/spMohanty/PlantVillage-Dataset
- PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset

```bat
python prepare_data.py --pv "D:\datasets\plantvillage dataset" --plantdoc "D:\datasets\PlantDoc-Dataset"
```

### Train and evaluate

```bat
run_main.bat
```

This trains 3 models × (baseline, BG-Suppress) for 30 epochs, then runs `evaluate.py --all` and `make_figures.py`. The outputs go to `runs/` and `results/`.

### Trained weights

The `best.pt` checkpoints are attached to the GitHub **Releases** page. Put each one in `runs/<model>_<mode>/best.pt`, for example `runs/vit_small_bgsuppress/best.pt`.

## Dashboard

```bat
streamlit run app.py
```

- Pick a lab test image, upload your own, or pick a farm photo.
- See the prediction, confidence, top-3 classes and Grad-CAM heatmap for the **baseline** and **BG-Suppress** models side by side.
- Try the background-swap and leaf-removed tests live.
- Browse all results in the Results tab.

## Files

| File | Role |
|---|---|
| `config.py` | Classes, model names, Grad-CAM layers |
| `prepare_data.py` | Resizes images, builds leaf masks and the split |
| `data.py` | Datasets, background bank, counterfactual tests |
| `models.py` | timm models, Grad-CAM, alignment loss, LFS |
| `train.py` | Training: `--mode baseline`, `bgsuppress`, `bgrand` or `attloss` |
| `evaluate.py` | All tests with bootstrap 95% CIs → `results/results.csv` |
| `make_figures.py` | Figures and tables |
| `app.py` | Streamlit dashboard |

## References

1. Mohanty et al., *Using deep learning for image-based plant disease detection*, Front. Plant Sci., 2016.
2. Singh et al., *PlantDoc: A dataset for visual plant disease detection*, CoDS-COMAD, 2020.
3. Selvaraju et al., *Grad-CAM*, ICCV, 2017.
4. Geirhos et al., *Shortcut learning in deep neural networks*, Nature Machine Intelligence, 2020.
5. Xiao et al., *Noise or signal: The role of image backgrounds in object recognition*, ICLR, 2021.
