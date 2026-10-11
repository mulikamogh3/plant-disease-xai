"""Shared settings for the Explainability Audit of Plant Disease Models project."""

# PlantVillage tomato folder names (as in the public 'color' / 'segmented' release)
PV_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
SHORT_NAMES = [
    "Bacterial spot", "Early blight", "Late blight", "Leaf mold", "Septoria",
    "Spider mites", "Target spot", "Yellow leaf curl", "Mosaic virus", "Healthy",
]

# PlantDoc (cropped classification release) tomato folders -> PlantVillage class index.
# Target Spot has no PlantDoc counterpart, so the field test uses 9 classes.
PLANTDOC_MAP = {
    "tomato leaf bacterial spot": 0,
    "tomato early blight leaf": 1,
    "tomato leaf late blight": 2,
    "tomato mold leaf": 3,
    "tomato septoria leaf spot": 4,
    "tomato leaf yellow virus": 7,
    "tomato leaf mosaic virus": 8,
    "tomato leaf": 9,
}
FIELD_CLASSES = sorted(set(PLANTDOC_MAP.values()))  # 9 indices, excludes 6

NUM_CLASSES = len(PV_CLASSES)
IMG_SIZE = 224
STORE_SIZE = 256          # images are pre-resized to this once
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# timm model name, Grad-CAM layer (dotted path), default learning rate
MODELS = {
    "resnet50":        {"timm": "resnet50.a1_in1k",            "layer": "layer4",       "lr": 2e-4},
    "efficientnet_b0": {"timm": "efficientnet_b0.ra_in1k",     "layer": "bn2",          "lr": 2e-4},
    "mobilenetv3":     {"timm": "mobilenetv3_large_100.ra_in1k", "layer": "blocks",     "lr": 2e-4},
    "vit_small":       {"timm": "vit_small_patch16_224.augreg_in21k_ft_in1k", "layer": "blocks.-1.norm1", "lr": 1e-4},
}
