import os, json, base64, io, time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Config ─────────────────────────────────────────────────────────────────────
HF_MODEL_URL = os.getenv(
    "HF_MODEL_URL",
    "https://huggingface.co/YOUR_USERNAME/spicecheck/resolve/main/efficientnet_b3_best.pth"
)
HF_CLASSES_URL = os.getenv(
    "HF_CLASSES_URL",
    "https://huggingface.co/YOUR_USERNAME/spicecheck/resolve/main/class_names.json"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224
DEVICE        = torch.device("cpu")   # Render free tier is CPU-only

# Adulteration level mapping for each class
CLASS_INFO = {
    "chilli_pure": {
        "label":       "Pure",
        "adulterant":  "None detected",
        "pct_range":   "0%",
        "safe":        True,
        "fssai":       "Sample appears pure. Safe to consume.",
        "color":       "green",
    },
    "chilli_brick_low": {
        "label":       "Adulterated — Low",
        "adulterant":  "Brick powder",
        "pct_range":   "5–10%",
        "safe":        False,
        "fssai":       "Brick powder detected (low level). FSSAI violation. Avoid consumption. Report to local food safety officer.",
        "color":       "orange",
    },
    "chilli_brick_mid": {
        "label":       "Adulterated — Medium",
        "adulterant":  "Brick powder",
        "pct_range":   "15–25%",
        "safe":        False,
        "fssai":       "Significant brick powder adulteration detected. FSSAI violation under FSS Act 2006. Do not consume.",
        "color":       "red",
    },
    "chilli_brick_high": {
        "label":       "Heavily Adulterated",
        "adulterant":  "Brick powder",
        "pct_range":   "30–50%+",
        "safe":        False,
        "fssai":       "Heavy brick powder adulteration. Serious FSSAI violation. Immediately stop use and report to FSSAI helpline 1800-112-100.",
        "color":       "darkred",
    },
}

# ── Model loader ───────────────────────────────────────────────────────────────
model      = None
class_names = None

def build_model(num_classes: int):
    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    return m

def load_model_from_hf():
    """Download model weights + class names from Hugging Face Hub."""
    import urllib.request
    weights_path = Path("/tmp/efficientnet_b3_best.pth")
    classes_path = Path("/tmp/class_names.json")

    if not classes_path.exists():
        print(f"Downloading class_names.json from HF...")
        urllib.request.urlretrieve(HF_CLASSES_URL, classes_path)

    if not weights_path.exists():
        print(f"Downloading model weights from HF... (this takes ~45s)")
        urllib.request.urlretrieve(HF_MODEL_URL, weights_path)

    with open(classes_path) as f:
        names = json.load(f)

    m = build_model(len(names))
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m.eval()
    print(f"Model loaded. Classes: {names}")
    return m, names

def load_model_local():
    """Load from local files — used during development."""
    local_weights = Path("efficientnet_b3_best.pth")
    local_classes = Path("class_names.json")
    if not local_weights.exists() or not local_classes.exists():
        raise FileNotFoundError(
            "Local model files not found. Put efficientnet_b3_best.pth and "
            "class_names.json in the same folder as main.py for local dev."
        )
    with open(local_classes) as f:
        names = json.load(f)
    m = build_model(len(names))
    m.load_state_dict(torch.load(local_weights, map_location=DEVICE))
    m.eval()
    print(f"Local model loaded. Classes: {names}")
    return m, names

# ── Lifespan (startup) ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names
    print("Loading model at startup...")
    try:
        # Try local first (for development), then HF (for production)
        if Path("efficientnet_b3_best.pth").exists():
            model, class_names = load_model_local()
        else:
            model, class_names = load_model_from_hf()
        print("Model ready.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise
    yield
    print("Shutting down.")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SpiceCheck API",
    description="Food adulteration detection using EfficientNet-B3 + GradCAM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Image preprocessing ────────────────────────────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return eval_transform(img).unsqueeze(0).to(DEVICE)

def generate_gradcam(img: Image.Image, pred_idx: int) -> str:
    """Generate GradCAM heatmap and return as base64 PNG string."""
    raw = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    tensor = preprocess_image(
        raw.astype(np.float32),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ).to(DEVICE)

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(pred_idx)]
    )[0]
    overlay = show_cam_on_image(raw.astype(np.float32), grayscale_cam, use_rgb=True)

    # Convert numpy array to base64 PNG
    overlay_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "SpiceCheck API",
        "version": "1.0.0",
        "status": "running",
        "model": "EfficientNet-B3",
        "classes": class_names,
        "endpoints": {
            "predict":  "POST /predict  — upload image, get prediction + heatmap",
            "health":   "GET  /health   — check if model is loaded",
            "docs":     "GET  /docs     — interactive API docs",
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "classes": class_names,
        "device": str(DEVICE),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a food image → get adulteration prediction + GradCAM heatmap.

    Returns:
    - prediction: class name
    - label: human readable label
    - confidence: model confidence %
    - adulterant: detected adulterant name
    - pct_range: estimated adulteration percentage range
    - safe: whether safe to consume
    - fssai_advisory: FSSAI guidance
    - heatmap_base64: GradCAM overlay image as base64 PNG
    - inference_time_ms: time taken for inference
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    start = time.time()

    # Read + preprocess
    file_bytes = await file.read()
    img = read_image(file_bytes)
    tensor = pil_to_tensor(img)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred_idx    = probs.argmax().item()
        confidence  = round(probs[pred_idx].item() * 100, 2)
        all_probs   = {class_names[i]: round(probs[i].item() * 100, 2)
                       for i in range(len(class_names))}

    pred_class = class_names[pred_idx]
    info = CLASS_INFO.get(pred_class, {})

    # Generate GradCAM
    heatmap_b64 = generate_gradcam(img, pred_idx)

    inference_ms = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "prediction":       pred_class,
        "label":            info.get("label", pred_class),
        "confidence":       confidence,
        "all_probabilities": all_probs,
        "adulterant":       info.get("adulterant", "Unknown"),
        "pct_range":        info.get("pct_range", "Unknown"),
        "safe":             info.get("safe", False),
        "color":            info.get("color", "gray"),
        "fssai_advisory":   info.get("fssai", ""),
        "heatmap_base64":   heatmap_b64,
        "inference_time_ms": inference_ms,
    })

@app.post("/predict/fast")
async def predict_fast(file: UploadFile = File(...)):
    """
    Same as /predict but WITHOUT GradCAM — faster response (~200ms vs ~2s).
    Use this if you only need the classification result.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    file_bytes = await file.read()
    img = read_image(file_bytes)
    tensor = pil_to_tensor(img)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred_idx   = probs.argmax().item()
        confidence = round(probs[pred_idx].item() * 100, 2)

    pred_class = class_names[pred_idx]
    info = CLASS_INFO.get(pred_class, {})

    return JSONResponse({
        "prediction":  pred_class,
        "label":       info.get("label", pred_class),
        "confidence":  confidence,
        "adulterant":  info.get("adulterant", "Unknown"),
        "pct_range":   info.get("pct_range", "Unknown"),
        "safe":        info.get("safe", False),
        "fssai_advisory": info.get("fssai", ""),
    })
