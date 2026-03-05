"""
Krishi Kavach – Unified ML Inference Server
Supports: 
1. YOLOv8 Detection (5 main crops: Banana, Chilli, Radish, Groundnut, Cauliflower)
2. PyTorch Classification (38 PlantVillage classes across 14 crops)
3. Yield Analysis (Via AI/Gemini integration in backend)

Run: python app.py (starts on http://localhost:8000)
"""

import os
import sys
import io
import json
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ── Global Model Status ──────────────────────────────────────────────────────
yolo_ready = False
classifier_ready = False
vit_ready = False
vit_gen_ready = False

# ── Global Libraries (Lazy Loaded) ───────────────────────────────────────────
torch = None
nn = None
models = None
T = None
YOLO = None
F = None
ViTForImageClassification = None
ViTImageProcessor = None
preprocess = None

# Imported local services
from youtube_search import search_videos
from scraper_service import get_hybrid_facilities

# ── Lifespan for Async Startup ──────────────────────────────────────────────
def background_model_loading():
    global yolo_ready, classifier_ready, vit_ready, vit_gen_ready
    global torch, nn, models, T, YOLO, F, ViTForImageClassification, ViTImageProcessor, preprocess
    
    print("[*] Loading AI libraries and models in a background thread...")
    try:
        import torch as _torch
        import torch.nn as _nn
        import torchvision.models as _models
        import torchvision.transforms as _T
        import torch.nn.functional as _F
        from ultralytics import YOLO as _YOLO
        from transformers import ViTForImageClassification as _ViT, ViTImageProcessor as _Proc
        
        torch, nn, models, T, YOLO, F = _torch, _nn, _models, _T, _YOLO, _F
        ViTForImageClassification, ViTImageProcessor = _ViT, _Proc

        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        yolo_ready = load_yolo()
        classifier_ready = load_classification_models()
        vit_ready = load_vit()
        vit_gen_ready = load_vit_general()
        
        print(f"[+] AI Engine ready. Models: YOLO={yolo_ready}, Classifier={classifier_ready}, ViT={vit_ready}")
    except Exception as e:
        print(f"[-] Startup Failure: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    print("[*] Web server starting. Backgrounding model load for Render compatibility...")
    thread = threading.Thread(target=background_model_loading)
    thread.start()
    yield
    # No cleanup really needed for simple server

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
YOLO_FALLBACK = os.path.join(BASE_DIR, "yolov8s.pt")

CLASS_NAMES_PATH = os.path.join(BASE_DIR, "pretrained_models", "class_names.json")
EFFICIENTNET_PATH = os.path.join(BASE_DIR, "pretrained_models", "efficientnet_plant.pth", "efficientnet_plant")
MOBILENET_PATH = os.path.join(BASE_DIR, "pretrained_models", "mobilenetv2_plant.pth", "mobilenetv2_plant")

app = FastAPI(title="Krishi Kavach Unified ML Server", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Models ─────────────────────────────────────────────────────────────
yolo_model = None
classifier_model = None
classification_classes = []

# ── Hugging Face Fallback Model (ViT) ────────────────────────────────────────
vit_model = None
vit_processor = None
vit_gen_model = None
vit_gen_processor = None
VIT_MODEL_NAME = "nateraw/vit-base-patch16-224-plant-village"
VIT_FALLBACK_NAME = "google/vit-base-patch16-224" # General ViT
VIT_GENERAL_NAME = "google/vit-base-patch16-224"

def load_vit_general():
    global vit_gen_model, vit_gen_processor
    try:
        print(f"[*] Loading General ViT Model ({VIT_GENERAL_NAME})...")
        vit_gen_processor = ViTImageProcessor.from_pretrained(VIT_GENERAL_NAME)
        vit_gen_model = ViTForImageClassification.from_pretrained(VIT_GENERAL_NAME)
        vit_gen_model.eval()
        print("[+] General ViT Model loaded successfully")
        return True
    except Exception as e:
        print(f"[-] General ViT load error: {e}")
        return False

def load_vit():
    global vit_model, vit_processor
    try:
        print(f"[*] Loading Hugging Face Fallback Model ({VIT_MODEL_NAME})...")
        try:
            vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
            vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)
        except Exception as e:
            print(f"[*] Primary ViT failed: {e}. Trying standard fallback: {VIT_FALLBACK_NAME}...")
            vit_processor = ViTImageProcessor.from_pretrained(VIT_FALLBACK_NAME)
            vit_model = ViTForImageClassification.from_pretrained(VIT_FALLBACK_NAME)
            
        vit_model.eval()
        print("[+] ViT Fallback Model loaded successfully")
        return True
    except Exception as e:
        print(f"[-] ViT Fallback Model could not be loaded: {e}")
        return False

def load_yolo():
    global yolo_model
    target = YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else YOLO_FALLBACK
    if not os.path.exists(target):
        print(f"[!] WARNING: Trained model not found at {target}")
        return False

    try:
        print(f"[*] Loading YOLO model from: {target}...")
        yolo_model = YOLO(target)
        print(f"[+] YOLO loaded: {yolo_model.names}")
        return True
    except Exception as e:
        print(f"[-] YOLO load error: {e}")
        return False

import zipfile
import tempfile
import shutil

def load_directory_model(dir_path, map_location='cpu'):
    if not os.path.isdir(dir_path):
        return torch.load(dir_path, map_location=map_location, weights_only=False)
    
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        print(f"[*] Zipping directory model for safe loading: {dir_path} -> {tmp_path}")
        parent_dir_name = os.path.basename(dir_path)
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_STORED) as z:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, dir_path)
                    z.write(full_path, os.path.join(parent_dir_name, rel_path))
        
        return torch.load(tmp_path, map_location=map_location, weights_only=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def load_classification_models():
    global classifier_model, classification_classes
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r") as f:
                classification_classes = json.load(f)
        except Exception as e:
            print(f"[!] Error loading class names: {e}")
    
    try:
        num_classes = len(classification_classes) if classification_classes else 38
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        if os.path.exists(EFFICIENTNET_PATH):
            try:
                state_dict = load_directory_model(EFFICIENTNET_PATH)
                if isinstance(state_dict, torch.nn.Module):
                    classifier_model = state_dict
                else:
                    model.load_state_dict(state_dict)
                    model.eval()
                    classifier_model = model
                return True
            except Exception as e:
                print(f"[!] Primary load failed, trying fallback...")
                return load_mobilenet_fallback()
    except Exception as e:
        print(f"[!] Critical error in classifier setup: {e}")
    return False

def load_mobilenet_fallback():
    global classifier_model
    try:
        model = models.mobilenet_v2(weights=None)
        num_classes = len(classification_classes) if classification_classes else 38
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.last_channel, num_classes)
        )
        if os.path.exists(MOBILENET_PATH):
            state_dict = load_directory_model(MOBILENET_PATH)
            model.load_state_dict(state_dict)
            model.eval()
            classifier_model = model
            return True
    except: pass
    return False

@app.get("/health")
def health():
    return {
        "status": "online",
        "yolo_ready": yolo_ready,
        "classifier_ready": classifier_ready,
        "supported_crops": ["Banana", "Chilli", "Radish", "Groundnut", "Cauliflower"]
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(default=""),
    mode: str = Form(default="auto")
):
    if not (yolo_ready or classifier_ready or vit_ready):
        raise HTTPException(status_code=503, detail="AI models are still loading. Please wait 1-2 minutes.")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    yolo_native_crops = ["banana", "chilli", "radish", "groundnut", "cauliflower"]
    crop_low = crop.lower() if crop else ""
    use_yolo = yolo_ready

    if crop_low and crop_low not in yolo_native_crops and classifier_ready:
        use_yolo = False

    if mode == "classification": use_yolo = False
    if mode == "yolo": use_yolo = True

    if use_yolo and yolo_ready:
        results = yolo_model(img, verbose=False)
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            conf = round(float(box.conf[0]) * 100, 2)
            label = yolo_model.names[int(box.cls[0])]
            
            if conf < 40.0:
                 if vit_ready: return run_vit_fallback(img, crop)
                 if classifier_ready: return run_classification(img, crop)
            
            return {"predicted_class": label, "confidence": conf, "crop": crop, "method": "yolo"}
        
        if vit_ready: return run_vit_fallback(img, crop)
        if classifier_ready: return run_classification(img, crop)
        return {"predicted_class": "Not Detected", "confidence": 0.0, "crop": crop}
    
    elif classifier_ready:
        return run_classification(img, crop)
    elif vit_ready:
        return run_vit_fallback(img, crop)
    
    raise HTTPException(status_code=503, detail="No models available")

def run_vit_fallback(img, crop):
    inputs = vit_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        conf, idx = torch.max(probs, -1)
    class_name = vit_model.config.id2label[idx.item()]
    label = class_name.split("___")[1].replace("_", " ") if "___" in class_name else class_name
    return {"predicted_class": label, "confidence": round(float(conf) * 100, 2), "crop": crop, "method": "vit"}

def run_classification(img, crop):
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = classifier_model(input_tensor)
        probs = F.softmax(output[0], dim=0)
        conf, idx = torch.max(probs, 0)
    class_name = classification_classes[idx.item()] if classification_classes else f"Class_{idx.item()}"
    return {"predicted_class": class_name, "confidence": round(float(conf) * 100, 2), "crop": crop, "method": "classification"}

@app.post("/youtube-search")
async def youtube_search(query: str = Form(...), language: str = Form(default="english")):
    try:
        results = search_videos(query=query, language=language)
        return {"success": True, "videos": results}
    except Exception as e:
        return {"success": False, "error": str(e), "videos": []}

@app.post("/identify-crop")
async def identify_crop(file: UploadFile = File(...)):
    if not vit_gen_ready: raise HTTPException(status_code=503, detail="Models loading")
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = vit_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        conf, idx = torch.max(probs, -1)
    label = vit_model.config.id2label[idx.item()]
    return {"relevant": True, "detectedCrop": label.split("___")[0] if "___" in label else label, "confidence": round(float(conf) * 100, 2)}

@app.get("/search-facilities")
async def search_facilities(lat: float, lon: float, radius: float = 50, city: str = None):
    results = get_hybrid_facilities(lat, lon, radius, city)
    return {"success": True, "data": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
