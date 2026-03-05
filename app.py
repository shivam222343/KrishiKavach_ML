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

# Windows Stability Flags
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"
# Disable Fortran/MKL console handler which causes crashes on background window signals
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "T"
os.environ["FOR_IGNORE_EXCEPTIONS"] = "T"
os.environ["FOR_FORCE_STACK_TRACE"] = "F"
# Force torch to be non-interactive
os.environ["PYTHONUNBUFFERED"] = "1"

import io
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import uvicorn
from youtube_search import search_videos
from transformers import ViTForImageClassification, ViTImageProcessor
import torch.nn.functional as F
from scraper_service import get_hybrid_facilities

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
YOLO_FALLBACK = os.path.join(BASE_DIR, "yolov8s.pt")

CLASS_NAMES_PATH = os.path.join(BASE_DIR, "pretrained_models", "class_names.json")
EFFICIENTNET_PATH = os.path.join(BASE_DIR, "pretrained_models", "efficientnet_plant.pth", "efficientnet_plant")
MOBILENET_PATH = os.path.join(BASE_DIR, "pretrained_models", "mobilenetv2_plant.pth", "mobilenetv2_plant")

app = FastAPI(title="Krishi Kavach Unified ML Server", version="3.0")

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
        if os.path.exists(FALLBACK_MODEL):
            print(f"[*] Using fallback base model: {FALLBACK_MODEL}")
            target = FALLBACK_MODEL
        else:
            print(f"[X] CRITICAL ERROR: No model found at {target} or {FALLBACK_MODEL}")
            print(f"[*] Please ensure you have copied the 'runs' folder from your training directory.")
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
    """
    On Windows, torch.load often fails on unzipped directories with PermissionError.
    This helper zips the directory into a temporary file and loads it.
    """
    if not os.path.isdir(dir_path):
        return torch.load(dir_path, map_location=map_location, weights_only=False)
    
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        print(f"[*] Zipping directory model for safe loading: {dir_path} -> {tmp_path}")
        # PyTorch 1.6+ zip format EXPECTS all files to be under a single root folder in the zip.
        # The name of this folder is usually what torch.load uses as a prefix.
        parent_dir_name = os.path.basename(dir_path)
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_STORED) as z:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    # We MUST preserve the inner folder structure but prefixed with a single root
                    rel_path = os.path.relpath(full_path, dir_path)
                    z.write(full_path, os.path.join(parent_dir_name, rel_path))
        
        return torch.load(tmp_path, map_location=map_location, weights_only=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def load_classification_models():
    global classifier_model, classification_classes
    
    # Load class names
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r") as f:
                classification_classes = json.load(f)
            print(f"[*] Loaded {len(classification_classes)} classification classes")
        except Exception as e:
            print(f"[!] Error loading class names: {e}")
    
    # Try loading EfficientNet
    try:
        print("[*] Setting up EfficientNet model architecture...")
        num_classes = len(classification_classes) if classification_classes else 38
        model = models.efficientnet_b0(weights=None)
        
        # Adjusting to match the unexpected key: classifier.1.1
        # This implies classifier[1] is a Sequential or similar
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        load_path = EFFICIENTNET_PATH
        print(f"[*] Target load path: {load_path}")
        
        if os.path.exists(load_path):
            try:
                print(f"[*] Calling robust loader on: {load_path}")
                state_dict = load_directory_model(load_path)
                print("[+] Successfully loaded state_dict/model from path")
                if isinstance(state_dict, torch.nn.Module):
                    classifier_model = state_dict
                else:
                    model.load_state_dict(state_dict)
                    model.eval()
                    classifier_model = model
                print("[+] EfficientNet Classifier initialized")
                return True
            except Exception as e:
                print(f"[!] Primary load failed: {type(e).__name__}: {e}")
                
                inner_path = os.path.join(load_path, "efficientnet_plant", "data.pkl")
                if os.path.exists(inner_path):
                    try:
                        print(f"[*] Trying inner path: {inner_path}")
                        state_dict = torch.load(inner_path, map_location='cpu', weights_only=False)
                        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                            model.load_state_dict(state_dict["model_state_dict"])
                        else:
                            model.load_state_dict(state_dict)
                        model.eval()
                        classifier_model = model
                        print("[+] EfficientNet loaded from inner file")
                        return True
                    except Exception as e2:
                        print(f"[!] Inner load failed: {e2}")
                
                print("[*] Falling back to MobileNet...")
                return load_mobilenet_fallback()
        else:
            print(f"[!] EfficientNet path does not exist: {load_path}")
    except Exception as e:
        print(f"[!] Critical error in classifier setup: {e}")
    
    return False

def load_mobilenet_fallback():
    global classifier_model
    try:
        print("[*] Attempting MobileNet fallback...")
        model = models.mobilenet_v2(weights=None)
        num_classes = len(classification_classes) if classification_classes else 38
        
        # Match classifier.1.1
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.last_channel, num_classes)
        )
        
        load_path = MOBILENET_PATH
        if os.path.exists(load_path):
            path_to_try = load_path
            if os.path.isdir(load_path):
                inner = os.path.join(load_path, "mobilenetv2_plant", "data.pkl")
                if os.path.exists(inner):
                    path_to_try = inner
            
            try:
                print(f"[*] Loading MobileNet from: {path_to_try} using robust loader")
                state_dict = load_directory_model(path_to_try)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    model.load_state_dict(state_dict["model_state_dict"])
                else:
                    model.load_state_dict(state_dict)
                model.eval()
                classifier_model = model
                print("[+] MobileNet Classifier loaded")
                return True
            except Exception as e:
                print(f"[!] MobileNet load failed: {e}")
    except Exception as e:
        print(f"[!] MobileNet setup error: {e}")
    return False

# Load models at startup
yolo_ready = load_yolo()
classifier_ready = load_classification_models()
vit_ready = load_vit()
vit_gen_ready = load_vit_general()

# ── Transforms ────────────────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "online",
        "yolo_active": yolo_ready,
        "classifier_active": classifier_ready,
        "yolo_classes": yolo_model.names if yolo_ready else {},
        "classifier_classes": classification_classes,
        "supported_crops": ["Banana", "Chilli", "Radish", "Groundnut", "Cauliflower", "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"]
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(default=""),
    mode: str = Form(default="auto") # auto | yolo | classification
):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="PDF detected. Please upload an image of the leaf (JPG, PNG, or WebP) for AI disease detection.")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {file.content_type}. Please use JPG, PNG, or WebP.")

    # ── Determining which model to use (Smart Mapping) ────────────────────────
    yolo_native_crops = ["banana", "chilli", "radish", "groundnut", "cauliflower"]
    classification_native_crops = ["apple", "blueberry", "cherry", "corn", "grape", "orange", "peach", "pepper", "potato", "raspberry", "soybean", "squash", "strawberry", "tomato"]
    
    crop_low = crop.lower() if crop else ""
    use_yolo = yolo_ready

    if crop_low:
        if crop_low in yolo_native_crops:
            use_yolo = True
        elif crop_low in classification_native_crops and classifier_ready:
            use_yolo = False
        else:
            # Fallback based on class name matching
            yolo_supports_crop = any(crop_low in v.lower() for v in yolo_model.names.values()) if yolo_ready else False
            if not yolo_supports_crop and classifier_ready:
                use_yolo = False

    if mode == "classification": use_yolo = False
    if mode == "yolo": use_yolo = True

    if use_yolo and yolo_ready:
        results = yolo_model(img, verbose=False)
        box = None
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
        
        yolo_conf = round(float(box.conf[0]) * 100, 2) if box else 0.0
        yolo_class = yolo_model.names[int(box.cls[0])] if box else "Not Detected"
        
        # ── SMART FALLBACK & SECOND OPINION ──
        # Trigger Fallback if:
        # 1. No box was found (might be a disease YOLO doesn't know)
        # 2. YOLO confidence is very low (< 40%)
        # 3. YOLO says "Healthy" but confidence is not absolute (< 95%) - to catch missed diseases
        needs_second_opinion = (
            not box or 
            yolo_conf < 40.0 or 
            (yolo_class.lower() == "healthy" and yolo_conf < 95.0)
        )

        if needs_second_opinion:
            if vit_ready:
                print(f"[*] YOLO Uncertainty (Class: {yolo_class}, Conf: {yolo_conf}%). Consulting ViT Second Opinion...")
                vit_res = run_vit_fallback(img, crop)
                # If ViT finds a disease when YOLO said healthy or nothing, prefer ViT
                if not vit_res["details"]["is_healthy"] or not box:
                    print(f"[+] ViT override: {vit_res['predicted_class']} ({vit_res['confidence']}%)")
                    return vit_res
            
            if classifier_ready and (not box or yolo_conf < 40.0):
                print(f"[*] Consulting local EfficientNet...")
                return run_classification(img, crop)

        if not box:
             print(f"[*] YOLO found no bounding box for {crop}. Falling back to ViT/Classifier...")
             if vit_ready: 
                 return run_vit_fallback(img, crop)
             if classifier_ready:
                 return run_classification(img, crop)
             return {"predicted_class": "Not Detected", "confidence": 0.0, "crop": crop, "method": "none"}

        return {
            "predicted_class": yolo_class,
            "confidence": yolo_conf,
            "crop": crop,
            "method": "yolo"
        }
    
    elif classifier_ready:
        return run_classification(img, crop)
    
    elif vit_ready:
        print("[*] No local models definitive. Consulting Hugging Face ViT...")
        return run_vit_fallback(img, crop)
    
    else:
        raise HTTPException(status_code=503, detail="No ML models available")

def run_vit_fallback(img, crop):
    print(f"[*] Running ViT Fallback for crop: {crop}")
    inputs = vit_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, -1)
        
    class_name = vit_model.config.id2label[idx.item()]
    
    # Heuristic: PlantVillage labels usually look like "Tomato___Bacterial_spot"
    # We strip the crop name to make it look consistent with our UI
    short_label = class_name
    if "___" in class_name:
        parts = class_name.split("___")
        short_label = parts[1].replace("_", " ")

    is_healthy = "healthy" in class_name.lower()
    
    return {
        "predicted_class": short_label,
        "confidence": round(float(conf) * 100, 2),
        "crop": crop,
        "method": "vit_fallback",
        "yield_estimation": "90-100%" if is_healthy else "60-80%",
        "details": {
            "is_healthy": is_healthy,
            "original_label": class_name,
            "source": "Hugging Face ViT"
        }
    }

def run_classification(img, crop):
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = classifier_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        conf, idx = torch.max(probabilities, 0)
        
    class_name = classification_classes[idx.item()] if classification_classes else f"Class_{idx.item()}"
    
    # Yield Estimation Logic (Heuristic based on health)
    predicted_healthy = "healthy" in class_name.lower()
    yield_est = "90-100%" if predicted_healthy else "60-80%"
    
    return {
        "predicted_class": class_name,
        "confidence": round(float(conf) * 100, 2),
        "crop": crop,
        "method": "classification",
        "yield_estimation": yield_est,
        "details": {
            "is_healthy": predicted_healthy,
            "impact_on_yield": "Low" if predicted_healthy else "High"
        }
    }

@app.post("/youtube-search")
async def youtube_search(
    query: str = Form(...),
    language: str = Form(default="english"),
    max_duration: int = Form(default=20)
):
    try:
        results = search_videos(query=query, language=language, max_duration_minutes=max_duration)
        return {"success": True, "videos": results}
    except Exception as e:
        print(f"YouTube search error: {e}")
        return {"success": False, "error": str(e), "videos": []}

@app.post("/identify-crop")
async def identify_crop(file: UploadFile = File(...)):
    if not vit_gen_ready or not vit_ready:
        raise HTTPException(status_code=503, detail="Identification models not ready")
        
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 1. Check Relevance (Is it a plant/leaf/crop?)
    inputs_gen = vit_gen_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs_gen = vit_gen_model(**inputs_gen)
        # Check top-10 classes (increased from 5) for anything plant-related
        probs = F.softmax(outputs_gen.logits, dim=-1)
        top10_conf, top10_idx = torch.topk(probs, 10)
        
    is_relevant = False
    # Broader keyword list to allow for hands/backgrounds
    plant_keywords = ["leaf", "plant", "tree", "grass", "corn", "maize", "fruit", "veggie", "vegetable", "crop", "banana", "green", "agriculture"]
    for i in range(10):
        label = vit_gen_model.config.id2label[top10_idx[0][i].item()].lower()
        if any(kw in label for kw in plant_keywords):
            is_relevant = True
            break
            
    if not is_relevant:
        return {"relevant": False, "message": "The uploaded image does not appear to be a crop or leaf. Please upload a clear photo of your plant."}

    # 2. Identify Specific Crop
    inputs_crop = vit_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs_crop = vit_model(**inputs_crop)
        probs_crop = F.softmax(outputs_crop.logits, dim=-1)
        conf, idx = torch.max(probs_crop, -1)
        
    full_label = vit_model.config.id2label[idx.item()] # e.g. "Tomato___Bacterial_spot"
    crop_name = full_label.split("___")[0] if "___" in full_label else full_label
    
    # Map to our standard IDs (Fuzzy/Partial match)
    crop_mapping = {
        "Apple": "Apple",
        "Blueberry": "Blueberry",
        "Cherry": "Cherry",
        "Corn": "Corn",
        "Maize": "Corn",
        "Grape": "Grape",
        "Orange": "Orange",
        "Peach": "Peach",
        "Pepper": "Pepper",
        "Potato": "Potato",
        "Raspberry": "Raspberry",
        "Soybean": "Soybean",
        "Squash": "Squash",
        "Strawberry": "Strawberry",
        "Tomato": "Tomato",
        "Banana": "Banana",
        "Chilli": "Chilli",
        "Radish": "Radish",
        "Groundnut": "Groundnut",
        "Peanut": "Groundnut",
        "Cauliflower": "Cauliflower"
    }
    
    # Try direct match
    detected_id = crop_mapping.get(crop_name, "Other")
    
    # Try partial match if no direct match found
    if detected_id == "Other":
        for key, val in crop_mapping.items():
            if key.lower() in crop_name.lower():
                detected_id = val
                break
    
    return {
        "relevant": True, 
        "detectedCrop": detected_id, 
        "confidence": round(float(conf) * 100, 2),
        "label": full_label
    }

@app.get("/search-facilities")
async def search_facilities(lat: float, lon: float, radius: float = 50, city: str = None):
    try:
        results = get_hybrid_facilities(lat, lon, radius, city)
        return {"success": True, "count": len(results), "data": results}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
