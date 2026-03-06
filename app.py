"""
Krishi Kavach – Unified ML Inference Server
Supports: 
1. Tri-Model Ensemble (YOLOv8, MobileNetV2, EfficientNet-B0)
2. ViT General for Crop Identification
3. AI Retrieval Layer (via Backend + Gemini)

Run: python app.py (starts on http://localhost:8000)
"""

# Standard Imports
import os
import sys
import io
import json
import uvicorn
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Pre-declare globals
ensemble = None
vit_gen_ready = False
vit_gen_model = None
vit_gen_processor = None

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
YOLO_FALLBACK = os.path.join(BASE_DIR, "yolov8s.pt")

CLASS_NAMES_PATH = os.path.join(BASE_DIR, "pretrained_models", "class_names.json")
EFFICIENTNET_PATH = os.path.join(BASE_DIR, "pretrained_models", "efficientnet_plant.pth", "efficientnet_plant")
MOBILENET_PATH = os.path.join(BASE_DIR, "pretrained_models", "mobilenetv2_plant.pth", "mobilenetv2_plant")

VIT_GENERAL_NAME = "google/vit-base-patch16-224"

def background_model_loading():
    global ensemble, vit_gen_ready, vit_gen_model, vit_gen_processor
    
    print("[*] Loading AI models in a background thread...")
    try:
        # Heavy Imports
        from multi_model import MultiModelEnsemble
        from transformers import ViTForImageClassification, ViTImageProcessor
        
        # Load Ensemble
        yolo_target = YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else YOLO_FALLBACK
        ensemble = MultiModelEnsemble(
            yolo_path=yolo_target,
            eff_path=EFFICIENTNET_PATH,
            mob_path=MOBILENET_PATH,
            class_names_path=CLASS_NAMES_PATH
        )
        
        # Load General ViT for /identify-crop
        print(f"[*] Loading General ViT Model ({VIT_GENERAL_NAME})...")
        vit_gen_processor = ViTImageProcessor.from_pretrained(VIT_GENERAL_NAME)
        vit_gen_model = ViTForImageClassification.from_pretrained(VIT_GENERAL_NAME)
        vit_gen_model.eval()
        vit_gen_ready = True
        
        print(f"[+] AI Engine ready. Tri-Model Ensemble initialized.")
    except Exception as e:
        print(f"[-] Startup Failure: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    print("[*] Web server starting. Backgrounding model load for Render compatibility...")
    thread = threading.Thread(target=background_model_loading)
    thread.start()
    yield

app = FastAPI(title="Krishi Kavach Unified ML Server", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "online" if ensemble else "loading",
        "ensemble_ready": ensemble is not None,
        "yolo_active": ensemble.yolo_model is not None if ensemble else False,
        "efficientnet_active": ensemble.eff_model is not None if ensemble else False,
        "mobilenet_active": ensemble.mob_model is not None if ensemble else False,
        "vit_ready": vit_gen_ready,
        "supported_crops": ["Banana", "Chilli", "Radish", "Groundnut", "Cauliflower"]
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(default=""),
    mode: str = Form(default="auto")
):
    if not ensemble:
        raise HTTPException(status_code=503, detail="AI models are still loading. Please wait 1-2 minutes.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Use Ensemble for inference
        result = ensemble.predict(img, user_crop=crop)
        return result
    except Exception as e:
        print(f"[-] Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify-crop")
async def identify_crop(file: UploadFile = File(...)):
    if not vit_gen_ready:
        raise HTTPException(status_code=503, detail="Identification model loading")
    
    try:
        import torch
        import torch.nn.functional as F
        
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        inputs = vit_gen_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_gen_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            conf, idx = torch.max(probs, -1)
            
        label = vit_gen_model.config.id2label[idx.item()]
        return {
            "relevant": True, 
            "detectedCrop": label, 
            "confidence": round(float(conf) * 100, 2)
        }
    except Exception as e:
        print(f"[-] Identification Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/youtube-search")
async def youtube_search(query: str = Form(...), language: str = Form(default="english")):
    try:
        from youtube_search import search_videos
        results = search_videos(query=query, language=language)
        return {"success": True, "videos": results}
    except Exception as e:
        return {"success": False, "error": str(e), "videos": []}

@app.get("/search-facilities")
async def search_facilities(lat: float, lon: float, radius: float = 50, city: str = None):
    from scraper_service import get_hybrid_facilities
    results = get_hybrid_facilities(lat, lon, radius, city)
    return {"success": True, "data": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
