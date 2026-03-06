import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
from ultralytics import YOLO
import json
import zipfile
import tempfile
from PIL import Image

class MultiModelEnsemble:
    def __init__(self, yolo_path, eff_path, mob_path, class_names_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_path = yolo_path
        self.eff_path = eff_path
        self.mob_path = mob_path
        self.class_names_path = class_names_path
        
        self.yolo_model = None
        self.eff_model = None
        self.mob_model = None
        self.class_names = []
        
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.load_all()

    def load_directory_model(self, dir_path):
        if not os.path.isdir(dir_path):
            return torch.load(dir_path, map_location=self.device, weights_only=False)
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            print(f"[*] Zipping directory model into 'archive/' structure: {dir_path} -> {tmp_path}")
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_STORED) as z:
                # PyTorch 1.6+ ZIP format expects a top-level folder (usually named 'archive')
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, dir_path)
                        # Standard PyTorch internal root is 'archive'
                        z.write(full_path, os.path.join('archive', rel_path))
            
            return torch.load(tmp_path, map_location=self.device, weights_only=False)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_all(self):
        # Load Class Names
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, "r") as f:
                self.class_names = json.load(f)
        
        num_classes = len(self.class_names) if self.class_names else 38
        
        # Load YOLO
        try:
            if os.path.exists(self.yolo_path):
                self.yolo_model = YOLO(self.yolo_path)
                print(f"[+] YOLO loaded from {self.yolo_path}")
        except Exception as e:
            print(f"[-] YOLO load error: {e}")

        # Load EfficientNet-B0
        try:
            model_eff = models.efficientnet_b0(weights=None)
            in_features = model_eff.classifier[1].in_features
            model_eff.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes)
            )
            if os.path.exists(self.eff_path):
                state_dict = self.load_directory_model(self.eff_path)
                if isinstance(state_dict, torch.nn.Module):
                    self.eff_model = state_dict
                else:
                    model_eff.load_state_dict(state_dict)
                    model_eff.eval()
                    self.eff_model = model_eff.to(self.device)
                print(f"[+] EfficientNet loaded from {self.eff_path}")
        except Exception as e:
            print(f"[-] EfficientNet load error: {e}")

        # Load MobileNetV2
        try:
            model_mob = models.mobilenet_v2(weights=None)
            model_mob.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(model_mob.last_channel, num_classes)
            )
            if os.path.exists(self.mob_path):
                state_dict = self.load_directory_model(self.mob_path)
                if isinstance(state_dict, torch.nn.Module):
                    self.mob_model = state_dict
                else:
                    model_mob.load_state_dict(state_dict)
                    model_mob.eval()
                    self.mob_model = model_mob.to(self.device)
                print(f"[+] MobileNet loaded from {self.mob_path}")
        except Exception as e:
            print(f"[-] MobileNet load error: {e}")

    def predict(self, img, user_crop=None):
        results = []
        
        # 1. Image Preprocessing for PyTorch
        input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # 2. Parallel Inference
        # YOLOv8
        yolo_native_crops = ["banana", "chilli", "radish", "groundnut", "cauliflower"]
        if self.yolo_model and (not user_crop or user_crop.lower() in yolo_native_crops):
            yolo_res = self.yolo_model(img, verbose=False)
            if yolo_res and len(yolo_res[0].boxes) > 0:
                box = yolo_res[0].boxes[0]
                conf = float(box.conf[0]) * 100
                label = self.yolo_model.names[int(box.cls[0])]
                results.append({"model": "YOLOv8", "class": label, "confidence": conf})

        # EfficientNet-B0
        if self.eff_model:
            with torch.no_grad():
                output = self.eff_model(input_tensor)
                probs = F.softmax(output[0], dim=0)
                conf, idx = torch.max(probs, 0)
                label = self.class_names[idx.item()] if self.class_names else f"Class_{idx.item()}"
                results.append({"model": "EfficientNet-B0", "class": label, "confidence": float(conf) * 100})

        # MobileNetV2
        if self.mob_model:
            with torch.no_grad():
                output = self.mob_model(input_tensor)
                probs = F.softmax(output[0], dim=0)
                conf, idx = torch.max(probs, 0)
                label = self.class_names[idx.item()] if self.class_names else f"Class_{idx.item()}"
                results.append({"model": "MobileNetV2", "class": label, "confidence": float(conf) * 100})

        if not results:
            return {"predicted_class": "Not Detected", "confidence": 0.0, "method": "none"}

        # 3. Conflict Resolution (The "Winner" Logic)
        # Filter by user crop if provided
        final_winner = None
        if user_crop:
            user_crop_low = user_crop.lower()
            filtered_results = [r for r in results if user_crop_low in r["class"].lower()]
            if filtered_results:
                final_winner = max(filtered_results, key=lambda x: x["confidence"])
        
        # If no result matches user crop specifically, or no user crop provided, pick highest confidence overall
        if not final_winner:
            final_winner = max(results, key=lambda x: x["confidence"])

        # Format label (PlantVillage style: Crop___Disease -> Disease)
        display_label = final_winner["class"]
        if "___" in display_label:
            parts = display_label.split("___")
            display_label = parts[1].replace("_", " ") if len(parts) > 1 else parts[0]
        
        return {
            "predicted_class": display_label,
            "confidence": round(final_winner["confidence"], 2),
            "method": f"ensemble ({final_winner['model']})",
            "full_results": results # For debugging
        }
