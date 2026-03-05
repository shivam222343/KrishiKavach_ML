import torch
import json

def test_load():
    path = "pretrained_models/efficientnet_plant.pth/efficientnet_plant/data.pkl"
    try:
        # data.pkl in a torch save is usually the metadata, but sometimes it works
        data = torch.load(path, map_location='cpu')
        print(f"✅ Loaded {path}")
        print(f"Keys: {list(data.keys())[:5]}")
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")

if __name__ == "__main__":
    test_load()
