import torch
import torchvision.models as models
import json
import os

def test_load():
    # Attempt to load EfficientNet
    try:
        # Define model architecture
        model = models.efficientnet_b0(pretrained=False)
        # Modify the head
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, 38)
        
        # Point to the data.pkl or use the directory if torch handles it
        model_path = "pretrained_models/efficientnet_plant.pth"
        
        # If it's a directory, we might need to load the components or it's a special format
        # Let's try to load it using torch.load
        # NOTE: On some systems, unzipped torch saves are directories.
        # We might need to point to the directory if torch supports it, or use a workaround.
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ EfficientNet loaded successfully with load_state_dict")
    except Exception as e:
        print(f"❌ EfficientNet failed: {e}")

    try:
        # Try loading directly (maybe it's a scripted model?)
        model = torch.load("pretrained_models/efficientnet_plant.pth", map_location='cpu')
        print(f"✅ EfficientNet loaded directly: {type(model)}")
    except Exception as e:
        print(f"❌ EfficientNet direct load failed: {e}")

if __name__ == "__main__":
    test_load()
