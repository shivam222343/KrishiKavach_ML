print("START")
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
print("ENV SET")
from transformers import ViTForImageClassification, ViTImageProcessor
print("TRANSFORMERS DONE")
