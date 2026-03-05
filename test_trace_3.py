print("START")
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
print("ENV SET")
import torch
print("TORCH DONE")
from ultralytics import YOLO
print("YOLO DONE")
from transformers import ViTForImageClassification, ViTImageProcessor
print("TRANSFORMERS DONE")
