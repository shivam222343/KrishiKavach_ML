import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "T"
os.environ["FOR_IGNORE_EXCEPTIONS"] = "T"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
print("ENV SET")
from transformers import ViTForImageClassification, ViTImageProcessor
print("TRANSFORMERS DONE")
