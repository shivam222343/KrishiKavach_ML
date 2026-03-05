import os
import time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("Importing torch...")
import torch
print("Importing transformers...")
import transformers
print("Importing ultralytics...")
import ultralytics
print("Staying alive for 10 seconds...")
time.sleep(10)
print("Finished.")
