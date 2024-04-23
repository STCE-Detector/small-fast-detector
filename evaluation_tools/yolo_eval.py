import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
from ultralytics import YOLO


# SETTING UP PARAMETERS
# Better not to change these parameters
dataset_root = './../../ultralytics/cfg/datasets/custom_dataset.yaml' #'./data/client_test/data.yaml'
#dataset_root = './data/client_test/data.yaml'
model_path = './models/v3/8sp2_150e_64b.pt'
outputs_root = './outputs'
experiment_name = '8sp2_150e_64b@1280'
#experiment_name = time.strftime("%Y%m%d-%H%M%S")
# Can be changed
imgsz = 1280
batch = 32
device = [0]


#  START OF EVALUATION
print("🚀...WELCOME TO EVALUATION DETECTOR MODEL...")

print("🚀...Initializing model...")
model = YOLO(model_path, task='detect')

print("🚀...INFERENCE MODE...🚀")
print("📦...GETTING PREDICTIONS...📦")
metrics = model.val(
    data= dataset_root,
    imgsz=imgsz,
    batch=batch,
    device=device,
    iou=0.7,
    save=True,
    save_json=True,
    plots=True,
    save_txt=False,      # Text files
    save_conf=False,     # Save confidences
    # save results to project/name relative to script directory or absolute path
    project=outputs_root,
    name=experiment_name,
)
