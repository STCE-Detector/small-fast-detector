import os
from ultralytics import YOLO
import comet_ml
from ultralytics.utils import SETTINGS

# Only for tune method because it uses all GPUs by default
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True  # set True to log using Comet.ml
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('yolov8s-p2-emb.yaml', task='detectemb').load('./../models/yolov8s.pt')
results = model.train(
    save=False,
    verbose=True,
    plots=False,
    project='debug',
    name='8s',
    data='coco8-tagged.yaml',
    epochs=100,
    batch=4,
    imgsz=320,
)