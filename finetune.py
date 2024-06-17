import os
import comet_ml
from ultralytics import YOLO

# Set number of threads
N_THREADS = '8'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS

# Initialize COMET logger, first log done in notebook where API key was asked, now seems to be saved in .comet.config
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('yolov8s-p2.yaml', task='detect').load('./../models/yolov8s.pt')

epochs = 300
batch = 128
optimizer = 'auto'

model.train(
    resume=False,
    data='custom_dataset.yaml',
    epochs=epochs,
    batch=batch,
    optimizer=optimizer,
    save=True,
    #exp_lr=True,
    #close_mosaic=5,
    #fraction=0.5,
    save_json=True,
    plots=True,
    device=[0,1,2,3,4,5,6,7],
    project='fine-tune-cdv3',
    imgsz=640,
    name=f'8sp2-300e-128b-{optimizer}',
    verbose=True,
    patience=25,
    cache=False,
    amp=False
)