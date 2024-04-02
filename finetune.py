import os
import comet_ml
from ultralytics import YOLO

# Set number of threads
N_THREADS = '12'
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

epochs = 50
batch = 16
fraction = 0.5
optimizer = 'auto'

model.train(
    resume=False,
    data='custom_dataset.yaml',
    epochs=epochs,
    batch=batch,
    optimizer=optimizer,
    save=True,
    fraction=fraction,
    save_json=True,
    plots=True,
    device=[0,1],
    project='boosting_v8sp2',
    name=f'{epochs}e-{batch}b-{fraction}f-{optimizer}-6pf',
    verbose=True,
    patience=25,
    cache=False,
    amp=False,
    ##############
    #scale=0.3,
    #mixup=0.15,
    #copy_paste=0.3,
    #close_mosaic=25,
    #mosaic=0.0,
)