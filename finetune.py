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
model = YOLO('yolov8s-p2-emb.yaml', task='detectemb').load('./../models/yolov8s.pt')

epochs = 50
batch = 16
optimizer = 'auto'

model.train(
    resume=False,
    data='custom_dataset-tagged.yaml',
    epochs=epochs,
    batch=batch,
    optimizer=optimizer,
    save=True,
    #fraction=0.5,
    save_json=True,
    plots=True,
    device=[0],
    project='reid-embeddings',
    imgsz=640,
    name=f'8sp2-{epochs}e-{batch}b-TELight',
    verbose=True,
    patience=25,
    cache=False,
    amp=False
)