import os

from sklearn.cluster import KMeans

from ultralytics import YOLO
import comet_ml
import torch
from ultralytics.utils import SETTINGS
from torchvision.ops import roi_align

# Only for tune method because it uses all GPUs by default
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True  # set True to log using Comet.ml
comet_ml.init()

"""# Initialize model and load matching weights
model = YOLO('yolov8s.yaml', task='detect').load('./../models/yolov8s.pt')
results = model.train(
    save=True,
    verbose=True,
    plots=True,
    project='debug',
    name='8s',
    data='coco8.yaml',
    epochs=2,
    batch=4,
    imgsz=320,
)"""


model = YOLO('/Users/inaki-eab/Desktop/small-fast-detector/inference_tools/models/8sp2_150.pt', task='detect')
# Inference
results = model(
    '/Users/inaki-eab/Desktop/small-fast-detector/inference_tools/data/8001.jpg',
    imgsz=640,
    device='cpu',
    project='./',
    name='test',
    ######################
    visualize=False,
    embed=[18] # index of the layer
)

embeddings = model.model.embeddings

# Normalized bboxes
original = results[0].boxes.xyxyn
# Match feature map scale
original[:, [0, 2]] *= embeddings[0].shape[-1]
original[:, [1, 3]] *= embeddings[0].shape[-2]
# Add batch dimension
batch_assignation = torch.zeros((original.shape[0], 1))

roi_embeddings = roi_align(
    input=embeddings[0],
    boxes=torch.cat((batch_assignation, original), dim=1),
    output_size=(1, 1),     # 1x1 is basically the same as max pooling in every channel
    spatial_scale=1,    # Because the boxes are already in the same scale as the feature map
    aligned=True
)

kmeans = KMeans(n_clusters=2, random_state=0).fit(roi_embeddings.squeeze().detach().numpy())




# TODO:
# 1. Store embeddings in a df with the real classes and image names
# 2. Cluster embeddings up to num_classes
# 3. Compare against the real classes
# 4. Try with different layers, scales, operations, etc.

x=0
