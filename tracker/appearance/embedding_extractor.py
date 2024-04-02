import os
import torch
import numpy as np
import pandas as pd
from torchvision.ops import roi_align
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy

data_path = '/data-fast/128-data1/ierregue/datasets/custom_dataset_v2/images/val/'
model_path = './../models/8sp2_150.pt'

model = YOLO(model_path, task='detect')

# Inference
files = os.listdir(data_path)
image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

data = []
for image_file in image_files:
    results = model(
        data_path + image_file,
        imgsz=640,
        device='cpu',
        project='./',
        name='test',
        embed=[18]  # index of the layer
    )

    embeddings = model.model.embeddings

    # Normalized bboxes
    original = xywh2xyxy(results[0].boxes.xywhn)
    # Match feature map scale
    original[:, [0, 2]] *= embeddings[0].shape[-1]
    original[:, [1, 3]] *= embeddings[0].shape[-2]
    # Add batch dimension
    batch_assignation = torch.zeros((original.shape[0], 1))

    roi_embeddings = roi_align(
        input=embeddings[0],
        boxes=torch.cat((batch_assignation, original), dim=1),
        output_size=(1, 1),  # 1x1 is basically the same as max pooling in every channel
    )

    num_preds = len(results[0])
    for i in range(num_preds):
        data.append({
            'image_file': image_file,
            'class': int(results[0].boxes.cls[i]),
            'bbox': results[0].boxes.xywhn[i].numpy(),
            'embedding': roi_embeddings[i].squeeze().numpy()
        })

df = pd.DataFrame(data)

# Expand "bbox" and "embedding" arrays into separate columns
bbox_cols = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
embedding_cols = [f'embedding_{i}' for i in range(len(df['embedding'][0]))]

df[bbox_cols] = pd.DataFrame(df['bbox'].tolist(), index=df.index)
df[embedding_cols] = pd.DataFrame(df['embedding'].tolist(), index=df.index)

# Drop original "bbox" and "embedding" columns
df.drop(columns=['bbox', 'embedding'], inplace=True)

# Save dataframe to CSV
df.to_csv('embeddings.csv', index=False)

