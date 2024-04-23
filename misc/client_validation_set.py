"""import pandas as pd

# Load CSV file of CDV3 validation set
raw_df = pd.read_csv("./data/dataset_creation/dataframes/v3/df_val.csv")

# Filter DataFrame: HD is the minimum resolution
df = raw_df[(raw_df['img_w'] >= 1280) & (raw_df['img_h'] >= 720)]

# Get list of image names
img_names = df['img'].unique()

# Generate filtered dataset
import os
import shutil

data_root = '/data-fast/128-data1/ierregue/datasets'

# Create folder structure for the filtered dataset
os.makedirs(f'{data_root}/cdv3_hd_val/labels/val', exist_ok=True)
os.makedirs(f'{data_root}/cdv3_hd_val/images/val', exist_ok=True)
os.makedirs(f'{data_root}/cdv3_hd_val/annotations', exist_ok=True)

# Copy images and labels to the filtered dataset
for img_name in img_names:
    shutil.copy(f'{data_root}/custom_dataset_v3/labels/val/{img_name}.txt',
                f'{data_root}/cdv3_hd_val/labels/val/{img_name}.txt')

    shutil.copy(f'{data_root}/custom_dataset_v3/images/val/{img_name}.jpg',
                f'{data_root}/cdv3_hd_val/images/val/{img_name}.jpg')"""

# MANUALLY CREATE THE ANNOTATIONS FILE AND THE YAML FILE
# EVALUATE MODEL ON THE FILTERED DATASET AND MOVE IT TO THE NEW DATASET FOLDER

"""
# EVALUATE THE PREDICTIONS USING BRAMBOX
import brambox as bb
import numpy as np

data_root = '/data-fast/128-data1/ierregue/datasets'

# Label mapping
label_mapping = {
    'person': 0,
    'car': 1,
    'truck': 2,
    'uav': 3,
    'airplane': 4,
    'boat': 5
}

# Load the annotations
anns = bb.io.load('anno_coco',f'{data_root}/cdv3_hd_val/annotations/instances_val2017.json')
anns['image'] = anns['image'].astype(int)
anns['class_label'] = anns['class_label'].map(label_mapping)
anns['class_label'] = anns['class_label'].astype(int)

# Load the predictions
det = bb.io.load('det_coco', f'{data_root}/cdv3_hd_val/8sp2_150e_64b/predictions.json')
det['image'] = det['image'].astype(int)
det['class_label'] = det['class_label'].astype(int)

#det.image = det.image.cat.add_categories(set(anns.image.cat.categories) - set(det.image.cat.categories))

# Evaluate on every image
per_img_metrics = {
    'img': [],
    'mAP': [],
    'P': [],
    'R': [],
    'F1': []
}

for img in det['image'].unique():
    img_det = det[det['image'] == img]
    img_anns = anns[anns['image'] == img]


    # Evaluate the image
    evaluator = bb.eval.TIDE(img_det, img_anns)
    m_dets, m_anns = evaluator.compute_matches(img_det, img_anns, iou=0.5)
    tp = m_dets['tp'].sum()
    fp = m_dets['fp'].sum()

    p = tp / (tp + fp)
    r = tp / len(img_anns)
    f1 = 2 * (p * r) / (p + r)

    # Compute the metrics
    ap = evaluator.AP.mean()

    per_img_metrics['img'].append(img)
    per_img_metrics['mAP'].append(ap)
    per_img_metrics['P'].append(p)
    per_img_metrics['R'].append(r)
    per_img_metrics['F1'].append(f1)

#SAVE THE METRICS
import pandas as pd
per_img_metrics_df = pd.DataFrame(per_img_metrics)
per_img_metrics_df['F1'] = per_img_metrics_df['F1'].fillna(0)
per_img_metrics_df.to_csv(f'{data_root}/cdv3_hd_val/8sp2_150e_64b/per_img_metrics.csv', index=False)

"""
#"""
# READ PER IMAGE METRICS AND USE THEM TO FILTER AGAIN THE DATASET

import pandas as pd
import os
import shutil

data_root = '/data-fast/128-data1/ierregue/datasets'

per_img_metrics_df = pd.read_csv(f'{data_root}/cdv3_hd_val/8sp2_150e_64b/per_img_metrics.csv')

def create_and_export(df, top_k=2000, metric='mAP'):
    # Sort by metric and select top k images
    df = df.sort_values(by=metric, ascending=False)
    top = df.head(top_k)

    # Read original dataset
    raw_df = pd.read_csv(f"./data/dataset_creation/dataframes/v3/df_val.csv")

    # Select only the images with highers metric
    df_filtered = raw_df[raw_df['img'].isin(top['img'].values)]

    # Compute bbox area and size category
    df_filtered['bbox_area'] = df_filtered['wn'] * df_filtered['img_w'] * df_filtered['hn'] * df_filtered['img_h']
    bin_edges = [0, 16 ** 2, 32 ** 2, 96 ** 2, float('inf')]
    bin_labels = ['Tiny', 'Small', 'Medium', 'Large']
    df_filtered['bbox_size_category'] = pd.cut(df_filtered['bbox_area'], bins=bin_edges, labels=bin_labels, right=False, retbins=False)

    # Compute statistics
    print(f"Statistics for {metric}")
    print(f"Number of images: {len(df_filtered.img.unique())}")
    print(f"Number of annotations: {len(df_filtered)}")
    print(f"Mean number of annotations per image: {len(df_filtered) / len(df_filtered.img.unique())}")
    print(df_filtered.class_name.value_counts())
    print(df_filtered.bbox_size_category.value_counts())

    # Create the new dataset structure for metric dataset
    os.makedirs(f'{data_root}/cdv3_hd_val_top{top_k}_{metric}/labels/val', exist_ok=True)
    os.makedirs(f'{data_root}/cdv3_hd_val_top{top_k}_{metric}/images/val', exist_ok=True)
    os.makedirs(f'{data_root}/cdv3_hd_val_top{top_k}_{metric}/annotations', exist_ok=True)

    # Copy images and labels to the filtered dataset
    for img_name in df_filtered['img'].unique():
        shutil.copy(f'{data_root}/custom_dataset_v3/labels/val/{img_name}.txt',
                    f'{data_root}/cdv3_hd_val_top{top_k}_{metric}/labels/val/{img_name}.txt')

        shutil.copy(f'{data_root}/custom_dataset_v3/images/val/{img_name}.jpg',
                    f'{data_root}/cdv3_hd_val_top{top_k}_{metric}/images/val/{img_name}.jpg')

metrics = ['mAP', 'P', 'R', 'F1']
for metric in metrics:
    create_and_export(per_img_metrics_df, top_k=2000, metric=metric)


#"""


