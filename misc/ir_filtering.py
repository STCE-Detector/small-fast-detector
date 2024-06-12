import os
import brambox as bb
import shutil
import pandas as pd

dataset = 'anti-uav'
"""
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
anns = bb.io.load('anno_coco', f'/Users/inaki-eab/Desktop/IR/{dataset}/annotations/instances_val2017.json')
#anns['image'] = anns['image'].astype(int)
anns['class_label'] = anns['class_label'].map(label_mapping)
anns['class_label'] = anns['class_label'].astype(int)

# Load the predictions
det = bb.io.load('det_coco', f'/Users/inaki-eab/Desktop/small-fast-detector/evaluation_tools/outputs/{dataset}/predictions.json')
#det['image'] = det['image'].astype(int)
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
    if p == 0 and r == 0:
        f1 = 0
    else:
        f1 = 2 * (p * r) / (p + r)

    # Compute the metrics
    ap = evaluator.AP.mean()

    per_img_metrics['img'].append(img)
    per_img_metrics['mAP'].append(ap)
    per_img_metrics['P'].append(p)
    per_img_metrics['R'].append(r)
    per_img_metrics['F1'].append(f1)

#SAVE THE METRICS
per_img_metrics_df = pd.DataFrame(per_img_metrics)
per_img_metrics_df['F1'] = per_img_metrics_df['F1'].fillna(0)
per_img_metrics_df.to_csv(f'/Users/inaki-eab/Desktop/small-fast-detector/evaluation_tools/outputs/{dataset}/per_img_metrics.csv', index=False)
"""
per_img_metrics_df = pd.read_csv(f'/Users/inaki-eab/Desktop/small-fast-detector/evaluation_tools/outputs/{dataset}/per_img_metrics.csv')
metric = 'mAP'
df = per_img_metrics_df.sort_values(by=f'{metric}', ascending=False)
top = df.head(50)

# Create the new dataset structure for metric dataset
os.makedirs(f'/Users/inaki-eab/Desktop/IR/{dataset}_50{metric}/labels/val', exist_ok=True)
os.makedirs(f'/Users/inaki-eab/Desktop/IR/{dataset}_50{metric}/images/val', exist_ok=True)
os.makedirs(f'/Users/inaki-eab/Desktop/IR/{dataset}_50{metric}/annotations', exist_ok=True)

for img_name in top['img'].unique():
    shutil.copy(f'/Users/inaki-eab/Desktop/IR/{dataset}/labels/val/{img_name}.txt',
                f'/Users/inaki-eab/Desktop/IR/{dataset}_50{metric}/labels/val/{img_name}.txt')

    shutil.copy(f'/Users/inaki-eab/Desktop/IR/{dataset}/images/val/{img_name}.jpg',
                f'/Users/inaki-eab/Desktop/IR/{dataset}_50{metric}/images/val/{img_name}.jpg')