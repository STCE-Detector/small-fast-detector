import numpy as np
import os

dataset_root = '/data-fast/128-data1/ierregue/datasets/custom_dataset_v3'

# Read folders in the labels directory
labels_dir = os.path.join(dataset_root, 'labels')
splits = [f for f in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, f))]

# Create new "tag_labels" directory
tag_labels_dir = os.path.join(dataset_root, 'tagged_labels')
os.makedirs(tag_labels_dir, exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(tag_labels_dir, split), exist_ok=True)

# Read all the labels in the labels directory
i = 1
for split in splits:
    split_dir = os.path.join(labels_dir, split)
    txt_files = [f for f in os.listdir(split_dir) if f.endswith('.txt')]
    for file in txt_files:
        # Load the data
        data = np.loadtxt(os.path.join(split_dir, file)).reshape(-1, 5)
        # Generate unique identifiers sequentially
        ids = np.arange(i, i + data.shape[0])
        # Add IDS column to the data
        data = np.hstack((data, ids[:, None]))
        # Save the data
        np.savetxt(os.path.join(tag_labels_dir, split, file), data, fmt='%d %1.10f %1.10f %1.10f %1.10f %d')
        i += data.shape[0]
