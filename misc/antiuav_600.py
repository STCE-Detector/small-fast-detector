import os
import json
import numpy as np
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import cv2
train_dir = '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/train'
val_dir = '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/validation'

train_labels_dir = '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/train-labels'
val_labels_dir = '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/val-labels'


"""def convert_json_to_txt(json_path, image_dir, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for image in data['images']:
        image_id = image['id']
        image_file = image['file_name']
        image_name = os.path.splitext(image_file)[0]
        label_file_name = f"{image_name}.txt"
        label_file_path = os.path.join(output_dir, label_file_name)

        # Leer la imagen para obtener sus dimensiones
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        label_arrays = []
        for annotation in annotations:
            bbox = np.array(annotation['bbox'])
            xt, yt, w, h = bbox
            image_height, image_width, _ = frame.shape
            xcn = (xt + w / 2) / image_width
            ycn = (yt + h / 2) / image_height
            wn = w / image_width
            hn = h / image_height
            label_arrays.append([3, xcn, ycn, wn, hn])

        os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
        np.savetxt(label_file_path, label_arrays, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])

# Convertir anotaciones de train y val a archivos .txt
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

convert_json_to_txt('/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/train.json', '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/train', train_labels_dir)
convert_json_to_txt('/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/validation.json', '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/validation', val_labels_dir)



def sample_and_rename_images_and_labels(source_image_dir, source_label_dir, target_image_dir, target_label_dir,
                                        sample_rate=75):
    image_id = 1
    subdirs = [os.path.join(source_image_dir, d) for d in os.listdir(source_image_dir) if
               os.path.isdir(os.path.join(source_image_dir, d))]

    for subdir in tqdm(subdirs, desc="Sampling images and labels from subdirectories"):
        _, _, files = next(os.walk(subdir))
        files = sorted([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
        sampled_files = files[::sample_rate]

        for file in sampled_files:
            src_image_path = os.path.join(subdir, file)
            new_image_name = f"{image_id:06d}.jpg"
            dst_image_path = os.path.join(target_image_dir, new_image_name)
            os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
            shutil.copy(src_image_path, dst_image_path)

            # Copiar y renombrar el archivo de anotaciones
            label_name = os.path.splitext(file)[0] + ".txt"
            src_label_path = os.path.join(source_label_dir, subdir.split('/')[-1], label_name)  # Adjusted path
            new_label_name = f"{image_id:06d}.txt"
            dst_label_path = os.path.join(target_label_dir, new_label_name)
            os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)

            image_id += 1


# Samplear y renombrar imágenes y archivos .txt
train_subset_dir = '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/train-subset'
val_subset_dir = '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/val-subset'
train_subset_labels_dir = os.path.join(train_subset_dir, 'labels')
val_subset_labels_dir = os.path.join(val_subset_dir, 'labels')
os.makedirs(train_subset_labels_dir, exist_ok=True)
os.makedirs(val_subset_labels_dir, exist_ok=True)

sample_and_rename_images_and_labels(train_dir, train_labels_dir, train_subset_dir, train_subset_labels_dir)
sample_and_rename_images_and_labels(val_dir, val_labels_dir, val_subset_dir, val_subset_labels_dir)

# Crear archivos zip de los subconjuntos
shutil.make_archive(train_subset_dir, 'zip', train_subset_dir)
shutil.make_archive(val_subset_dir, 'zip', val_subset_dir)"""

# Definir directorios
train_subset_dir = '/Users/johnny/Projects/datasets/3rd_Anti-UAV_train_val/train-subset'
train_subset_labels_dir = os.path.join(train_subset_dir, 'labels')


def load_annotations(label_file_path):
    if os.path.exists(label_file_path):
        annotations = np.loadtxt(label_file_path)
        if annotations.ndim == 1:
            annotations = annotations[np.newaxis, :]
        return annotations
    else:
        return np.array([])


def draw_bounding_boxes(image, annotations):
    h, w, _ = image.shape
    for annotation in annotations:
        class_id, xcn, ycn, wn, hn = annotation
        xt = int(xcn * w - wn * w / 2)
        yt = int(ycn * h - hn * h / 2)
        xb = int(xcn * w + wn * w / 2)
        yb = int(ycn * h + hn * h / 2)
        cv2.rectangle(image, (xt, yt), (xb, yb), (0, 255, 0), 2)
    return image


def visualize_train_subset(image_dir, label_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        image = cv2.imread(image_path)
        if image is not None:
            annotations = load_annotations(label_path)
            image_with_boxes = draw_bounding_boxes(image, annotations)

            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
            plt.title(image_file)
            plt.axis('off')
            plt.show()


# Visualizar las imágenes del conjunto de entrenamiento con las anotaciones
visualize_train_subset(train_subset_dir, train_subset_labels_dir)