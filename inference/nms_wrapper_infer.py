from pathlib import Path
import cv2  # OpenCV for reading images
from tqdm import tqdm
import time
import os
import csv

def nms_wrapper_infer(config):
    """Inference pipeline for Yolov8 model with NMS included as a node

    Args:
        config (json): Configuration dictionary
    """
    from tracker.jetson.model.model import Yolov8

    # Read all image paths
    image_paths = [str(p) for p in Path(config['input_data_dir']).rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    # Initialize Yolov8 model with appropriate class names
    class_names = {
        0: "person",
        1: "car",
        2: "truck",
        3: "uav",
        4: "airplane",
        5: "boat",
    }
    model = Yolov8(config, class_names)

    # Create output directory for annotated images, predicted labels and timing results
    output_dir = config['output_dir'] + "/" + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    output_labels = output_dir + "/labels/"
    os.makedirs(output_labels)

    timing_results = []
    for img_path in tqdm(image_paths, desc="Processing images with Yolov8"):
        img = cv2.imread(img_path)  # Read image with OpenCV
        results = model.predict(img)
        if results:
            timing_results.append(model.last_prediction_time)
            annotated_img = results[0].plot()  # Assuming `results` is a list of `Results` objects
            save_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, annotated_img)
            label_path = os.path.join(output_labels, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            with open(label_path, 'w') as f:
                for result in results:
                    for bbox in results[0].boxes:
                        class_id = int(bbox.cls.item())
                        x_center, y_center, width, height = bbox.xywh.flatten().tolist()
                        confidence = bbox.conf.item()
                        f.write(f"{class_id} {x_center} {y_center} {width} {height} {confidence}\n")
    # Save inference times to csv
    csv_output_path = os.path.join(output_dir, 'inference_time.csv')
    with open(csv_output_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image Path", "Pre-process (ms)", "Inference (ms)", "Post-process (ms)"])
        for img_path, timing in zip(image_paths, timing_results):
            image_name = img_path.split('/')[-1].split('.')[0]
            writer.writerow([image_name, *timing])
