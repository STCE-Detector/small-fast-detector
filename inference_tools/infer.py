from ultralytics import YOLO
# For more info visit: https://docs.ultralytics.com/modes/predict/

# Load model
model = YOLO('/Users/inaki-eab/Desktop/small-fast-detector/inference_tools/models/yolov8n.onnx', task='detect')

# Source
source = '/Users/inaki-eab/Desktop/datasets/coco128/images/train2017'

# Output directory
root = '/Users/inaki-eab/Desktop/small-fast-detector/inference_tools/ouputs'
experiment_name = 'exp_1'

# Inference
results = model(
    source,
    conf=0.25,          # confidence threshold
    iou=0.7,            # NMS IoU threshold
    imgsz=640,
    half=False,         # use FP16 half-precision inference
    device='cpu',
    save=True,          # Images
    save_txt=True,      # Text files
    save_conf=True,     # Save confidences
    # save results to project/name relative to script directory or absolute path
    project=root,
    name=experiment_name,
)

# Results saved to JSON if needed
"""
for i, result in enumerate(results):
    if i == 0:
        json_result = result.tojson()

json_result
"""