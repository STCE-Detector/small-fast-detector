from ultralytics import YOLO
# For more info visit: https://docs.ultralytics.com/modes/predict/

# Load model
model = YOLO('./models/yolov8n.onnx', task='detect')

# Source
source = './images/val2017'

# Output directory
root = './images/ouputs'
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