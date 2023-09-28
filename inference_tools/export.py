from ultralytics import YOLO
# For more info visit: # https://docs.ultralytics.com/modes/export/

# Load model
model = YOLO('./models/yolov8n.pt')

# Export to ONNX, the model is saved next to original with .onnx extension
model.export(
    format='onnx',  # destination format
    imgsz=640,  # inference size (pixels)
    half=False,  # use FP16 half-precision inference
    dynamic=False,  # ONNX export only supports static size
    simplify=False,  # ONNX export only supports static size
    opset=None,  # ONNX opset version
)
