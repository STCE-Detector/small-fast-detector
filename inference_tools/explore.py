import onnx

model_path = "./models/yolov8n.onnx"
model = onnx.load(model_path)
onnx.checker.check_model(model)
print(model.ir_version)