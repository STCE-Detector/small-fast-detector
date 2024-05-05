from ultralytics import YOLO

export_config = {'format': 'engine', 'args': {'imgsz': 640, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True}}

model = YOLO("./detectors/8sp2_150e_64b.pt")
model.export(format=export_config['format'], device="cuda", **export_config['args'], project='./detectors/')