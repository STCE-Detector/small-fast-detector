from ultralytics import YOLO
config_export = {'format': 'engine', 'args': {'imgsz': 640, 'half': True, 'dynamic': False, 'int8': False, 'simplify': True}}


model = YOLO('./detectors/8sp2_150e_64b.pt', task='detect')
model.export(format=config_export['format'], device='cuda', **config_export['args'], project='./detectors/')
