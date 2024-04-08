from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device="cpu")