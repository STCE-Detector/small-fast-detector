from ultralytics import YOLO


model_name = '8sp2_150'

# Initialize model and load matching weights
model = YOLO('./evaluation_tools/models/'+model_name+'.pt')

metrics = model.val(
    data='custom_dataset.yaml',
    imgsz=640,
    batch=8,
    device=[1],
    verbose=True,
    save=True,
    save_json=True,
    plots=True,
    # save results to project/name relative to script directory or absolute path
    project='validations',
    name=model_name,
)