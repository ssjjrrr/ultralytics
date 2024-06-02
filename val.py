from ultralytics.models import YOLO

model = YOLO("yolov8n_640_ep300.pt")
metric = model.val(data="/home/edge/work/ultralytics/ultralytics/cfg/datasets/PANDA.yaml", imgsz=640)
print(metric)