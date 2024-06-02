from ultralytics.models import YOLO

model_s = YOLO("yolov8s_768_ep300.pt")
results = model_s(source="/home/edge/work/datasets/PANDA_dataset/images/val", imgsz=768)
