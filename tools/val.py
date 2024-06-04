from ultralytics.models import YOLO

model = YOLO("yolov8l_1024_ep300.pt")
metric = model.val(data="/home/edge/work/ultralytics/ultralytics/cfg/datasets/PANDA.yaml", imgsz=1024, max_det=1000)
