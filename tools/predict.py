from ultralytics.models import YOLO

model_s = YOLO("yolov8s_768_ep300.pt")
results = model_s(source="/home/edge/work/datasets/PANDA_dataset/images/sliced_val", imgsz=768, conf=0.001, iou=0.7, save_txt=True, save_conf=True, max_det=1000)
