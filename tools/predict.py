from ultralytics.models import YOLO

model_s = YOLO("yolov8n_640_ep300.pt")
results = model_s(source="/home/edge/work/datasets/PANDA_dataset/images/val", imgsz=640, conf=0.001, iou=0.7, save_txt=True, save_conf=True, max_det=1000)
                                                                                                       