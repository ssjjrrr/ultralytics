from pathlib import Path

from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.yolov8 import download_yolov8s_model

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="/home/edge/work/ultralytics/yolov8n_640_ep300.pt",
    confidence_threshold=0.3,
    device="cuda",  # or 'cuda:0'
)

result = get_sliced_prediction(
    "/home/edge/work/datasets/PANDA_dataset/images/train/IMG_01_01.jpg",
    detection_model,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

print(result)
