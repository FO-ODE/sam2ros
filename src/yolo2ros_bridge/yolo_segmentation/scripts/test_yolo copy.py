# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../../YOLO_models/yolo11l-pose.pt")  
model_path = os.path.abspath(model_path)
model = YOLO(model_path)
model.to('cuda:0')

file_path = os.path.join(current_dir, "../../../test/test_images/workers.jpg")
file_path = os.path.join(current_dir, "../../../test/test_videos/test.mp4")
file_path = os.path.abspath(file_path)


# Run inference with the YOLO12n model on the 'bus.jpg' image
results = model(file_path, save=True, save_txt=True, show=True)


