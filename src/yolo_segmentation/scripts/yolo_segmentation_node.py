# -*- coding: utf-8 -*-
from ultralytics import YOLO

# # 系统ROS Python路径
# import sys
# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages') 


# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12n.pt")



# Run inference with the YOLO12n model on the 'bus.jpg' image
results = model("src/test/test_images/workers.jpg", save=True, save_txt=True, show=True)

