#!/usr/bin/env python3
import os
from pathlib import Path
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


def process_with_yolo(input_image, model):
    
    results = model(input_image, verbose=False)[0]  # [0]: first image processed by YOLO

    return results


class Yolosegmentation:
    def __init__(self):
        rospy.init_node('yolo_segmentation_node', anonymous=True)
        rospy.loginfo("Yolo Segmentation Node Initialized")

        # 初始化模型
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.current_dir, "../../YOLO_models/yolo12m.pt")
        self.model_path = os.path.abspath(self.model_path)
        self.model_name = Path(self.model_path).stem
        self.model = YOLO(self.model_path).to("cuda")
        self.bridge = CvBridge()
        rospy.loginfo(f"Using model: {self.model_name}")

        # 发布器（暂时未使用）
        self.stop_pub = rospy.Publisher("/test_topic", String, queue_size=1)
        self.grasp_pub = rospy.Publisher("/start_grasp", String, queue_size=1)

        # 订阅摄像头图像
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)


    def image_callback(self, msg):
        try:
            # rospy.loginfo(f"Image encoding: {msg.encoding}")
            input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            segmented_image = process_with_yolo(input_image, self.model)
            # cv2.imwrite("/tmp/input_debug.jpg", input_image)  # 保存一帧图片调试
            # cv2.imshow(f"{self.model_name}", segmented_image)
            # cv2.waitKey(1)


        except Exception as e:
            rospy.logerr(f"error: {str(e)}")



if __name__ == '__main__':
    Yolosegmentation()
    rospy.spin()



