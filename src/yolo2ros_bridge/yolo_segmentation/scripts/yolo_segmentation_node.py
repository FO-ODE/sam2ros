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

        # 初始化OpenCV窗口
        cv2.namedWindow("yolo segmentation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("yolo segmentation", 1280, 720)

    def image_callback(self, msg):
        try:
            # 图像转换
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 推理检测
            results = self.model(cv_image, verbose=False)

            # 可视化处理
            if results[0].boxes is not None:
                annotated_frame = results[0].plot()  # 绘制检测框
            else:
                annotated_frame = cv_image.copy()

            # 显示处理后的图像
            cv2.imshow("yolo segmentation", annotated_frame)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Processing error: {str(e)}")
        finally:
            if rospy.is_shutdown():
                cv2.destroyAllWindows()

if __name__ == '__main__':
    Yolosegmentation()
    rospy.spin()
