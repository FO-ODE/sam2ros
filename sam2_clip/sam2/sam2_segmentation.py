# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import SAM



def process_with_sam2(input_image):
    segmented_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    model = SAM("SAM_models/sam2.1_b.pt")
    segmented_image = model()  
    segmented_image = model(input_image)
    segmented_image = segmented_image[0].plot()
    
    return segmented_image

class Sam2SegmentationNode:
    def __init__(self):
        rospy.init_node("sam2_segmentation_node", anonymous=True)

        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)

        self.bridge = CvBridge()

        rospy.loginfo("SAM2 分割节点已启动")
        rospy.spin()

    def image_callback(self, msg):
        try:
            # 将 ROS Image 转换为 OpenCV 格式
            input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            segmented_image = process_with_sam2(input_image)

            cv2.imshow("Original Image", input_image)
            cv2.imshow("Segmented Image", segmented_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"错误: {e}")

if __name__ == "__main__":
    Sam2SegmentationNode()
