# -*- coding: utf-8 -*-
# use global env
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import SAM 

# 系统ROS Python路径
import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages') 


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

        rospy.loginfo("SAM2 segmentation node has started!")
        rospy.spin()

    def image_callback(self, msg):
        try:
            # set ROS Image to OpenCV format
            input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            segmented_image = process_with_sam2(input_image)

            cv2.imshow("Original Image", input_image)
            cv2.imshow("Segmented Image", segmented_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"error: {e}")

if __name__ == "__main__":
    Sam2SegmentationNode()
