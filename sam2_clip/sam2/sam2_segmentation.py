# -*- coding: utf-8 -*-
# use sam2-env

# must run in the same terminal before running the script (done in the activate script of sam2-env)
# export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7" 

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import SAM 
from pathlib import Path

# # 系统ROS Python路径
# import sys
# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages') 


def process_with_sam2(input_image):
    model_path = "SAM_models/sam2.1_b.pt"   # model_path = "SAM_models/mobile_sam.pt"
    model = SAM(model_path)
    model.to('cuda:0')
    model.info()

    model_name = Path(model_path).stem
    print(f"Using model: {model_name}")

    segmented_image = model(input_image)
    segmented_image = segmented_image[0].plot()
    
    return segmented_image

class Sam2SegmentationNode:
    def __init__(self):
        rospy.init_node("sam2_segmentation_node", anonymous=True)

        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/xtion/rgb/image_processed", Image, queue_size=1) # queue_size=10
        self.bridge = CvBridge()
        
        rospy.loginfo("SAM2 segmentation node has started!")
        rospy.spin()

    def image_callback(self, msg):
        try:
            # set ROS Image to OpenCV format
            input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            segmented_image = process_with_sam2(input_image)

            # resize
            h1, w1 = input_image.shape[:2]
            h2, w2 = segmented_image.shape[:2]

            if (h1, w1) != (h2, w2):
                segmented_image = cv2.resize(segmented_image, (w1, h1))
                
            combined_image = np.hstack((input_image, segmented_image))
            
            # cv2.imshow("Original Image", input_image)
            # cv2.imshow("Segmented Image", segmented_image)
            cv2.imshow("Original | Segmented", combined_image)
            cv2.waitKey(1)
            
            ros_image = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
            self.image_pub.publish(ros_image)


        except Exception as e:
            rospy.logerr(f"error: {e}")

if __name__ == "__main__":
    Sam2SegmentationNode()
