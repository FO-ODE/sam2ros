#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def main():
    rospy.init_node('static_image_publisher', anonymous=True)
    pub = rospy.Publisher('/xtion/image_raw', Image, queue_size=10)
    bridge = CvBridge()

    img = cv2.imread('/catkin_ws/src/test/test_images/goods.png')
    
    if img is None:
        rospy.logerr("Image not found!")
    else:        
        rospy.loginfo("Image loaded successfully.")

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    main()
