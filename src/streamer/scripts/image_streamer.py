#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

def main():
    topic = '/xtion/rgb/image_raw'
    pub_rate = 10  # Hz
    img_path = os.path.join(os.path.dirname(__file__), '../../test/test_images/goods.png')

    rospy.init_node('image_streamer_node', anonymous=True)
    pub = rospy.Publisher(topic, Image, queue_size=10)
    bridge = CvBridge()

    img = cv2.imread(img_path)
    
    if img is None:
        rospy.logerr("Image not found!")
    else:        
        rospy.loginfo("Image loaded successfully.")
        rospy.loginfo(f"Publishing image to {topic}")
        rospy.loginfo(f"Publishing at {pub_rate} Hz")

    rate = rospy.Rate(pub_rate)  # 10 Hz
    while not rospy.is_shutdown():
        msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    main()
