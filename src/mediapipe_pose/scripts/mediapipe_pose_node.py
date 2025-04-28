#!/usr/bin/env python3
import rospy
import cv2
import mediapipe as mp
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class MediaPipePosePublisher:
    def __init__(self):
        rospy.init_node('mediapipe_pose_publisher', anonymous=True)
        
        self.bridge = CvBridge()

        # 订阅原始RGB图像
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        
        # 发布带骨架的图像
        self.image_pub = rospy.Publisher("/mediapipe_pose/image", Image, queue_size=10)

        # 初始化 MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def image_callback(self, data):
        try:
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        # 处理图像
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            # 在图上画骨架
            self.mp_drawing.draw_landmarks(
                cv_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        # OpenCV -> ROS Image
        try:
            pose_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            pose_image_msg.header = data.header  # 保持时间戳同步
            self.image_pub.publish(pose_image_msg)
        except Exception as e:
            rospy.logerr("CV Bridge publish error: %s", e)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = MediaPipePosePublisher()
    node.run()
