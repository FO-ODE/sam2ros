#!/usr/bin/env python3
import rospy
import cv2
import mediapipe as mp
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class MediaPipePosePostureNode:
    def __init__(self):
        rospy.init_node('mediapipe_pose_posture_node', anonymous=True)

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/mediapipe_pose/image", Image, queue_size=10)
        self.posture_pub = rospy.Publisher("/posture_status", String, queue_size=10)
        self.pose_gesture_pub = rospy.Publisher("/posture_gesture_status", String, queue_size=10)


        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)

    def angle_between_3points(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def judge_posture(self, landmarks):
        try:
            key_indices = [
                self.mp_pose.PoseLandmark.LEFT_HIP, # 左髋 23
                self.mp_pose.PoseLandmark.RIGHT_HIP, # 右髋 24
                
                self.mp_pose.PoseLandmark.LEFT_KNEE, # 左膝 25
                self.mp_pose.PoseLandmark.RIGHT_KNEE, # 右膝 26
                
                self.mp_pose.PoseLandmark.LEFT_ANKLE, # 左踝 27               
                self.mp_pose.PoseLandmark.RIGHT_ANKLE # 右踝 28
            ]

            # visibility too low ==> unknown
            for idx in key_indices:
                lm = landmarks[idx.value]
                if lm.visibility < 0.5:
                    rospy.logwarn(f" {idx.name} visibility too low: {lm.visibility:.2f}")
                    return "unknown"


            left_angle = self.angle_between_3points(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            )
            right_angle = self.angle_between_3points(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            )

            rospy.loginfo(f"Left angle: {left_angle:.1f}, Right angle: {right_angle:.1f}")

            if left_angle < 145 or right_angle < 145:
                return "sitting"
            else:
                return "standing / walking"

        except Exception as e:
            rospy.logwarn("Posture judgment failed: %s", e)
            return "unknown"


    # def judge_gesture(self, hand_landmarks):
    #     palm = hand_landmarks.landmark[0]
    #     tip_ids = {
    #         "index": 8,
    #         "middle": 12,
    #         "ring": 16,
    #         "pinky": 20
    #     }

    #     def dist(a, b):
    #         return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    #     index_tip_dist = dist(hand_landmarks.landmark[tip_ids["index"]], palm)
    #     middle_tip_dist = dist(hand_landmarks.landmark[tip_ids["middle"]], palm)
    #     ring_tip_dist = dist(hand_landmarks.landmark[tip_ids["ring"]], palm)
    #     pinky_tip_dist = dist(hand_landmarks.landmark[tip_ids["pinky"]], palm)

    #     # 经验阈值，可能需要根据图像大小微调
    #     if (index_tip_dist > 0.05 and  # 食指伸出
    #         middle_tip_dist < 0.10 and
    #         ring_tip_dist < 0.10 and
    #         pinky_tip_dist < 0.10):
    #         return "Pointing"

    #     elif all(dist(hand_landmarks.landmark[tip_ids[finger]], palm) < 0.05  # 所有手指都靠近掌心
    #             for finger in tip_ids):
    #         return "Fist"

    #     else:
    #         return "Open hand"
    

    def judge_gesture(self, hand_landmarks):
        palm = hand_landmarks.landmark[0]
        
        # 食指的关键点索引
        index_joints = {
            "mcp": 5,   # 掌指关节 (metacarpophalangeal)
            "pip": 6,   # 近指间关节 (proximal interphalangeal) 
            "dip": 7,   # 远指间关节 (distal interphalangeal)
            "tip": 8    # 指尖
        }
        
        # 其他手指指尖
        tip_ids = {
            "middle": 12,
            "ring": 16,
            "pinky": 20
        }
        
        def dist(a, b):
            return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        
        def calculate_angle(p1, p2, p3):
            """计算三点之间的夹角，p2为顶点"""
            # 创建向量
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            # 计算角度
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # 防止数值误差导致的domain error
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)  # 转换为度数
        
        # 检查食指是否伸直
        def is_index_straight():
            # 计算食指各关节的角度
            mcp_joint = hand_landmarks.landmark[index_joints["mcp"]]
            pip_joint = hand_landmarks.landmark[index_joints["pip"]]
            dip_joint = hand_landmarks.landmark[index_joints["dip"]]
            tip_joint = hand_landmarks.landmark[index_joints["tip"]]
            
            # 计算关节角度
            pip_angle = calculate_angle(mcp_joint, pip_joint, dip_joint)
            dip_angle = calculate_angle(pip_joint, dip_joint, tip_joint)
            
            # 如果角度接近180度（伸直），则认为是pointing
            # 阈值可以根据需要调整
            straight_threshold = 160  # 度数
            
            return pip_angle > straight_threshold and dip_angle > straight_threshold
        
        # 检查其他手指是否弯曲
        def are_other_fingers_bent():
            bent_threshold = 0.08  # 距离阈值，可根据需要调整
            
            middle_bent = dist(hand_landmarks.landmark[tip_ids["middle"]], palm) < bent_threshold
            ring_bent = dist(hand_landmarks.landmark[tip_ids["ring"]], palm) < bent_threshold  
            pinky_bent = dist(hand_landmarks.landmark[tip_ids["pinky"]], palm) < bent_threshold
            
            # 至少两个手指弯曲就认为是pointing姿势
            return sum([middle_bent, ring_bent, pinky_bent]) >= 2
        
        # 判断手势
        if is_index_straight() and are_other_fingers_bent():
            return "Pointing"
        # elif all(dist(hand_landmarks.landmark[tip_ids[finger]], palm) < 0.07
        #         for finger in tip_ids):
        #     return "Fist"
        else:
            return "Open hand"


    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        posture = "unknown"

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                cv_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            posture = self.judge_posture(results.pose_landmarks.landmark)
            
            
        # gesture detection
        hand_results = self.hands.process(rgb_image)
        gesture = "no hands"
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    cv_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                gesture = self.judge_gesture(hand_landmarks)
                break 


        cv2.putText(cv_image, f"Posture: {posture}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        
        cv2.putText(cv_image, f"Gesture: {gesture}", (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

        
        self.posture_pub.publish(posture)
        self.pose_gesture_pub.publish(f"{posture}, {gesture}")
        rospy.loginfo(f"Posture: {posture}, Gesture: {gesture}")


        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            img_msg.header = data.header
            self.image_pub.publish(img_msg)
        except Exception as e:
            rospy.logerr("Image publish error: %s", e)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = MediaPipePosePostureNode()
    node.run()
