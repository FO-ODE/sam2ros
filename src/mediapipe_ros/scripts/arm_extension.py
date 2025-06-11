#!/usr/bin/env python3
import rospy
import cv2
import mediapipe as mp # type: ignore
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
from cv_bridge import CvBridge
import tf.transformations as tf_trans

class MediaPipePosePostureNode:
    def __init__(self):
        rospy.init_node('mediapipe_pose_posture_node', anonymous=True)

        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/xtion/depth_registered/image", Image, self.depth_callback)
        
        # Publishers
        self.image_pub = rospy.Publisher("/mediapipe_pose/image", Image, queue_size=10)
        self.posture_pub = rospy.Publisher("/posture_status", String, queue_size=10)
        self.skeleton_pub = rospy.Publisher("/skeleton_markers", MarkerArray, queue_size=10)
        self.arm_extension_pub = rospy.Publisher("/arm_extension_marker", Marker, queue_size=10)

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        # Camera parameters
        self.camera_matrix = np.array([
            [579.7253645876941, 0.0, 329.1726287752489],
            [0.0, 581.0305537009599, 252.8879914424264],
            [0.0, 0.0, 1.0]
        ])
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        # Frame parameters
        self.frame_id = "xtion_rgb_optical_frame"
        self.width = 640
        self.height = 480
        
        # Depth image storage
        self.depth_image = None
        
        # Arm extension detection parameters
        self.arm_straightness_threshold = 20.0  # degrees, maximum deviation from straight line
        self.cylinder_diameter = 0.10  # 20cm
        self.cylinder_length = 2.0     # 2m
        
        # Pose connections for visualization
        self.pose_connections = [
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            # Right arm  
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
            # Face (simplified)
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)
        ]

    def depth_callback(self, data):
        """接收深度图像"""
        try:
            # 深度图像通常是16位单通道，单位为毫米
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr("Depth image conversion error: %s", e)

    def pixel_to_3d(self, u, v, depth):
        """将像素坐标和深度转换为3D坐标"""
        if depth <= 0:
            return None
        
        # 深度值通常以毫米为单位，转换为米，并添加5cm的偏移
        z = depth / 1000.0 + 0.05  # 向后偏移5cm
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return Point(x=x, y=y, z=z)

    def get_depth_at_pixel(self, u, v):
        """获取指定像素位置的深度值"""
        if self.depth_image is None:
            return 0
        
        # 确保坐标在图像范围内
        u = int(np.clip(u, 0, self.width - 1))
        v = int(np.clip(v, 0, self.height - 1))
        
        # 获取深度值，可能需要处理无效深度
        depth = self.depth_image[v, u]
        
        # 过滤无效深度值
        if depth == 0 or np.isnan(depth) or np.isinf(depth):
            return 0
            
        return depth

    def calculate_vector(self, point1, point2):
        """计算两点之间的向量"""
        if point1 is None or point2 is None:
            return None
        return np.array([point2.x - point1.x, point2.y - point1.y, point2.z - point1.z])

    def calculate_angle_between_vectors(self, vec1, vec2):
        """计算两个向量之间的角度（度）"""
        if vec1 is None or vec2 is None:
            return None
        
        # 归一化向量
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
        
        # 计算点积
        dot_product = np.dot(vec1_norm, vec2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 计算角度
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def calculate_3d_angle_between_3points(self, point1, point2, point3):
        """计算三个3D点形成的角度（以point2为顶点）"""
        if point1 is None or point2 is None or point3 is None:
            return None
        
        # 计算向量
        vec1 = np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
        vec2 = np.array([point3.x - point2.x, point3.y - point2.y, point3.z - point2.z])
        
        # 计算向量长度
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)
        
        if len1 < 1e-6 or len2 < 1e-6:
            return None
        
        # 归一化向量
        vec1_norm = vec1 / len1
        vec2_norm = vec2 / len2
        
        # 计算点积
        dot_product = np.dot(vec1_norm, vec2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 计算角度（弧度转度）
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def is_right_arm_straight(self, landmarks_3d):
        """检测右手臂是否伸直"""
        # 右肩、右肘、右手腕的索引
        right_shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value  # 12
        right_elbow_idx = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value        # 14
        right_wrist_idx = self.mp_pose.PoseLandmark.RIGHT_WRIST.value        # 16
        
        # 检查关键点是否有效
        if (right_shoulder_idx >= len(landmarks_3d) or 
            right_elbow_idx >= len(landmarks_3d) or 
            right_wrist_idx >= len(landmarks_3d)):
            return False, None, None
            
        right_shoulder = landmarks_3d[right_shoulder_idx]
        right_elbow = landmarks_3d[right_elbow_idx]
        right_wrist = landmarks_3d[right_wrist_idx]
        
        if right_shoulder is None or right_elbow is None or right_wrist is None:
            return False, None, None
        
        # 计算肘关节的角度（肩-肘-腕三点形成的角度）
        elbow_angle = self.calculate_3d_angle_between_3points(right_shoulder, right_elbow, right_wrist)
        
        if elbow_angle is None:
            return False, None, None
        
        # 当肘关节角度接近180度时，手臂是直的
        deviation_from_straight = abs(180.0 - elbow_angle)
        is_straight = deviation_from_straight < self.arm_straightness_threshold
        
        # 计算手臂延伸方向（从肩到腕的方向）
        arm_direction = self.calculate_vector(right_shoulder, right_wrist)
        
        rospy.loginfo(f"Right elbow angle: {elbow_angle:.1f}°, deviation from straight: {deviation_from_straight:.1f}°, is straight: {is_straight}")
        
        return is_straight, right_wrist, arm_direction

    def create_cylinder_marker(self, start_point, direction_vector, header):
        """创建圆柱体marker"""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = self.frame_id
        marker.ns = "arm_extension"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # 归一化方向向量
        direction_norm = direction_vector / (np.linalg.norm(direction_vector) + 1e-6)
        
        # 计算圆柱体中心点（从手腕开始，沿着方向延伸一半长度）
        center_offset = direction_norm * (self.cylinder_length / 2.0)
        marker.pose.position.x = start_point.x + center_offset[0]
        marker.pose.position.y = start_point.y + center_offset[1]
        marker.pose.position.z = start_point.z + center_offset[2]
        
        # 计算圆柱体的朝向
        # 默认圆柱体沿Z轴，需要旋转到arm_direction方向
        z_axis = np.array([0, 0, 1])
        
        # 计算旋转轴和角度
        rotation_axis = np.cross(z_axis, direction_norm)
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0))
        
        if np.linalg.norm(rotation_axis) > 1e-6:
            # 归一化旋转轴
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # 创建四元数
            quat = tf_trans.quaternion_about_axis(rotation_angle, rotation_axis)
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
        else:
            # 方向向量与Z轴平行，不需要旋转
            marker.pose.orientation.w = 1.0
        
        # 设置圆柱体尺寸
        marker.scale.x = self.cylinder_diameter  # 直径
        marker.scale.y = self.cylinder_diameter  # 直径
        marker.scale.z = self.cylinder_length    # 长度
        
        # 设置颜色（蓝色，半透明）
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.6
        
        # 设置生存时间
        marker.lifetime = rospy.Duration(1.0)
        
        return marker

    def create_skeleton_markers(self, landmarks_3d, header):
        """创建骨架的MarkerArray"""
        marker_array = MarkerArray()
        
        # 清除之前的markers
        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # 创建关键点markers（红色球）
        for i, point_3d in enumerate(landmarks_3d):
            if point_3d is not None:
                marker = Marker()
                marker.header = header
                marker.ns = "skeleton_joints"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position = point_3d
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.05  # 5cm直径的球
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                
                marker.color.r = 1.0  # 红色
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                
                marker_array.markers.append(marker)
        
        # 创建连接线markers（白色线）
        line_id = 1000  # 从1000开始避免与关键点ID冲突
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d) and 
                landmarks_3d[start_idx] is not None and landmarks_3d[end_idx] is not None):
                
                marker = Marker()
                marker.header = header
                marker.ns = "skeleton_connections"
                marker.id = line_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.02  # 线宽2cm
                
                marker.color.r = 1.0  # 白色
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                
                # 添加两个点
                marker.points.append(landmarks_3d[start_idx])
                marker.points.append(landmarks_3d[end_idx])
                
                marker_array.markers.append(marker)
                line_id += 1
        
        return marker_array

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
                    # rospy.logwarn(f" {idx.name} visibility too low: {lm.visibility:.2f}")
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

            # rospy.loginfo(f"Left angle: {left_angle:.1f}, Right angle: {right_angle:.1f}")

            if left_angle < 145 or right_angle < 145:
                return "sitting"
            else:
                return "standing / walking"

        except Exception as e:
            # rospy.logwarn("Posture judgment failed: %s", e)
            return "unknown"

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        posture = "unknown"
        landmarks_3d = []
        arm_straight = False

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                cv_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            posture = self.judge_posture(results.pose_landmarks.landmark)
            
            # 转换2D关键点为3D关键点
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # 将归一化坐标转换为像素坐标
                u = int(landmark.x * self.width)
                v = int(landmark.y * self.height)
                
                # 获取深度值
                depth = self.get_depth_at_pixel(u, v)
                
                # 转换为3D坐标
                point_3d = self.pixel_to_3d(u, v, depth)
                landmarks_3d.append(point_3d)
            
            # 创建并发布骨架markers
            if self.depth_image is not None:
                skeleton_markers = self.create_skeleton_markers(landmarks_3d, data.header)
                skeleton_markers.markers[0].header.frame_id = self.frame_id  # 设置正确的frame_id
                for marker in skeleton_markers.markers[1:]:
                    marker.header.frame_id = self.frame_id
                self.skeleton_pub.publish(skeleton_markers)
                
                
                # 检测右手臂是否伸直
                is_straight, wrist_point, arm_direction = self.is_right_arm_straight(landmarks_3d)
                arm_straight = is_straight
                
                # 如果右手臂伸直，创建并发布圆柱体marker
                if is_straight and wrist_point is not None and arm_direction is not None:
                    cylinder_marker = self.create_cylinder_marker(wrist_point, arm_direction, data.header)
                    self.arm_extension_pub.publish(cylinder_marker)
                else:
                    # 如果手臂不直，发布删除marker
                    delete_marker = Marker()
                    delete_marker.header = data.header
                    delete_marker.header.frame_id = self.frame_id
                    delete_marker.ns = "arm_extension"
                    delete_marker.id = 0
                    delete_marker.action = Marker.DELETE
                    self.arm_extension_pub.publish(delete_marker)

        # 在图像上显示信息
        cv2.putText(cv_image, f"Posture: {posture}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # 显示手臂状态
        arm_status = "Right Arm: Straight" if arm_straight else "Right Arm: Bent"
        arm_color = (0, 255, 255) if arm_straight else (0, 0, 255)  # 黄色表示伸直，红色表示弯曲
        cv2.putText(cv_image, arm_status, (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, arm_color, 2, cv2.LINE_AA)

        # 显示深度信息状态
        depth_status = "Depth: Available" if self.depth_image is not None else "Depth: Not Available"
        cv2.putText(cv_image, depth_status, (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        self.posture_pub.publish(posture)
        # rospy.loginfo(f"Posture: {posture}, Right arm straight: {arm_straight}")

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