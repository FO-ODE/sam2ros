#!/usr/bin/env python3
import rospy
import cv2
import mediapipe as mp
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler

class MediaPipe3DSkeletonNode:
    def __init__(self):
        rospy.init_node('mediapipe_3d_skeleton_node', anonymous=True)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 订阅器
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/xtion/depth_registered/image", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/xtion/rgb/camera_info", CameraInfo, self.camera_info_callback)
        self.pointcloud_sub = rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2, self.pointcloud_callback)
        
        # 发布器
        self.image_pub = rospy.Publisher("/mediapipe_pose/image", Image, queue_size=10)
        self.posture_pub = rospy.Publisher("/posture_status", String, queue_size=10)
        self.pose_gesture_pub = rospy.Publisher("/posture_gesture_status", String, queue_size=10)
        self.skeleton_marker_pub = rospy.Publisher("/skeleton_markers", MarkerArray, queue_size=10)
        self.skeleton_points_pub = rospy.Publisher("/skeleton_points", PointCloud2, queue_size=10)
        self.segmented_person_pub = rospy.Publisher("/person_pointcloud", PointCloud2, queue_size=10)
        self.segmentation_mask_pub = rospy.Publisher("/person_segmentation_mask", Image, queue_size=10)

        # MediaPipe初始化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=True,  # 启用分割
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)

        # 相机参数
        self.camera_matrix = None
        self.depth_image = None
        self.latest_pointcloud = None
        self.pointcloud_data = None
        
        # 骨架连接定义（MediaPipe pose landmarks）
        self.skeleton_connections = [
            # 躯干
            (11, 12), (11, 23), (12, 24), (23, 24),
            # 左臂
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            # 右臂
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            # 左腿
            (23, 25), (25, 27), (27, 29), (27, 31),
            # 右腿
            (24, 26), (26, 28), (28, 30), (28, 32),
            # 头部
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10)
        ]

    def camera_info_callback(self, msg):
        """获取相机内参"""
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)

    def depth_callback(self, msg):
        """获取深度图像"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr("Depth image conversion error: %s", e)

    def pointcloud_callback(self, msg):
        """获取点云数据并转换为numpy数组"""
        self.latest_pointcloud = msg
        try:
            # 将点云转换为numpy数组以便处理
            points_list = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])
            
            if points_list:
                self.pointcloud_data = np.array(points_list)
            else:
                self.pointcloud_data = None
                
        except Exception as e:
            rospy.logwarn("Point cloud processing error: %s", e)
            self.pointcloud_data = None

    def pixel_to_3d(self, u, v, depth):
        """将像素坐标转换为3D坐标"""
        if self.camera_matrix is None or depth <= 0:
            return None
        
        # 相机内参
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 转换为3D坐标（相机坐标系）
        z = depth / 1000.0  # 假设深度单位是毫米，转换为米
        x = (u - cx) * z / fx
        y = (v - cy) * z / cy
        
        return np.array([x, y, z])

    def extract_person_pointcloud(self, segmentation_mask, pointcloud_msg):
        """使用分割掩码提取人体点云"""
        if pointcloud_msg is None or segmentation_mask is None:
            return None
            
        try:
            # 获取图像尺寸
            h, w = segmentation_mask.shape
            
            # 对分割掩码进行形态学腐蚀，缩小人体区域以去除边缘噪声
            kernel_size = rospy.get_param('~mask_erosion_kernel_size', 5)  # 可调参数
            iterations = rospy.get_param('~mask_erosion_iterations', 2)    # 可调参数
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            eroded_mask = cv2.erode(segmentation_mask, kernel, iterations=iterations)
            
            # 可选：进行轻微膨胀以恢复一些细节（开运算）
            use_opening = rospy.get_param('~use_mask_opening', True)
            if use_opening:
                opening_iterations = rospy.get_param('~mask_opening_iterations', 1)
                eroded_mask = cv2.dilate(eroded_mask, kernel, iterations=opening_iterations)
            
            # 发布处理后的掩码用于调试
            try:
                processed_mask_msg = self.bridge.cv2_to_imgmsg(eroded_mask, "mono8")
                processed_mask_msg.header.stamp = rospy.Time.now()
                processed_mask_msg.header.frame_id = "xtion_rgb_optical_frame"
                
                # 创建新的发布器用于处理后的掩码
                if not hasattr(self, 'processed_mask_pub'):
                    self.processed_mask_pub = rospy.Publisher("/person_segmentation_mask_processed", 
                                                            Image, queue_size=10)
                self.processed_mask_pub.publish(processed_mask_msg)
            except Exception as e:
                rospy.logwarn("Processed mask publish error: %s", e)
            
            # 将点云投影到图像平面
            person_points = []
            
            # 读取点云数据
            for i, point_data in enumerate(pc2.read_points(pointcloud_msg, 
                                                         field_names=("x", "y", "z"), 
                                                         skip_nans=True)):
                x, y, z = point_data
                
                if z <= 0:  # 跳过无效深度
                    continue
                    
                # 将3D点投影到图像平面
                if self.camera_matrix is not None:
                    fx = self.camera_matrix[0, 0]
                    fy = self.camera_matrix[1, 1]
                    cx = self.camera_matrix[0, 2]
                    cy = self.camera_matrix[1, 2]
                    
                    u = int(fx * x / z + cx)
                    v = int(fy * y / z + cy)
                    
                    # 检查是否在图像范围内且在处理后的人体分割区域内
                    if 0 <= u < w and 0 <= v < h:
                        if eroded_mask[v, u] > 128:  # 使用腐蚀后的掩码
                            person_points.append([x, y, z])
            
            if person_points:
                rospy.loginfo(f"Extracted {len(person_points)} person points after mask erosion")
                return np.array(person_points)
            else:
                return None
                
        except Exception as e:
            rospy.logwarn("Person point cloud extraction error: %s", e)
            return None

    def estimate_skeleton_from_person_cloud(self, person_points, landmarks_2d, image_shape):
        """从人体点云中估计3D骨架位置"""
        if person_points is None or len(person_points) == 0:
            return [None] * 33
            
        landmarks_3d = [None] * 33
        h, w = image_shape[:2]
        
        try:
            # 计算人体点云的边界框和质心
            min_coords = np.min(person_points, axis=0)
            max_coords = np.max(person_points, axis=0)
            centroid = np.mean(person_points, axis=0)
            
            # 人体尺寸估计
            person_height = max_coords[1] - min_coords[1]  # Y轴是高度
            person_width = max_coords[0] - min_coords[0]   # X轴是宽度
            person_depth = max_coords[2] - min_coords[2]   # Z轴是深度
            
            rospy.loginfo(f"Person cloud stats - Height: {person_height:.2f}m, "
                         f"Width: {person_width:.2f}m, Depth: {person_depth:.2f}m")
            rospy.loginfo(f"Point cloud Y range: {min_coords[1]:.2f} to {max_coords[1]:.2f}")
            
            for i, landmark in enumerate(landmarks_2d):
                if landmark.visibility > 0.5:
                    # 2D归一化坐标 (MediaPipe格式: x,y ∈ [0,1])
                    x_norm = landmark.x  # 0=左边, 1=右边
                    y_norm = landmark.y  # 0=顶部, 1=底部 (图像坐标系)
                    z_norm = landmark.z if hasattr(landmark, 'z') else 0
                    
                    # 坐标系转换和镜像修复：
                    # MediaPipe: (0,0)在左上角，x向右，y向下
                    # 需要修复：上下镜像 + 左右镜像
                    
                    # X坐标映射（左右方向）- 修复左右镜像
                    # MediaPipe的x=0是左边，x=1是右边
                    # 不翻转X坐标，保持正常映射
                    x_3d = min_coords[0] + x_norm * person_width
                    
                    # Y坐标映射（上下方向）- 修复上下镜像  
                    # MediaPipe的y=0是顶部，y=1是底部
                    # 直接映射：y=0对应最高点，y=1对应最低点
                    y_3d = min_coords[1] + y_norm * person_height
                    
                    # Z坐标使用质心深度加上相对偏移
                    z_offset = z_norm * person_depth * 0.1
                    z_3d = centroid[2] + z_offset
                    
                    landmarks_3d[i] = np.array([x_3d, y_3d, z_3d])
                    
                    # 调试信息：打印关键点
                    if i in [0, 11, 12, 23, 24]:  # 鼻子、肩膀、髋部
                        landmark_names = {0: "nose", 11: "left_shoulder", 12: "right_shoulder", 
                                        23: "left_hip", 24: "right_hip"}
                        rospy.loginfo(f"{landmark_names.get(i, i)}: "
                                    f"2D({x_norm:.2f},{y_norm:.2f}) -> "
                                    f"3D({x_3d:.2f},{y_3d:.2f},{z_3d:.2f})")
                        
                        # 显示点云边界用于调试
                        if i == 0:  # 只在鼻子那里打印一次
                            rospy.loginfo(f"Point cloud bounds: X[{min_coords[0]:.2f}, {max_coords[0]:.2f}], "
                                        f"Y[{min_coords[1]:.2f}, {max_coords[1]:.2f}], "
                                        f"Z[{min_coords[2]:.2f}, {max_coords[2]:.2f}]")
                            rospy.loginfo(f"Person dimensions: W={person_width:.2f}m, H={person_height:.2f}m, D={person_depth:.2f}m")
                    
        except Exception as e:
            rospy.logwarn("Skeleton estimation from person cloud failed: %s", e)
            
        return landmarks_3d

    def create_skeleton_markers(self, landmarks_3d, header):
        """创建骨架的可视化标记"""
        marker_array = MarkerArray()
        
        # 清除之前的标记
        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # 关节点标记
        for i, point in enumerate(landmarks_3d):
            if point is not None:
                marker = Marker()
                marker.header = header
                marker.ns = "skeleton_joints"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = point[2]
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                
                marker_array.markers.append(marker)
        
        # 骨架连接线
        line_marker = Marker()
        line_marker.header = header
        line_marker.ns = "skeleton_lines"
        line_marker.id = 1000
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        
        line_marker.scale.x = 0.015  # 线宽
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        
        for connection in self.skeleton_connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d) and
                landmarks_3d[start_idx] is not None and landmarks_3d[end_idx] is not None):
                
                # 起点
                start_point = Point()
                start_point.x = landmarks_3d[start_idx][0]
                start_point.y = landmarks_3d[start_idx][1]
                start_point.z = landmarks_3d[start_idx][2]
                line_marker.points.append(start_point)
                
                # 终点
                end_point = Point()
                end_point.x = landmarks_3d[end_idx][0]
                end_point.y = landmarks_3d[end_idx][1]
                end_point.z = landmarks_3d[end_idx][2]
                line_marker.points.append(end_point)
        
        marker_array.markers.append(line_marker)
        return marker_array

    def create_skeleton_pointcloud(self, landmarks_3d, header):
        """将骨架关节点转换为点云格式"""
        points = []
        for point in landmarks_3d:
            if point is not None:
                points.append([point[0], point[1], point[2]])
        
        if not points:
            return None
            
        # 创建点云消息
        skeleton_cloud = pc2.create_cloud_xyz32(header, points)
        return skeleton_cloud

    def angle_between_3points(self, a, b, c):
        """计算三点之间的角度"""
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def judge_posture(self, landmarks):
        """判断姿态"""
        try:
            key_indices = [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ]

            for idx in key_indices:
                lm = landmarks[idx.value]
                if lm.visibility < 0.5:
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

            if left_angle < 145 or right_angle < 145:
                return "sitting"
            else:
                return "standing / walking"

        except Exception as e:
            rospy.logwarn("Posture judgment failed: %s", e)
            return "unknown"

    def judge_gesture(self, hand_landmarks):
        """判断手势"""
        palm = hand_landmarks.landmark[0]
        
        index_joints = {
            "mcp": 5, "pip": 6, "dip": 7, "tip": 8
        }
        
        tip_ids = {
            "middle": 12, "ring": 16, "pinky": 20
        }
        
        def dist(a, b):
            return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        
        def calculate_angle(p1, p2, p3):
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        def is_index_straight():
            mcp_joint = hand_landmarks.landmark[index_joints["mcp"]]
            pip_joint = hand_landmarks.landmark[index_joints["pip"]]
            dip_joint = hand_landmarks.landmark[index_joints["dip"]]
            tip_joint = hand_landmarks.landmark[index_joints["tip"]]
            
            pip_angle = calculate_angle(mcp_joint, pip_joint, dip_joint)
            dip_angle = calculate_angle(pip_joint, dip_joint, tip_joint)
            
            straight_threshold = 160
            return pip_angle > straight_threshold and dip_angle > straight_threshold
        
        def are_other_fingers_bent():
            bent_threshold = 0.08
            middle_bent = dist(hand_landmarks.landmark[tip_ids["middle"]], palm) < bent_threshold
            ring_bent = dist(hand_landmarks.landmark[tip_ids["ring"]], palm) < bent_threshold  
            pinky_bent = dist(hand_landmarks.landmark[tip_ids["pinky"]], palm) < bent_threshold
            return sum([middle_bent, ring_bent, pinky_bent]) >= 2
        
        if is_index_straight() and are_other_fingers_bent():
            return "Pointing"
        else:
            return "Open hand"

    def image_callback(self, data):
        """主要的图像处理回调"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        posture = "unknown"
        landmarks_3d = [None] * 33

        if results.pose_landmarks:
            # 绘制2D骨架
            self.mp_drawing.draw_landmarks(
                cv_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # 判断姿态
            posture = self.judge_posture(results.pose_landmarks.landmark)
            
            # 处理分割掩码
            if results.segmentation_mask is not None:
                # 将分割掩码转换为二值图像
                segmentation_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                
                # 发布分割掩码用于可视化
                try:
                    mask_msg = self.bridge.cv2_to_imgmsg(segmentation_mask, "mono8")
                    mask_msg.header = data.header
                    self.segmentation_mask_pub.publish(mask_msg)
                except Exception as e:
                    rospy.logwarn("Segmentation mask publish error: %s", e)
                
                # 从点云中提取人体部分
                if self.latest_pointcloud is not None:
                    person_points = self.extract_person_pointcloud(segmentation_mask, self.latest_pointcloud)
                    
                    if person_points is not None and len(person_points) > 100:  # 至少100个点
                        # 发布分割后的人体点云
                        header = Header()
                        header.stamp = rospy.Time.now()
                        header.frame_id = data.header.frame_id
                        
                        person_cloud = pc2.create_cloud_xyz32(header, person_points)
                        self.segmented_person_pub.publish(person_cloud)
                        
                        # 从人体点云估计3D骨架
                        landmarks_3d = self.estimate_skeleton_from_person_cloud(
                            person_points, results.pose_landmarks.landmark, cv_image.shape
                        )
                        
                        # 发布3D骨架标记
                        skeleton_markers = self.create_skeleton_markers(landmarks_3d, header)
                        self.skeleton_marker_pub.publish(skeleton_markers)
                        
                        # 发布骨架点云
                        skeleton_cloud = self.create_skeleton_pointcloud(landmarks_3d, header)
                        if skeleton_cloud:
                            self.skeleton_points_pub.publish(skeleton_cloud)
                        
                        rospy.loginfo(f"Person cloud size: {len(person_points)} points")
                    else:
                        rospy.logwarn("Person point cloud too small or not found")

        # 手势检测
        hand_results = self.hands.process(rgb_image)
        gesture = "no hands"
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                gesture = self.judge_gesture(hand_landmarks)
                break

        # 在图像上显示信息
        cv2.putText(cv_image, f"Posture: {posture}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(cv_image, f"Gesture: {gesture}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

        # 发布消息
        self.posture_pub.publish(posture)
        self.pose_gesture_pub.publish(f"{posture}, {gesture}")
        rospy.loginfo(f"Posture: {posture}, Gesture: {gesture}")

        # 发布处理后的图像
        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            img_msg.header = data.header
            self.image_pub.publish(img_msg)
        except Exception as e:
            rospy.logerr("Image publish error: %s", e)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = MediaPipe3DSkeletonNode()
    node.run()