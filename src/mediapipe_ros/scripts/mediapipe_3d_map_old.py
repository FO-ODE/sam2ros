#!/usr/bin/env python3
import rospy
import cv2
import mediapipe as mp
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

class MediaPipePosePostureNode:
    def __init__(self):
        rospy.init_node('mediapipe_pose_posture_node', anonymous=True)

        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.pointcloud_sub = rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2, self.pointcloud_callback)
        
        # Publishers
        self.image_pub = rospy.Publisher("/mediapipe_pose/image", Image, queue_size=10)
        self.posture_pub = rospy.Publisher("/posture_status", String, queue_size=10)
        self.pose_gesture_pub = rospy.Publisher("/posture_gesture_status", String, queue_size=10)
        self.joints_3d_pub = rospy.Publisher("/pose_joints_3d", String, queue_size=10)
        self.skeleton_marker_pub = rospy.Publisher("/pose_skeleton_markers", MarkerArray, queue_size=10)

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)

        # Point cloud data
        self.latest_pointcloud = None
        self.pc_width = None
        self.pc_height = None
        self.image_width = None
        self.image_height = None
        
        rospy.loginfo("MediaPipe Pose Posture Node initialized")

    def pointcloud_callback(self, pc_msg):
        """处理点云数据"""
        self.latest_pointcloud = pc_msg
        self.pc_width = pc_msg.width
        self.pc_height = pc_msg.height
        rospy.logdebug(f"Received point cloud: {self.pc_width}x{self.pc_height}")

    def get_3d_coordinates(self, x_2d, y_2d):
        """将2D图像坐标映射到3D点云坐标 - 支持structured和unstructured点云"""
        if self.latest_pointcloud is None:
            rospy.logwarn("No point cloud data available")
            return None
            
        # 确保坐标在有效范围内
        if x_2d < 0 or y_2d < 0 or x_2d >= self.image_width or y_2d >= self.image_height:
            rospy.logdebug(f"2D coordinates out of bounds: ({x_2d}, {y_2d})")
            return None
        
        try:
            # 检查点云是否为structured (organized) 格式
            if self.pc_width > 1 and self.pc_height > 1:
                # Structured点云 - 直接像素对应
                pc_x = int((x_2d / self.image_width) * self.pc_width)
                pc_y = int((y_2d / self.image_height) * self.pc_height)
                
                # 确保点云坐标在有效范围内
                if pc_x >= self.pc_width or pc_y >= self.pc_height:
                    rospy.logdebug(f"Point cloud coordinates out of bounds: ({pc_x}, {pc_y})")
                    return None
                    
                # 从structured点云中读取对应像素的3D坐标
                points_list = list(pc2.read_points(self.latest_pointcloud, 
                                                 field_names=("x", "y", "z"), 
                                                 skip_nans=True,
                                                 uvs=[(pc_x, pc_y)]))
            else:
                # Unstructured点云 - 需要通过投影找到最近的点
                rospy.loginfo(f"Using unstructured point cloud: {self.pc_width} x {self.pc_height}")
                
                # 读取所有点云数据
                all_points = list(pc2.read_points(self.latest_pointcloud, 
                                                field_names=("x", "y", "z"), 
                                                skip_nans=True))
                
                if not all_points:
                    rospy.logwarn("No valid points in point cloud")
                    return None
                
                # 将图像坐标转换为normalized coordinates (假设相机intrinsics)
                # 这里使用简化的投影模型，实际应用中需要相机标定参数
                norm_x = (x_2d - self.image_width/2) / (self.image_width/2)  # -1 to 1
                norm_y = (y_2d - self.image_height/2) / (self.image_height/2)  # -1 to 1
                
                # 找到最接近该方向的点
                min_distance = float('inf')
                best_point = None
                
                for point in all_points:
                    # 计算点在相机坐标系中的投影方向
                    if point[2] > 0.1:  # 确保点在相机前方
                        proj_x = point[0] / point[2]  # 简化投影
                        proj_y = -point[1] / point[2]  # Y轴翻转
                        
                        # 计算与目标方向的距离
                        distance = np.sqrt((proj_x - norm_x)**2 + (proj_y - norm_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_point = point
                
                if best_point is not None:
                    points_list = [best_point]
                else:
                    rospy.logwarn(f"No suitable point found for 2D coordinates ({x_2d}, {y_2d})")
                    return None
            
            if points_list:
                point_3d = points_list[0]
                # 检查是否为有效的3D点
                if not (np.isnan(point_3d[0]) or np.isnan(point_3d[1]) or np.isnan(point_3d[2])):
                    # 检查深度是否合理 (人体通常在1-4米范围内)
                    depth = point_3d[2]  # 在相机坐标系中，z就是深度
                    if 1.0 <= depth <= 4.0:
                        rospy.logdebug(f"2D ({x_2d}, {y_2d}) -> 3D ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}), depth: {depth:.3f}")
                        return Point(x=point_3d[0], y=point_3d[1], z=point_3d[2])
                    else:
                        rospy.logdebug(f"Invalid depth {depth:.3f}m for 2D coordinates ({x_2d}, {y_2d})")
                        return None
                else:
                    rospy.logdebug(f"Invalid 3D point (NaN)")
                    return None
            else:
                rospy.logdebug(f"No valid points found")
                return None
                
        except Exception as e:
            rospy.logwarn(f"Failed to get 3D coordinates: {e}")
            return None

    def extract_3d_joints(self, landmarks):
        """提取所有关节的3D坐标"""
        joints_3d = {}
        
        if self.image_width is None or self.image_height is None:
            rospy.logwarn("Image dimensions not available")
            return joints_3d

        rospy.loginfo(f"Extracting 3D joints from {len(landmarks)} landmarks")
        
        for idx, landmark in enumerate(landmarks):
            # 将归一化坐标转换为像素坐标
            x_pixel = int(landmark.x * self.image_width)
            y_pixel = int(landmark.y * self.image_height)
            
            # 获取3D坐标
            point_3d = self.get_3d_coordinates(x_pixel, y_pixel)
            
            if point_3d is not None:
                # 获取关节名称
                joint_name = self.mp_pose.PoseLandmark(idx).name
                joints_3d[joint_name] = {
                    'x': point_3d.x,
                    'y': point_3d.y, 
                    'z': point_3d.z,
                    'visibility': landmark.visibility
                }
                rospy.logdebug(f"Joint {joint_name}: ({point_3d.x:.3f}, {point_3d.y:.3f}, {point_3d.z:.3f}), vis: {landmark.visibility:.2f}")
        
        rospy.loginfo(f"Successfully extracted {len(joints_3d)} 3D joints")
        return joints_3d

    def publish_skeleton_markers(self, joints_3d, frame_id="xtion_rgb_optical_frame"):
        """发布骨架的MarkerArray用于RViz可视化"""
        if not joints_3d:
            rospy.logwarn("No 3D joints data available for marker publication")
            return
            
        rospy.loginfo(f"Publishing skeleton markers with {len(joints_3d)} joints")
        
        marker_array = MarkerArray()
        marker_id = 0
        
        # 创建关节点markers (球体)
        joints_published = 0
        for joint_name, joint_data in joints_3d.items():
            # 降低可见度阈值以获得更多关节点
            if joint_data['visibility'] < 0.3:  
                rospy.logdebug(f"Skipping joint {joint_name} due to low visibility: {joint_data['visibility']}")
                continue
                
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "joints"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # 位置
            marker.pose.position.x = joint_data['x']
            marker.pose.position.y = joint_data['y']
            marker.pose.position.z = joint_data['z']
            # 修复四元数未初始化问题
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # 大小
            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03
            
            # 所有关节点用红色
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # 红色
            
            marker.lifetime = rospy.Duration(0.5)  # 生命周期
            marker_array.markers.append(marker)
            marker_id += 1
            joints_published += 1
            
            rospy.logdebug(f"Added joint marker: {joint_name} at ({joint_data['x']:.3f}, {joint_data['y']:.3f}, {joint_data['z']:.3f})")
        
        rospy.loginfo(f"Created {joints_published} joint markers")
        
        # 创建骨架连接线markers - 使用简化的连接
        simplified_connections = [
            # 主要躯干
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_HIP'), 
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            
            # 手臂
            ('LEFT_SHOULDER', 'LEFT_ELBOW'), 
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), 
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            
            # 腿部
            ('LEFT_HIP', 'LEFT_KNEE'), 
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'), 
            ('RIGHT_KNEE', 'RIGHT_ANKLE'),
            
            # 头部到肩膀
            ('NOSE', 'LEFT_SHOULDER'),
            ('NOSE', 'RIGHT_SHOULDER')
        ]
        
        lines_published = 0
        for connection in simplified_connections:
            joint1_name, joint2_name = connection
            
            if joint1_name in joints_3d and joint2_name in joints_3d:
                joint1 = joints_3d[joint1_name]
                joint2 = joints_3d[joint2_name]
                
                # 检查两个关节的可见度
                if joint1['visibility'] < 0.3 or joint2['visibility'] < 0.3:
                    continue
                
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "skeleton_lines"
                marker.id = marker_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                
                # 线条的两个端点
                p1 = Point()
                p1.x = joint1['x']
                p1.y = joint1['y']
                p1.z = joint1['z']
                
                p2 = Point()
                p2.x = joint2['x']
                p2.y = joint2['y']
                p2.z = joint2['z']
                
                marker.points = [p1, p2]
                
                # 线条样式
                marker.scale.x = 0.01  # 线宽
                marker.color = ColorRGBA(1.0, 1.0, 1.0, 0.8)  # 白色半透明
                
                marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(marker)
                marker_id += 1
                lines_published += 1
                
                rospy.logdebug(f"Added line: {joint1_name} -> {joint2_name}")
        
        rospy.loginfo(f"Created {lines_published} skeleton lines")
        
        # 发布MarkerArray
        if marker_array.markers:
            self.skeleton_marker_pub.publish(marker_array)
            rospy.loginfo(f"Successfully published MarkerArray with {len(marker_array.markers)} markers")
            
            # 打印一些关节坐标用于调试
            for joint_name, joint_data in list(joints_3d.items())[:3]:  # 只打印前3个关节
                rospy.loginfo(f"Joint {joint_name}: x={joint_data['x']:.3f}, y={joint_data['y']:.3f}, z={joint_data['z']:.3f}")
        else:
            rospy.logwarn("No markers to publish!")

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
        else:
            return "Open hand"

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        # 更新图像尺寸
        self.image_height, self.image_width = cv_image.shape[:2]
        rospy.logdebug(f"Image dimensions: {self.image_width}x{self.image_height}")

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        posture = "unknown"
        joints_3d_info = ""

        if results.pose_landmarks:
            rospy.loginfo("Pose landmarks detected")
            
            self.mp_drawing.draw_landmarks(
                cv_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            posture = self.judge_posture(results.pose_landmarks.landmark)
            rospy.loginfo(f"Posture judged: {posture}")
            
            # 提取3D关节信息
            joints_3d = self.extract_3d_joints(results.pose_landmarks.landmark)
            
            if joints_3d:
                rospy.loginfo(f"Extracted {len(joints_3d)} 3D joints")
                
                # 发布3D骨架markers用于RViz可视化
                self.publish_skeleton_markers(joints_3d, frame_id="xtion_rgb_frame")
                
                # 构建3D关节信息字符串
                key_joints = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                             'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']
                
                joint_strings = []
                for joint_name in key_joints:
                    if joint_name in joints_3d:
                        joint = joints_3d[joint_name]
                        joint_strings.append(f"{joint_name}:({joint['x']:.3f},{joint['y']:.3f},{joint['z']:.3f})")
                
                joints_3d_info = "; ".join(joint_strings)
                rospy.loginfo(f"3D Joints string length: {len(joints_3d_info)}")
            else:
                rospy.logwarn("No 3D joints extracted")
        else:
            rospy.loginfo("No pose landmarks detected")
            
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

        # 发布消息
        rospy.loginfo(f"Publishing posture: {posture}")
        self.posture_pub.publish(posture)
        
        combined_msg = f"{posture}, {gesture}"
        rospy.loginfo(f"Publishing combined: {combined_msg}")
        self.pose_gesture_pub.publish(combined_msg)
        
        if joints_3d_info:
            rospy.loginfo(f"Publishing 3D joints info: {len(joints_3d_info)} chars")
            self.joints_3d_pub.publish(joints_3d_info)
        else:
            rospy.loginfo("No 3D joints info to publish")
        
        rospy.loginfo(f"Final status - Posture: {posture}, Gesture: {gesture}")

        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            img_msg.header = data.header
            self.image_pub.publish(img_msg)
        except Exception as e:
            rospy.logerr("Image publish error: %s", e)

    def run(self):
        rospy.loginfo("Starting MediaPipe Pose Posture Node...")
        rospy.spin()

if __name__ == '__main__':
    try:
        node = MediaPipePosePostureNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted")
    except Exception as e:
        rospy.logerr(f"Node failed: {e}")