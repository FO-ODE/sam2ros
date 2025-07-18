# -*- coding: utf-8 -*-
import os
import time
import ros_numpy
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
import message_filters
from collections import deque


class YoloDetectionNode:
    """
    YOLO检测节点类
    
    该类实现了基于YOLO的多功能检测系统，包括：
    - 目标检测（无跟踪）
    - 图像分割  
    - 姿态估计和手势识别（带跟踪）
    - 深度图像处理和点云生成
    """
    
    def __init__(self):
        """初始化YOLO检测节点"""
        # 初始化ROS节点
        rospy.init_node('yolo_detection_node', anonymous=True)
        rospy.loginfo("Yolo Detection Node Initialized")

        # 初始化模型和配置
        self._initialize_models()
        
        # 初始化参数
        self._initialize_parameters()
        
        # 初始化发布者和订阅者
        self._initialize_publishers_subscribers()
        
        # 输出模型信息
        self._log_model_info()
        
        # 开始ROS循环
        rospy.spin()
        
    def _initialize_models(self):
        """初始化YOLO模型"""
        # 模型文件路径
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../YOLO_models"))

        def load_model(name):
            """加载YOLO模型的辅助函数"""
            path = os.path.join(model_dir, name)
            return YOLO(path).to("cuda"), os.path.splitext(name)[0]

        # 跟踪器配置文件（仅用于姿态估计）
        self.tracker_config = os.path.join(model_dir, "botsort.yaml")

        # 加载三个不同的YOLO模型
        self.det_model, self.det_model_name = load_model("yolo11m.pt")      # 目标检测模型
        self.seg_model, self.seg_model_name = load_model("yolo11m-seg.pt")  # 分割模型
        self.pos_model, self.pos_model_name = load_model("yolo11m-pose.pt") # 姿态估计模型
        
    def _initialize_parameters(self):
        """初始化节点参数"""
        # 帧计数器，用于控制处理频率
        self.frame_counter = 0
        
        # 跳帧设置，用于控制不同模型的处理频率
        self.det_skip_frames = 1  # 检测模型每2帧处理一次
        self.seg_skip_frames = 2  # 分割模型每2帧处理一次  
        self.pos_skip_frames = 2  # 姿态模型每2帧处理一次

        # 手势发布时间控制，避免频繁发布
        self.last_gesture_pub_time = time.time()
        
        # 深度图像缓存 - 使用时间戳索引的缓存队列
        self.depth_image_buffer = deque(maxlen=10)  # 保存最近10帧深度图像
        self.latest_depth_image = None
        
        # 相机内参信息
        self.camera_info = None
        
        # 分割结果缓存，用于时间同步
        self.segmentation_cache = {}  # {timestamp: seg_result}
        self.cache_timeout = 2.0  # 缓存超时时间（秒）
        
    def _initialize_publishers_subscribers(self):
        """初始化ROS发布者和订阅者"""
        # === 使用消息同步的订阅者 ===
        # RGB图像订阅者
        self.rgb_sub = message_filters.Subscriber("/xtion/rgb/image_raw", Image)
        # 深度图像订阅者  
        self.depth_sub = message_filters.Subscriber("/xtion/depth/image_raw", Image)
        
        # 时间同步器 - 确保RGB和深度图像同步
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1  # 允许0.1秒的时间差
        )
        self.sync.registerCallback(self.synchronized_callback)
        
        # 相机内参订阅者
        self.camera_info_sub = rospy.Subscriber(
            "/xtion/depth/camera_info", CameraInfo, self.camera_info_callback
        )
        
        # === 发布者 ===
        # 深度图像相关发布者
        self.depth_bbox_pub = rospy.Publisher(
            "/ultralytics/pose/selected_person/depth_bbox", Image, queue_size=1
        )
        
        # 点云和位置发布者
        self.pointcloud_pub = rospy.Publisher(
            "/ultralytics/pose/selected_person/pointcloud", PointCloud2, queue_size=1
        )
        self.position_pub = rospy.Publisher(
            "/adv_robocup/waving_person/position", PointStamped, queue_size=10
        )
        
        # 控制指令发布者
        self.stop_head_movement = rospy.Publisher(
            "/adv_robocup/head_control/stop_head_movement", String, queue_size=1
        )
        self.stop_head_movement_debug = rospy.Publisher(
            "/head_scan_command", String, queue_size=1
        )

        # 图像结果发布者
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=1)
        self.seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=1)
        self.pos_image_pub = rospy.Publisher("/ultralytics/pose/image", Image, queue_size=1)
        
        # 检测结果发布者
        self.gesture_pub = rospy.Publisher("/ultralytics/pose/gesture", String, queue_size=5)
        self.classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)
        
        # 人物裁剪图像发布者
        self.person_crop_pub = rospy.Publisher("/ultralytics/person_crop/image", Image, queue_size=1)
        self.selected_person_pub = rospy.Publisher("/ultralytics/pose/selected_person", Image, queue_size=1)
        
    def _log_model_info(self):
        """输出模型信息到日志"""
        rospy.loginfo(f"Detection model: {self.det_model_name}")
        rospy.loginfo(f"Segmentation model: {self.seg_model_name}")
        rospy.loginfo(f"Pose model: {self.pos_model_name}")
        rospy.loginfo(f"Frame skipping - Det:{self.det_skip_frames}, Seg:{self.seg_skip_frames}, Pos:{self.pos_skip_frames}")
        
    def camera_info_callback(self, msg):
        """
        相机内参信息回调函数
        
        Args:
            msg (CameraInfo): 相机内参消息
        """
        self.camera_info = msg

    def synchronized_callback(self, rgb_msg, depth_msg):
        """
        同步的RGB和深度图像回调函数
        
        Args:
            rgb_msg (Image): RGB图像消息
            depth_msg (Image): 深度图像消息
        """
        try:
            # 转换图像数据
            input_image = ros_numpy.numpify(rgb_msg)
            depth_image = ros_numpy.numpify(depth_msg)
            
            # 存储深度图像到缓存，使用时间戳作为键
            timestamp = rgb_msg.header.stamp.to_sec()
            self.depth_image_buffer.append({
                'timestamp': timestamp,
                'image': depth_image,
                'header': depth_msg.header
            })
            
            # 更新最新深度图像
            self.latest_depth_image = depth_image
            
            # 清理过期的分割缓存
            self._cleanup_segmentation_cache()
            
            # 处理图像
            self._process_synchronized_images(input_image, depth_image, timestamp, rgb_msg.header)
            
        except Exception as e:
            rospy.logwarn(f"Error in synchronized callback: {e}")

    def _cleanup_segmentation_cache(self):
        """清理过期的分割结果缓存"""
        current_time = time.time()
        expired_keys = [
            ts for ts in self.segmentation_cache.keys() 
            if current_time - ts > self.cache_timeout
        ]
        for key in expired_keys:
            del self.segmentation_cache[key]

    def _process_synchronized_images(self, input_image, depth_image, timestamp, rgb_header):
        """
        处理同步的RGB和深度图像
        
        Args:
            input_image: RGB图像
            depth_image: 深度图像  
            timestamp: 时间戳
            rgb_header: RGB图像头信息
        """
        self.frame_counter += 1

        # 1. 目标检测处理
        if self._should_process_detection():
            self._process_detection(input_image)

        # 2. 图像分割处理 - 存储到缓存
        if self._should_process_segmentation():
            seg_result = self._process_segmentation_with_cache(input_image, timestamp)

        # 3. 姿态估计和手势识别处理
        if self._should_process_pose():
            self._process_pose_and_gesture_synchronized(input_image, depth_image, timestamp, rgb_header)

    def detect_waving_gesture(self, keypoints, person_id):
        """
        检测挥手手势
        
        通过分析关键点位置判断是否存在挥手动作：
        - 检查鼻子、肩膀、手腕关键点的置信度
        - 判断手腕是否高于头部位置
        
        Args:
            keypoints: 人体关键点坐标和置信度 [[x, y, confidence], ...]
            person_id: 人员ID
            
        Returns:
            bool: 是否检测到挥手手势
        """
        try:
            # 获取关键点 (COCO格式)
            nose = keypoints[0]           # 鼻子
            left_shoulder = keypoints[5]  # 左肩
            right_shoulder = keypoints[6] # 右肩
            left_wrist = keypoints[9]     # 左手腕
            right_wrist = keypoints[10]   # 右手腕

            # 检查关键点置信度是否足够高
            if (nose[2] > 0.5 and left_shoulder[2] > 0.5 and 
                right_shoulder[2] > 0.5 and left_wrist[2] > 0.5 and right_wrist[2] > 0.5):

                # 获取头部y坐标（鼻子位置）
                head_y = nose[1]
                
                # 判断手腕是否高于头部（y坐标更小表示更高）
                left_wrist_above_head = left_wrist[1] < head_y
                right_wrist_above_head = right_wrist[1] < head_y

                # 任一手腕高于头部即认为是挥手
                if left_wrist_above_head or right_wrist_above_head:
                    return True

        except Exception as e:
            rospy.logwarn(f"Error detecting gesture for person {person_id}: {e}")

        return False
            
    def publish_pointcloud_from_mask(self, depth_image, mask, x1, y1, camera_info_msg):
        """
        从分割掩码和深度图生成并发布点云
        
        Args:
            depth_image: 深度图像 (numpy array)
            mask: 分割掩码 (numpy array, 0或1)
            x1, y1: 边界框左上角坐标
            camera_info_msg: 相机内参信息
        """
        # 获取相机内参
        fx = camera_info_msg.K[0]  # x方向焦距
        fy = camera_info_msg.K[4]  # y方向焦距
        cx = camera_info_msg.K[2]  # 主点x坐标
        cy = camera_info_msg.K[5]  # 主点y坐标

        points = []

        # 遍历掩码区域，计算3D坐标
        for v in range(mask.shape[0]):
            for u in range(mask.shape[1]):
                if mask[v, u] == 1:  # 只处理掩码内的点
                    # 获取深度值并转换为米
                    depth = depth_image[y1 + v, x1 + u].astype(np.float32) / 1000.0  # mm → m
                    
                    # 跳过无效深度点
                    if depth == 0 or np.isnan(depth):
                        continue

                    # 使用相机内参转换为3D坐标
                    z = depth
                    x = (u + x1 - cx) * z / fx
                    y = (v + y1 - cy) * z / fy

                    points.append([x, y, z])

        # 创建并发布PointCloud2消息
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = camera_info_msg.header.frame_id

        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(cloud_msg)
        
        # 计算并发布平均位置
        self.compute_and_publish_average_position(points, frame_id=camera_info_msg.header.frame_id)
        
    def compute_and_publish_average_position(self, points, frame_id="camera_link"):
        """
        计算点云的平均位置并发布
        
        使用统计滤波方法：
        1. 计算所有点的几何中心
        2. 计算每个点到中心的距离
        3. 保留距离最近的80%的点
        4. 计算这些点的平均位置
        
        Args:
            points: 3D点列表 [[x, y, z], ...]
            frame_id: 坐标系名称
        """
        if len(points) < 10:
            rospy.logwarn("Too few points to compute position.")
            return

        # 转为numpy数组
        pts = np.array(points)  # shape: (N, 3)

        # Step 1: 计算几何中心
        center = np.mean(pts, axis=0)

        # Step 2: 计算每个点到中心的距离
        dists = np.linalg.norm(pts - center, axis=1)

        # Step 3: 按距离排序，保留前80%的点（去除离群点）
        num_keep = int(len(pts) * 0.8)
        indices = np.argsort(dists)[:num_keep]
        filtered_pts = pts[indices]

        # Step 4: 计算平均位置
        avg = np.mean(filtered_pts, axis=0)

        # Step 5: 创建并发布位置消息
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.point.x = float(avg[0])
        msg.point.y = float(avg[1])
        msg.point.z = float(avg[2])

    def compute_and_publish_average_position_synchronized(self, points, rgb_header):
        """
        计算点云的平均位置并发布 - 同步版本
        
        使用统计滤波方法：
        1. 计算所有点的几何中心
        2. 计算每个点到中心的距离
        3. 保留距离最近的80%的点
        4. 计算这些点的平均位置
        
        Args:
            points: 3D点列表 [[x, y, z], ...]
            rgb_header: RGB图像头信息
        """
        if len(points) < 10:
            rospy.logwarn("Too few points to compute position.")
            return

        # 转为numpy数组
        pts = np.array(points)  # shape: (N, 3)

        # Step 1: 计算几何中心
        center = np.mean(pts, axis=0)

        # Step 2: 计算每个点到中心的距离
        dists = np.linalg.norm(pts - center, axis=1)

        # Step 3: 按距离排序，保留前80%的点（去除离群点）
        num_keep = int(len(pts) * 0.8)
        indices = np.argsort(dists)[:num_keep]
        filtered_pts = pts[indices]

        # Step 4: 计算平均位置
        avg = np.mean(filtered_pts, axis=0)

        # Step 5: 创建并发布位置消息 - 使用RGB图像的时间戳
        msg = PointStamped()
        msg.header.stamp = rgb_header.stamp  # 使用RGB图像的时间戳确保同步
        msg.header.frame_id = rgb_header.frame_id
        msg.point.x = float(avg[0])
        msg.point.y = float(avg[1])
        msg.point.z = float(avg[2])

        self.position_pub.publish(msg)
        """
        主图像处理回调函数
        
        处理RGB图像，依次执行：
        1. 目标检测和跟踪
        2. 图像分割
        3. 姿态估计和手势识别
        4. 深度信息处理和点云生成
        
        Args:
            msg (Image): RGB图像消息
        """
        self.frame_counter += 1
        input_image = ros_numpy.numpify(msg)

        # 1. 目标检测处理
        if self._should_process_detection():
            self._process_detection(input_image)

        # 2. 图像分割处理
        if self._should_process_segmentation():
            self._process_segmentation(input_image)

        # 3. 姿态估计和手势识别处理
        if self._should_process_pose():
            self._process_pose_and_gesture(input_image)
    
    def _should_process_detection(self):
        """判断是否应该处理检测任务"""
        return ((self.det_image_pub.get_num_connections() or self.person_crop_pub.get_num_connections()) 
                and (self.frame_counter % self.det_skip_frames == 0))
    
    def _should_process_segmentation(self):
        """判断是否应该处理分割任务"""
        return (self.seg_image_pub.get_num_connections() 
                and (self.frame_counter % self.seg_skip_frames == 0))
    
    def _should_process_pose(self):
        """判断是否应该处理姿态估计任务"""
        return ((self.pos_image_pub.get_num_connections() or self.gesture_pub.get_num_connections()) 
                and (self.frame_counter % self.pos_skip_frames == 0))
    
    def _process_detection(self, input_image):
        """
        处理目标检测
        
        Args:
            input_image: 输入RGB图像
        """
        # 执行目标检测（不使用跟踪）
        det_result = self.det_model(
            source=input_image,
            verbose=False
        )
        
        # 生成标注图像
        det_annotated = det_result[0].plot(show=False)

        # 发布检测结果图像
        if self.det_image_pub.get_num_connections():
            self.det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

        # 处理检测到的对象
        if len(det_result[0].boxes) > 0:
            self._publish_detection_classes(det_result[0])
            # self._publish_person_crops(det_result[0], input_image)
    
    def _publish_detection_classes(self, result):
        """发布检测到的类别信息"""
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = [result.names[i] for i in classes]
        self.classes_pub.publish(String(data=str(names)))
    
    def _publish_person_crops(self, result, input_image):
        """发布检测到的人物裁剪图像"""
        if not self.person_crop_pub.get_num_connections():
            return
            
        classes = result.boxes.cls.cpu().numpy().astype(int)

        # 查找并发布第一个人物的裁剪图像
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), classes):
            if result.names[cls] == "person":
                x1, y1, x2, y2 = map(int, box)
                cropped_person = input_image[y1:y2, x1:x2].copy()
                if cropped_person.size > 0:
                    self.person_crop_pub.publish(ros_numpy.msgify(Image, cropped_person, encoding="rgb8"))
                    break  # 只发布第一个人物
    
    def _process_segmentation_with_cache(self, input_image, timestamp):
        """
        处理图像分割并缓存结果
        
        Args:
            input_image: 输入RGB图像
            timestamp: 时间戳
            
        Returns:
            分割结果
        """
        seg_result = self.seg_model(input_image, verbose=False)
        
        # 缓存分割结果
        self.segmentation_cache[timestamp] = seg_result
        
        # 发布分割图像（如果有订阅者）
        if self.seg_image_pub.get_num_connections():
            seg_annotated = seg_result[0].plot(show=False)
            self.seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))
        
        return seg_result

    def _get_synchronized_depth_image(self, target_timestamp, tolerance=0.1):
        """
        获取与目标时间戳同步的深度图像
        
        Args:
            target_timestamp: 目标时间戳
            tolerance: 时间容差（秒）
            
        Returns:
            tuple: (depth_image, header) 或 (None, None)
        """
        best_match = None
        min_time_diff = float('inf')
        
        for depth_data in self.depth_image_buffer:
            time_diff = abs(depth_data['timestamp'] - target_timestamp)
            if time_diff < tolerance and time_diff < min_time_diff:
                min_time_diff = time_diff
                best_match = depth_data
        
        if best_match:
            return best_match['image'], best_match['header']
        
        # 如果没有找到同步的深度图像，使用最新的
        rospy.logwarn(f"No synchronized depth image found for timestamp {target_timestamp}, using latest")
        return self.latest_depth_image, None

    def _process_pose_and_gesture_synchronized(self, input_image, depth_image, timestamp, rgb_header):
        """
        处理同步的姿态估计和手势识别
        
        Args:
            input_image: 输入RGB图像
            depth_image: 对应的深度图像
            timestamp: 时间戳
            rgb_header: RGB图像头信息
        """
        # 执行姿态估计和跟踪
        pos_result = self.pos_model.track(
            source=input_image,
            persist=True,
            tracker=self.tracker_config,
            verbose=False
        )
        pos_annotated = pos_result[0].plot(show=False)

        # 检测挥手手势
        waving_ids = self._detect_waving_gestures(pos_result[0])

        # 处理手势检测结果
        if waving_ids:
            self._handle_gesture_detection_synchronized(waving_ids, pos_result[0], input_image, depth_image, timestamp, rgb_header)

        # 发布姿态估计结果图像
        if self.pos_image_pub.get_num_connections():
            self.pos_image_pub.publish(ros_numpy.msgify(Image, pos_annotated, encoding="rgb8"))
    
    def _process_pose_and_gesture(self, input_image):
        """
        处理姿态估计和手势识别
        
        Args:
            input_image: 输入RGB图像
        """
        # 执行姿态估计和跟踪
        pos_result = self.pos_model.track(
            source=input_image,
            persist=True,
            tracker=self.tracker_config,
            verbose=False
        )
        pos_annotated = pos_result[0].plot(show=False)

        # 检测挥手手势
        waving_ids = self._detect_waving_gestures(pos_result[0])

        # 处理手势检测结果
        if waving_ids:
            self._handle_gesture_detection(waving_ids, pos_result[0], input_image)

        # 发布姿态估计结果图像
        if self.pos_image_pub.get_num_connections():
            self.pos_image_pub.publish(ros_numpy.msgify(Image, pos_annotated, encoding="rgb8"))
    
    def _detect_waving_gestures(self, pose_result):
        """
        检测挥手手势
        
        Args:
            pose_result: 姿态估计结果
            
        Returns:
            list: 挥手人员的ID列表
        """
        waving_ids = []

        # 检查是否有关键点和边界框
        if pose_result.keypoints is None or pose_result.boxes is None or len(pose_result.boxes) == 0:
            return waving_ids

        # 获取关键点和置信度
        keypoints = pose_result.keypoints.xy.cpu().numpy()
        confidences = pose_result.keypoints.conf.cpu().numpy()

        # 获取跟踪ID
        if pose_result.boxes.id is not None:
            ids = pose_result.boxes.id.cpu().numpy().astype(int)
        else:
            ids = list(range(len(keypoints)))

        # 逐个检测每个人的手势
        for i, (kpts, confs, person_id) in enumerate(zip(keypoints, confidences, ids)):
            kpts_with_conf = [[k[0], k[1], c] for k, c in zip(kpts, confs)]
            if self.detect_waving_gesture(kpts_with_conf, person_id):
                waving_ids.append(person_id)

        return waving_ids
    
    def _handle_gesture_detection_synchronized(self, waving_ids, pose_result, input_image, depth_image, timestamp, rgb_header):
        """
        处理同步的手势检测结果
        
        Args:
            waving_ids: 挥手人员ID列表
            pose_result: 姿态估计结果
            input_image: 输入RGB图像
            depth_image: 对应的深度图像
            timestamp: 时间戳
            rgb_header: RGB图像头信息
        """
        current_time = time.time()
        
        # 发布手势检测消息（限制频率）
        if current_time - self.last_gesture_pub_time > 1.0:
            self._publish_gesture_message(waving_ids)
            self._publish_stop_head_movement()
            self.last_gesture_pub_time = current_time

        # 处理选中的人员（ID最小的挥手人员）
        waving_ids.sort()
        selected_id = waving_ids[0]
        self._process_selected_person_synchronized(selected_id, pose_result, input_image, depth_image, timestamp, rgb_header)

    def _process_selected_person_synchronized(self, selected_id, pose_result, input_image, depth_image, timestamp, rgb_header):
        """
        处理选中的人员（挥手的人）- 同步版本
        
        Args:
            selected_id: 选中人员的ID
            pose_result: 姿态估计结果
            input_image: 输入RGB图像
            depth_image: 对应的深度图像
            timestamp: 时间戳
            rgb_header: RGB图像头信息
        """
        # 查找选中人员的边界框
        bbox, person_crop = self._find_person_bbox_and_crop(selected_id, pose_result, input_image)
        
        if bbox is None or person_crop is None:
            return
        
        # 发布选中人员的裁剪图像
        if self.selected_person_pub.get_num_connections() and person_crop.size > 0:
            self.selected_person_pub.publish(ros_numpy.msgify(Image, person_crop, encoding="rgb8"))
        
        # 获取同时间戳的分割结果，如果没有则实时计算
        seg_result = self._get_or_compute_segmentation(person_crop, timestamp)
        
        # 处理深度信息和可视化 - 使用同步的深度图像
        self._process_depth_visualization_synchronized(bbox, person_crop, seg_result, depth_image, rgb_header)

    def _get_or_compute_segmentation(self, person_crop, timestamp, tolerance=0.1):
        """
        获取或计算分割结果
        
        Args:
            person_crop: 人员裁剪图像
            timestamp: 时间戳
            tolerance: 时间容差
            
        Returns:
            分割结果
        """
        # 首先尝试从缓存获取同时间戳的分割结果
        best_cached_result = None
        min_time_diff = float('inf')
        
        for cached_timestamp, cached_result in self.segmentation_cache.items():
            time_diff = abs(cached_timestamp - timestamp)
            if time_diff < tolerance and time_diff < min_time_diff:
                min_time_diff = time_diff
                best_cached_result = cached_result
        
        # 如果找到了同步的缓存结果，直接使用
        if best_cached_result is not None:
            rospy.logdebug(f"Using cached segmentation result with time diff: {min_time_diff:.3f}s")
            return best_cached_result
        
        # 如果没有找到缓存结果，实时计算
        rospy.logdebug("Computing real-time segmentation for person crop")
        return self.seg_model(person_crop, verbose=False)

    def _process_depth_visualization_synchronized(self, bbox, person_crop, seg_result, depth_image, rgb_header):
        """
        处理同步的深度图可视化和点云生成
        
        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            person_crop: 人员裁剪图像
            seg_result: 分割结果
            depth_image: 同步的深度图像
            rgb_header: RGB图像头信息
        """
        if depth_image is None:
            rospy.logwarn("No synchronized depth image available")
            return
            
        x1, y1, x2, y2 = bbox
        
        # 创建深度图可视化
        depth_vis = self._create_depth_visualization_from_image(bbox, depth_image)
        
        # 获取人员分割掩码
        mask_resized = self._get_person_segmentation_mask(seg_result, bbox)
        
        if mask_resized is not None:
            # 在深度图上叠加分割掩码
            self._overlay_segmentation_on_depth(depth_vis, mask_resized, bbox)
            
            # 发布深度图可视化结果
            depth_image_msg = ros_numpy.msgify(Image, depth_vis, encoding="rgb8")
            self.depth_bbox_pub.publish(depth_image_msg)
            
            # 生成并发布点云 - 使用同步的深度图像
            self._generate_and_publish_pointcloud_synchronized(mask_resized, bbox, depth_image, rgb_header)

    def _create_depth_visualization_from_image(self, bbox, depth_image):
        """从指定深度图像创建深度图可视化"""
        x1, y1, x2, y2 = bbox
        
        # 转换深度图为可显示的RGB格式
        depth_vis = depth_image.copy()
        if len(depth_vis.shape) == 2:
            depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        # 绘制边界框
        cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return depth_vis

    def _generate_and_publish_pointcloud_synchronized(self, mask_resized, bbox, depth_image, rgb_header):
        """生成并发布同步的点云"""
        if self.camera_info is None:
            rospy.logwarn("Camera info not available for pointcloud generation")
            return
            
        x1, y1, x2, y2 = bbox
        
        try:
            # 使用同步的深度图像生成点云
            self.publish_pointcloud_from_mask_synchronized(
                depth_image, mask_resized, x1, y1, self.camera_info, rgb_header
            )
        except Exception as e:
            rospy.logwarn(f"Failed to publish synchronized pointcloud: {e}")

    def publish_pointcloud_from_mask_synchronized(self, depth_image, mask, x1, y1, camera_info_msg, rgb_header):
        """
        从同步的分割掩码和深度图生成并发布点云
        
        Args:
            depth_image: 同步的深度图像 (numpy array)
            mask: 分割掩码 (numpy array, 0或1)
            x1, y1: 边界框左上角坐标
            camera_info_msg: 相机内参信息
            rgb_header: RGB图像头信息
        """
        # 获取相机内参
        fx = camera_info_msg.K[0]  # x方向焦距
        fy = camera_info_msg.K[4]  # y方向焦距
        cx = camera_info_msg.K[2]  # 主点x坐标
        cy = camera_info_msg.K[5]  # 主点y坐标

        points = []

        # 遍历掩码区域，计算3D坐标
        for v in range(mask.shape[0]):
            for u in range(mask.shape[1]):
                if mask[v, u] == 1:  # 只处理掩码内的点
                    # 获取深度值并转换为米
                    depth = depth_image[y1 + v, x1 + u].astype(np.float32) / 1000.0  # mm → m
                    
                    # 跳过无效深度点
                    if depth == 0 or np.isnan(depth):
                        continue

                    # 使用相机内参转换为3D坐标
                    z = depth
                    x = (u + x1 - cx) * z / fx
                    y = (v + y1 - cy) * z / fy

                    points.append([x, y, z])

        # 创建并发布PointCloud2消息 - 使用RGB图像的时间戳
        header = rospy.Header()
        header.stamp = rgb_header.stamp  # 使用RGB图像的时间戳确保同步
        header.frame_id = camera_info_msg.header.frame_id

        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(cloud_msg)
        
        # 计算并发布平均位置 - 使用同步的时间戳
        self.compute_and_publish_average_position_synchronized(points, rgb_header)
    
    def _publish_gesture_message(self, waving_ids):
        """发布手势检测消息"""
        ids_str = ", ".join([f"person {pid}" for pid in waving_ids])
        gesture_msg = String(data=f"Detected waving gesture from {ids_str}")
        self.gesture_pub.publish(gesture_msg)
        rospy.loginfo(gesture_msg.data)
    
    def _publish_stop_head_movement(self):
        """发布停止头部运动指令"""
        stop_msg = String(data="stop")
        self.stop_head_movement.publish(stop_msg)
        self.stop_head_movement_debug.publish(stop_msg)
    
    def _process_selected_person(self, selected_id, pose_result, input_image):
        """
        处理选中的人员（挥手的人）
        
        Args:
            selected_id: 选中人员的ID
            pose_result: 姿态估计结果
            input_image: 输入RGB图像
        """
        # 查找选中人员的边界框
        bbox, person_crop = self._find_person_bbox_and_crop(selected_id, pose_result, input_image)
        
        if bbox is None or person_crop is None:
            return
        
        # 发布选中人员的裁剪图像
        if self.selected_person_pub.get_num_connections() and person_crop.size > 0:
            self.selected_person_pub.publish(ros_numpy.msgify(Image, person_crop, encoding="rgb8"))
        
        # 对人员区域进行分割
        seg_result = self.seg_model(person_crop, verbose=False)
        
        # 处理深度信息和可视化
        self._process_depth_visualization(bbox, person_crop, seg_result)
    
    def _find_person_bbox_and_crop(self, selected_id, pose_result, input_image):
        """
        查找指定人员的边界框和裁剪图像
        
        Args:
            selected_id: 人员ID
            pose_result: 姿态估计结果
            input_image: 输入RGB图像
            
        Returns:
            tuple: (bbox, person_crop) 或 (None, None)
        """
        if pose_result.boxes.id is None:
            return None, None
            
        for box, pid in zip(pose_result.boxes.xyxy.cpu().numpy(),
                           pose_result.boxes.id.cpu().numpy().astype(int)):
            if pid == selected_id:
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2, y2)
                person_crop = input_image[y1:y2, x1:x2].copy()
                if person_crop.size > 0:
                    return bbox, person_crop
        
        return None, None
    
    def _process_depth_visualization(self, bbox, person_crop, seg_result):
        """
        处理深度图可视化和点云生成
        
        Args:
            bbox: 边界框 (x1, y1, x2, y2)
            person_crop: 人员裁剪图像
            seg_result: 分割结果
        """
        if self.latest_depth_image is None:
            return
            
        x1, y1, x2, y2 = bbox
        
        # 创建深度图可视化
        depth_vis = self._create_depth_visualization(bbox)
        
        # 获取人员分割掩码
        mask_resized = self._get_person_segmentation_mask(seg_result, bbox)
        
        if mask_resized is not None:
            # 在深度图上叠加分割掩码
            self._overlay_segmentation_on_depth(depth_vis, mask_resized, bbox)
            
            # 发布深度图可视化结果
            depth_image_msg = ros_numpy.msgify(Image, depth_vis, encoding="rgb8")
            self.depth_bbox_pub.publish(depth_image_msg)
            
            # 生成并发布点云
            self._generate_and_publish_pointcloud(mask_resized, bbox)
    
    def _create_depth_visualization(self, bbox):
        """创建深度图可视化"""
        x1, y1, x2, y2 = bbox
        
        # 转换深度图为可显示的RGB格式
        depth_vis = self.latest_depth_image.copy()
        if len(depth_vis.shape) == 2:
            depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        # 绘制边界框
        cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return depth_vis
    
    def _get_person_segmentation_mask(self, seg_result, bbox):
        """获取人员分割掩码"""
        if not seg_result or seg_result[0].masks is None:
            return None
            
        x1, y1, x2, y2 = bbox
        class_names = seg_result[0].names
        classes = seg_result[0].boxes.cls.cpu().numpy().astype(int)

        # 查找人员类别的掩码
        for i, cls in enumerate(classes):
            if class_names[cls] == "person":
                mask_tensor = seg_result[0].masks.data[i]
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                
                # 调整掩码大小以匹配边界框
                mask_resized = cv2.resize(mask_np, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                return mask_resized
                
        return None
    
    def _overlay_segmentation_on_depth(self, depth_vis, mask_resized, bbox):
        """在深度图上叠加分割掩码"""
        x1, y1, x2, y2 = bbox
        
        try:
            # 提取深度图的边界框区域
            roi = depth_vis[y1:y2, x1:x2]

            # 创建绿色遮罩层
            mask_color = np.zeros_like(roi, dtype=np.uint8)
            mask_color[:, :] = [0, 255, 0]  # 绿色

            # 混合掩码和原图
            alpha = 0.5
            blended = roi.copy()
            blended[mask_resized == 1] = cv2.addWeighted(
                roi[mask_resized == 1], 1 - alpha, 
                mask_color[mask_resized == 1], alpha, 0
            )

            # 将混合结果写回深度图
            depth_vis[y1:y2, x1:x2] = blended

        except Exception as e:
            rospy.logwarn(f"Failed to blend mask on depth image: {e}")
    
    def _generate_and_publish_pointcloud(self, mask_resized, bbox):
        """生成并发布点云"""
        if self.camera_info is None:
            return
            
        x1, y1, x2, y2 = bbox
        
        try:
            self.publish_pointcloud_from_mask(
                self.latest_depth_image, mask_resized, x1, y1, self.camera_info
            )
        except Exception as e:
            rospy.logwarn(f"Failed to publish pointcloud: {e}")

if __name__ == '__main__':
    """
    主程序入口
    
    创建并运行YOLO检测节点，用于：
    - 实时检测和跟踪人员
    - 识别挥手手势
    - 生成点云数据
    - 控制机器人头部运动
    """
    try:
        yolo_detection_node = YoloDetectionNode()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLO Detection Node terminated.")
    except Exception as e:
        rospy.logerr(f"Failed to start YOLO Detection Node: {e}")
