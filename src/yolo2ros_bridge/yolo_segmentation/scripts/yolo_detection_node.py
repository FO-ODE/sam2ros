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


class YoloDetectionNode:
    def __init__(self):
        rospy.init_node('yolo_detection_node', anonymous=True)
        rospy.loginfo("Yolo Detection Node Initialized")

        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../YOLO_models"))

        def load_model(name):
            path = os.path.join(model_dir, name)
            return YOLO(path).to("cuda"), os.path.splitext(name)[0]

        self.tracker_config = os.path.join(model_dir, "botsort.yaml")

        self.det_model, self.det_model_name = load_model("yolo11m.pt")
        self.seg_model, self.seg_model_name = load_model("yolo11m-seg.pt")
        self.pos_model, self.pos_model_name = load_model("yolo11m-pose.pt")

        self.frame_counter = 0
        self.det_skip_frames = 2
        self.seg_skip_frames = 2
        self.pos_skip_frames = 2

        self.last_gesture_pub_time = time.time()
        
        self.latest_depth_image = None
        self.depth_sub = rospy.Subscriber("/xtion/depth/image_raw", Image, self.depth_callback, queue_size=1)
        self.depth_bbox_pub = rospy.Publisher("/ultralytics/pose/selected_person/depth_bbox", Image, queue_size=1)
        
        self.camera_info = None
        self.camera_info_sub = rospy.Subscriber("/xtion/depth/camera_info", CameraInfo, self.camera_info_callback)
        self.pointcloud_pub = rospy.Publisher("/ultralytics/pose/selected_person/pointcloud", PointCloud2, queue_size=1)
        self.position_pub = rospy.Publisher("/adv_robocup/waving_person/position", PointStamped, queue_size=1)

        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=1)
        self.seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=1)
        self.pos_image_pub = rospy.Publisher("/ultralytics/pose/image", Image, queue_size=1)
        self.gesture_pub = rospy.Publisher("/ultralytics/pose/gesture", String, queue_size=5)
        self.classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)
        self.person_crop_pub = rospy.Publisher("/ultralytics/person_crop/image", Image, queue_size=1)
        self.selected_person_pub = rospy.Publisher("/ultralytics/pose/selected_person", Image, queue_size=1)

        rospy.loginfo(f"Detection model: {self.det_model_name}")
        rospy.loginfo(f"Segmentation model: {self.seg_model_name}")
        rospy.loginfo(f"Pose model: {self.pos_model_name}")
        rospy.loginfo(f"Frame skipping - Det:{self.det_skip_frames}, Seg:{self.seg_skip_frames}, Pos:{self.pos_skip_frames}")
        rospy.spin()
        
    def camera_info_callback(self, msg):
        self.camera_info = msg

    def detect_waving_gesture(self, keypoints, person_id):
        try:
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]

            if (nose[2] > 0.5 and left_shoulder[2] > 0.5 and 
                right_shoulder[2] > 0.5 and left_wrist[2] > 0.5 and right_wrist[2] > 0.5):

                head_y = nose[1]
                left_wrist_above_head = left_wrist[1] < head_y
                right_wrist_above_head = right_wrist[1] < head_y

                if left_wrist_above_head or right_wrist_above_head:
                    return True

        except Exception as e:
            rospy.logwarn(f"Error detecting gesture for person {person_id}: {e}")

        return False
    
    def depth_callback(self, msg):
        try:
            self.latest_depth_image = ros_numpy.numpify(msg)
        except Exception as e:
            rospy.logwarn(f"Failed to receive depth image: {e}")
            
    def publish_pointcloud_from_mask(self, depth_image, mask, x1, y1, camera_info_msg):
        fx = camera_info_msg.K[0]
        fy = camera_info_msg.K[4]
        cx = camera_info_msg.K[2]
        cy = camera_info_msg.K[5]

        points = []

        for v in range(mask.shape[0]):
            for u in range(mask.shape[1]):
                if mask[v, u] == 1:
                    depth = depth_image[y1 + v, x1 + u].astype(np.float32) / 1000.0  # mm → m
                    if depth == 0 or np.isnan(depth):  # 无效点
                        continue

                    z = depth
                    x = (u + x1 - cx) * z / fx
                    y = (v + y1 - cy) * z / fy

                    points.append([x, y, z])

        # 创建 PointCloud2 消息
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = camera_info_msg.header.frame_id

        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(cloud_msg)
        
        # === 计算并发布平均位置 ===
        self.compute_and_publish_average_position(points, frame_id=camera_info_msg.header.frame_id)
        
    def compute_and_publish_average_position(self, points, frame_id="camera_link"):
        if len(points) < 10:
            rospy.logwarn("Too few points to compute position.")
            return

        # 转为 numpy 数组
        pts = np.array(points)  # shape: (N, 3)

        # Step 1: 计算几何中心
        center = np.mean(pts, axis=0)

        # Step 2: 计算每个点到中心的距离
        dists = np.linalg.norm(pts - center, axis=1)

        # Step 3: 按距离排序，保留前80%
        num_keep = int(len(pts) * 0.8)
        indices = np.argsort(dists)[:num_keep]
        filtered_pts = pts[indices]

        # Step 4: 计算均值
        avg = np.mean(filtered_pts, axis=0)

        # Step 5: 发布
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.point.x = float(avg[0])
        msg.point.y = float(avg[1])
        msg.point.z = float(avg[2])

        self.position_pub.publish(msg)

    def image_callback(self, msg):
        self.frame_counter += 1
        input_image = ros_numpy.numpify(msg)

        if (self.det_image_pub.get_num_connections() or self.person_crop_pub.get_num_connections()) and \
           (self.frame_counter % self.det_skip_frames == 0):

            det_result = self.det_model.track(
                source=input_image,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
            det_annotated = det_result[0].plot(show=False)

            if self.det_image_pub.get_num_connections():
                self.det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

            if len(det_result[0].boxes) > 0:
                classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
                names = [det_result[0].names[i] for i in classes]
                self.classes_pub.publish(String(data=str(names)))

                if det_result[0].boxes.id is not None:
                    ids = det_result[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = [-1] * len(det_result[0].boxes.xyxy)

                if self.person_crop_pub.get_num_connections():
                    for box, cls, track_id in zip(det_result[0].boxes.xyxy.cpu().numpy(), 
                                                  classes, ids):
                        if det_result[0].names[cls] == "person":
                            x1, y1, x2, y2 = map(int, box)
                            cropped_person = input_image[y1:y2, x1:x2].copy()
                            if cropped_person.size > 0:
                                self.person_crop_pub.publish(ros_numpy.msgify(Image, cropped_person, encoding="rgb8"))
                                break

        if self.seg_image_pub.get_num_connections() and (self.frame_counter % self.seg_skip_frames == 0):
            seg_result = self.seg_model(input_image, verbose=False)
            seg_annotated = seg_result[0].plot(show=False)
            self.seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))

        if (self.pos_image_pub.get_num_connections() or self.gesture_pub.get_num_connections()) and \
           (self.frame_counter % self.pos_skip_frames == 0):

            pos_result = self.pos_model.track(
                source=input_image,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
            pos_annotated = pos_result[0].plot(show=False)

            waving_ids = []

            if pos_result[0].keypoints is not None and pos_result[0].boxes is not None and len(pos_result[0].boxes) > 0:
                keypoints = pos_result[0].keypoints.xy.cpu().numpy()
                confidences = pos_result[0].keypoints.conf.cpu().numpy()

                if pos_result[0].boxes.id is not None:
                    ids = pos_result[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = list(range(len(keypoints)))

                for i, (kpts, confs, person_id) in enumerate(zip(keypoints, confidences, ids)):
                    kpts_with_conf = [[k[0], k[1], c] for k, c in zip(kpts, confs)]
                    if self.detect_waving_gesture(kpts_with_conf, person_id):
                        waving_ids.append(person_id)

            current_time = time.time()
            if waving_ids and (current_time - self.last_gesture_pub_time > 1.0):
                ids_str = ", ".join([f"person {pid}" for pid in waving_ids])
                gesture_msg = String(data=f"Detected waving gesture from {ids_str}")
                self.gesture_pub.publish(gesture_msg)
                rospy.loginfo(gesture_msg.data)
                self.last_gesture_pub_time = current_time

            if self.pos_image_pub.get_num_connections():
                self.pos_image_pub.publish(ros_numpy.msgify(Image, pos_annotated, encoding="rgb8"))
                
            if waving_ids:
                waving_ids.sort()
                selected_id = waving_ids[0]

                # 初始化 bbox 和裁剪区域
                bbox = None
                person_crop = None
                seg_result = None

                # 查找 selected_id 对应的 bbox 和 crop
                if pos_result[0].boxes.id is not None:
                    for box, pid in zip(pos_result[0].boxes.xyxy.cpu().numpy(),
                                        pos_result[0].boxes.id.cpu().numpy().astype(int)):
                        if pid == selected_id:
                            x1, y1, x2, y2 = map(int, box)
                            bbox = (x1, y1, x2, y2)
                            person_crop = input_image[y1:y2, x1:x2].copy()
                            if person_crop.size > 0:
                                seg_result = self.seg_model(person_crop, verbose=False)
                            break

                # ========== 可视化到深度图 ==========
                if bbox and person_crop is not None and self.latest_depth_image is not None:
                    x1, y1, x2, y2 = bbox

                    # 转为可显示 RGB 深度图
                    depth_vis = self.latest_depth_image.copy()
                    if len(depth_vis.shape) == 2:
                        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                    # 画 bounding box
                    cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # === 叠加 segmentation mask ===
                    if seg_result and seg_result[0].masks is not None:
                        mask_tensor = None
                        class_names = seg_result[0].names
                        classes = seg_result[0].boxes.cls.cpu().numpy().astype(int)

                        for i, cls in enumerate(classes):
                            if class_names[cls] == "person":
                                mask_tensor = seg_result[0].masks.data[i]
                                break  # 找到第一个人就停止

                        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)

                        mask_color = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                        mask_color[:, :] = [0, 255, 0]  # green

                        alpha = 0.5
                        overlay = person_crop.copy()
                        # Resize mask to match crop size
                        mask_resized = cv2.resize(mask_np, (person_crop.shape[1], person_crop.shape[0]), interpolation=cv2.INTER_NEAREST)

                        # 创建与 crop 同尺寸的彩色遮罩
                        mask_color = np.zeros_like(person_crop, dtype=np.uint8)
                        mask_color[:, :] = [0, 255, 0]  # green

                        # 应用 mask 叠加
                        overlay = person_crop.copy()
                        overlay[mask_resized == 1] = cv2.addWeighted(
                            overlay[mask_resized == 1], 1 - alpha, mask_color[mask_resized == 1], alpha, 0
                        )

                        # === 叠加 segmentation mask 到 depth_vis 的 bbox 区域 ===
                        try:
                            # Resize mask to bbox size
                            mask_resized = cv2.resize(mask_np, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                            # 在 bbox 区域中提取子图
                            roi = depth_vis[y1:y2, x1:x2]

                            # 创建绿色遮罩层（与 roi 同尺寸）
                            mask_color = np.zeros_like(roi, dtype=np.uint8)
                            mask_color[:, :] = [0, 255, 0]  # green

                            # 创建混合结果图
                            alpha = 0.5
                            blended = roi.copy()
                            blended[mask_resized == 1] = cv2.addWeighted(
                                roi[mask_resized == 1], 1 - alpha, mask_color[mask_resized == 1], alpha, 0
                            )

                            # 将混合结果写回原图
                            depth_vis[y1:y2, x1:x2] = blended

                        except Exception as e:
                            rospy.logwarn(f"Failed to blend mask on depth image: {e}")

                    # 发布带框 + mask 的深度图
                    depth_image_msg = ros_numpy.msgify(Image, depth_vis, encoding="rgb8")
                    self.depth_bbox_pub.publish(depth_image_msg)
                    
                # === 转点云并发布 ===
                if self.camera_info is not None:
                    try:
                        self.publish_pointcloud_from_mask(self.latest_depth_image, mask_resized, x1, y1, self.camera_info)
                    except Exception as e:
                        rospy.logwarn(f"Failed to publish pointcloud: {e}")

if __name__ == '__main__':
    yolo_detection_node = YoloDetectionNode()
