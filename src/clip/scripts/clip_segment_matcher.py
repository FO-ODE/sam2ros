#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import rospy
import torch
import clip
import numpy as np
import cv2
from PIL import Image as PILImage
from collections import defaultdict, deque

from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PointStamped
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

from sam2ros_msgs.msg import SegmentMask


class CLIPSegmentMatcher:
    def __init__(self):
        rospy.init_node("clip_segment_matcher", anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = os.path.join(os.path.dirname(__file__), "..", "CLIP_models", "ViT-L-14.pt")
        self.model, self.preprocess = clip.load(model_path, device=self.device)

        # === ROS ===
        self.bridge = CvBridge()
        self.segment_cache = defaultdict(list)
        self.crop_cache = defaultdict(dict)
        self.mask_cache = defaultdict(dict)
        self.frame_timers = {}
        self.prompt = None

        self.depth_image = None
        self.camera_info = None

        # === 订阅者 ===
        rospy.Subscriber("/adv_robocup/sam2clip/sam_segment_crop", SegmentMask, self.segment_callback)
        rospy.Subscriber("/adv_robocup/sam2clip/clip_query", String, self.prompt_callback)
        # rospy.Subscriber("/xtion/depth/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/xtion/depth_registered/image", Image, self.depth_callback)
        rospy.Subscriber("/xtion/depth_registered/camera_info", CameraInfo, self.camera_info_callback)

        # === 发布者 ===
        self.result_pub = rospy.Publisher("/adv_robocup/sam2clip/clip_matched_object", Image, queue_size=10)
        self.depth_bbox_pub = rospy.Publisher("/adv_robocup/sam2clip/depth_bbox", Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher("/adv_robocup/sam2clip/pointcloud", PointCloud2, queue_size=1)
        self.position_pub = rospy.Publisher("/adv_robocup/waving_person/position", PointStamped, queue_size=1)

        rospy.loginfo("CLIP Segment Matcher Initialized")
        rospy.logwarn("Please send prompts to /adv_robocup/sam2clip/clip_query")
        rospy.spin()

    def prompt_callback(self, msg):
        if msg.data != self.prompt:
            self.prompt = msg.data
            rospy.loginfo(f"New prompt received: {self.prompt}")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_header = msg.header

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def segment_callback(self, msg):
        if not self.prompt:
            rospy.logwarn_throttle(1.0, "Waiting for prompt...")
            return

        try:
            frame = msg.frame_seq
            seg_id = msg.segment_id
            crop_img = self.bridge.imgmsg_to_cv2(msg.crop, desired_encoding="passthrough")
            self.crop_cache[frame][seg_id] = msg.crop

            full_mask = self.bridge.imgmsg_to_cv2(msg.mask, desired_encoding="mono8")
            self.mask_cache[frame][seg_id] = full_mask
            # self.mask_cache[frame][seg_id] = msg.mask  # Store the full mask

            # For CLIP matching
            pil_img = PILImage.fromarray(crop_img)
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_feature = self.model.encode_image(img_tensor)
                text_feature = self.model.encode_text(clip.tokenize([self.prompt]).to(self.device))
                similarity = (image_feature / image_feature.norm(dim=-1, keepdim=True)) @ \
                             (text_feature / text_feature.norm(dim=-1, keepdim=True)).T
                score = similarity.item()

            self.segment_cache[frame].append((seg_id, score))
            self.crop_cache[frame][seg_id] = msg.crop

            if frame in self.frame_timers:
                self.frame_timers[frame].shutdown()
            self.frame_timers[frame] = rospy.Timer(rospy.Duration(0.2), lambda e, f=frame: self.process_frame(f), oneshot=True)

        except Exception as e:
            rospy.logerr(f"segment_callback error: {e}")

    def process_frame(self, frame):
        if frame not in self.segment_cache or not self.segment_cache[frame]:
            return

        best_id, best_score = max(self.segment_cache[frame], key=lambda x: x[1])
        rospy.loginfo(f"[frame {frame}] \"{self.prompt}\" matched ID: {best_id}, score: {best_score:.4f}")

        if best_id not in self.crop_cache[frame]:
            rospy.logwarn(f"Segment ID {best_id} not found in image cache.")
            return

        best_match_crop = self.crop_cache[frame][best_id]
        self.result_pub.publish(best_match_crop)
        
                # debugging output for mask
        mask = self.mask_cache[frame][best_id]
        rospy.loginfo(f"mask type: {type(mask)}, shape: {getattr(mask, 'shape', None)}, unique: {np.unique(mask) if isinstance(mask, np.ndarray) else 'N/A'}")


        # 点云推理部分使用原图大小的掩码
        if self.depth_image is not None and self.camera_info is not None:
            if best_id in self.mask_cache[frame]:
                mask = self.mask_cache[frame][best_id]
                self.publish_pointcloud_and_position(mask)
            else:
                rospy.logwarn(f"Mask for best ID {best_id} not found.")

        # 清理缓存
        del self.segment_cache[frame]
        del self.crop_cache[frame]
        if frame in self.frame_timers:
            self.frame_timers[frame].shutdown()
            del self.frame_timers[frame]

    def publish_pointcloud_and_position(self, mask):
        # 将 mask 向右移动 10 像素（左边补 0）
        # offset = 10 for TIAGo
        shift = 10
        shifted_mask = np.zeros_like(mask)
        shifted_mask[:, shift:] = mask[:, :-shift]
        mask = shifted_mask
        
        # 查找边界框
        ys, xs = np.where(mask == 1)  # 使用二值化掩码查找非零像素
        if len(xs) == 0 or len(ys) == 0:
            rospy.logwarn("No non-zero pixels in mask.")
            return

        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        bbox = (x1, y1, x2, y2)

        # 可视化 mask 边框
        depth_vis = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色边框
        depth_vis[mask == 1] = [0, 255, 0] # 将掩码区域标记为绿色
        self.depth_bbox_pub.publish(self.bridge.cv2_to_imgmsg(depth_vis, encoding="bgr8"))

        # 相机内参
        K = self.camera_info.K
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        # 点云生成
        points = []
        for v, u in zip(ys, xs):
            z = float(self.depth_image[v, u]) / 1000.0
            if z == 0 or np.isnan(z):
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])

        if len(points) < 10:
            return

        # 发布点云
        header = self.depth_header
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(cloud_msg)

        # 计算中心点并发布
        pts = np.array(points)
        center = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        topk = pts[np.argsort(dists)[:int(len(pts)*0.8)]]
        avg = np.mean(topk, axis=0)

        pt_msg = PointStamped()
        pt_msg.header = header
        pt_msg.point.x, pt_msg.point.y, pt_msg.point.z = avg
        self.position_pub.publish(pt_msg)


if __name__ == "__main__":
    CLIPSegmentMatcher()
