#!/usr/bin/env python
# coding: utf-8
############################ use clip-env

import rospy
import torch
import clip
from PIL import Image
from io import BytesIO
from cv_bridge import CvBridge
from sam2ros_msgs.msg import SegmentMask
import numpy as np
import time

class ClipNode:
    def __init__(self):
        rospy.init_node("clip_node")
        self.bridge = CvBridge()

        # 加载 CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.prompt = ["a man"]
        self.text_features = self.encode_text(self.prompt)

        # 图像缓存 & 当前帧处理 ID
        self.segment_buffer = {}
        self.last_frame_time = None
        self.frame_timeout = 0.5  # 每帧最多等待0.5秒收集完所有 segment

        rospy.Subscriber("/sam2ros/mask_segment", SegmentMask, self.segment_callback)
        rospy.loginfo("CLIP node started")
        rospy.spin()

    def encode_text(self, prompts):
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            return self.model.encode_text(text)

    def segment_callback(self, msg):
        if msg.segment_id == 0:
            # 原图，跳过
            self.segment_buffer.clear()
            self.last_frame_time = rospy.Time.now().to_sec()
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg.mask_image, "bgr8")
            self.segment_buffer[msg.segment_id] = cv_img

            # 如果超时 or 缓存数量大于一定值（可选），处理一次
            if self.last_frame_time and (rospy.Time.now().to_sec() - self.last_frame_time > self.frame_timeout):
                self.process_segments()
                self.segment_buffer.clear()
                self.last_frame_time = None

        except Exception as e:
            rospy.logerr(f"[segment_callback] Error: {e}")

    def process_segments(self):
        if not self.segment_buffer:
            return

        results = []
        with torch.no_grad():
            for seg_id, img in self.segment_buffer.items():
                pil_img = Image.fromarray(cv2rgb(img))
                image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_tensor)
                similarity = (image_features @ self.text_features.T).squeeze().item()
                results.append((seg_id, similarity))

        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)

        print("\n[CLIP] 排序结果：")
        for seg_id, score in results:
            print(f"Segment ID {seg_id}: similarity = {score:.4f}")
        print("-" * 40)


def cv2rgb(cv_img):
    """从BGR转RGB"""
    return cv_img[:, :, ::-1]

if __name__ == "__main__":
    ClipNode()
