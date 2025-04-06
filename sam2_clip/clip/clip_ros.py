#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sam2ros_msgs.msg import SegmentMask
import torch
import clip
from PIL import Image

class CLIPNode:
    def __init__(self):
        rospy.init_node('clip_node', anonymous=True)

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.current_frame_seq = None
        self.current_segments = {}

        self.text_prompts = ["a person"]  # 提示
        self.text_features = self.encode_text_prompts(self.text_prompts)

        rospy.Subscriber("/sam2ros/mask_segment", SegmentMask, self.mask_callback, queue_size=50)
        rospy.loginfo(f"CLIP Node started, using device: {self.device}, prompts: {self.text_prompts}")
        self.loop()

    def encode_text_prompts(self, prompts):
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            return self.model.encode_text(text).float().cpu()

    def mask_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg.mask_image, "bgr8")
            frame_seq = msg.frame_seq

            if self.current_frame_seq != frame_seq:
                self.current_frame_seq = frame_seq
                self.current_segments.clear()

            self.current_segments[msg.segment_id] = cv_image

        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def loop(self):
        rate = rospy.Rate(2)  # 降低频率，确保处理完一帧再进入下一帧
        while not rospy.is_shutdown():
            self.process_segments()
            rate.sleep()

    def process_segments(self):
        if not self.current_segments:
            return

        segments = {k: v for k, v in self.current_segments.items() if k != 0}
        if not segments:
            return

        image_tensors = []
        id_list = []

        for seg_id, img in sorted(segments.items()):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = self.preprocess(pil_img).unsqueeze(0)
            image_tensors.append(tensor)
            id_list.append(seg_id)

        image_input = torch.cat(image_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float().cpu()

        similarities = (image_features @ self.text_features.T).numpy()
        best_scores = similarities.max(axis=1)  # 每个 segment 与所有 prompts 的最大相似度

        # 将所有 segment 按得分排序
        sorted_results = sorted(zip(id_list, best_scores), key=lambda x: x[1], reverse=True)

        rospy.loginfo(f"[Frame {self.current_frame_seq}] CLIP ranking:")
        for rank, (seg_id, score) in enumerate(sorted_results, start=1):
            rospy.loginfo(f"  {rank}. Segment ID: {seg_id}, Score: {score:.4f}")

        self.current_segments.clear()




    def process_segments_single(self):
        if not self.current_segments:
            return

        segments = {k: v for k, v in self.current_segments.items() if k != 0}  # 排除原图
        if not segments:
            return

        image_tensors = []
        id_list = []

        for seg_id, img in sorted(segments.items()):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = self.preprocess(pil_img).unsqueeze(0)
            image_tensors.append(tensor)
            id_list.append(seg_id)

        image_input = torch.cat(image_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float().cpu()

        similarities = (image_features @ self.text_features.T).numpy()

        # 对每个 segment，取其与所有 prompt 的最大相似度
        best_scores = similarities.max(axis=1)  # shape: (num_segments,)
        best_idx = np.argmax(best_scores)       # index of best segment

        best_seg_id = id_list[best_idx]
        best_score = best_scores[best_idx]

        rospy.loginfo(
            f"[Frame {self.current_frame_seq}] Best Segment ID: {best_seg_id}, Score: {best_score:.4f}"
        )

        # 打印一次就清空，避免重复
        self.current_segments.clear()

if __name__ == '__main__':
    try:
        CLIPNode()
    except rospy.ROSInterruptException:
        pass
