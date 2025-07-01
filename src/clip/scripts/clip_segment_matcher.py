#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import rospy
import torch
import clip
from std_msgs.msg import String
from cv_bridge import CvBridge
from sam2ros_msgs.msg import SegmentMask
from sensor_msgs.msg import Image
from PIL import Image as PILImage

import numpy as np
import cv2
from collections import defaultdict

class CLIPSegmentMatcher:
    def __init__(self):
        rospy.init_node("clip_segment_matcher", anonymous=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "..", "CLIP_models", "ViT-L-14.pt") # ViT-L-14.pt # ViT-B-32.pt
        self.model, self.preprocess = clip.load(model_path, device=self.device)
        
        # ROS 相关
        self.bridge = CvBridge()
        self.segment_cache = defaultdict(list)  # 存储每一帧的 segment 图像列表
        self.prompt = None
        self.waiting_for_prompt_logged = False  # 标志，避免重复打印等待提示

        rospy.Subscriber("/adv_robocup/sam2clip/sam_segment_mask", SegmentMask, self.segment_callback)
        rospy.Subscriber("/adv_robocup/sam2clip/clip_query", String, self.prompt_callback)  # 可以通过这个 topic 动态设置提示词
        self.result_pub = rospy.Publisher("/adv_robocup/sam2clip/clip_matched_object", Image, queue_size=10)
        self.image_cache = defaultdict(dict)  # 保存每个 segment 的图像 {frame_seq: {segment_id: image}}

        rospy.loginfo("CLIP Segment Matcher Initialized")
        rospy.logwarn("Please send prompts to /adv_robocup/sam2clip/clip_query")
        rospy.spin()

    def prompt_callback(self, msg):
        new_prompt = msg.data
        if new_prompt != self.prompt:  # 只有当prompt真的改变时才处理
            self.prompt = new_prompt
            self.waiting_for_prompt_logged = False  # 重置标志，因为已经收到了新的 prompt
            rospy.loginfo(f"New prompt received: {self.prompt}")

    def segment_callback(self, msg):
        try:
            # 检查是否有 prompt
            if not self.prompt:
                if not self.waiting_for_prompt_logged:
                    rospy.logwarn(f"Waiting for prompt...")
                    self.waiting_for_prompt_logged = True
                return

            # 转换图像格式
            cv_img = self.bridge.imgmsg_to_cv2(msg.mask_image, desired_encoding="passthrough")
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            elif cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
            else:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # 预处理 + 编码
            pil_img = PILImage.fromarray(cv_img)
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_feature = self.model.encode_image(img_tensor)
                text_feature = self.model.encode_text(clip.tokenize([self.prompt]).to(self.device))

                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
                similarity = (image_feature @ text_feature.T).item()

            frame = msg.frame_seq
            seg_id = msg.segment_id

            self.segment_cache[frame].append((seg_id, similarity))
            self.image_cache[frame][seg_id] = msg.mask_image

            # 如果收到了旧帧（即 frame_seq 变了），处理旧帧的最佳 segment
            cached_frames = list(self.segment_cache.keys())
            for f in cached_frames:
                if f != frame:
                    best_id, best_score = max(self.segment_cache[f], key=lambda x: x[1])
                    rospy.loginfo(f"[frame {f}] \"{self.prompt}\" ID: {best_id}, score: {best_score:.4f}")

                    # 发布图像
                    if best_id in self.image_cache[f]:
                        self.result_pub.publish(self.image_cache[f][best_id])

                    # 清除缓存
                    del self.segment_cache[f]
                    del self.image_cache[f]

        except Exception as e:
            rospy.logerr(f"Error processing segment: {e}")
            # 移除了重复的等待提示信息



if __name__ == "__main__":
    CLIPSegmentMatcher()
