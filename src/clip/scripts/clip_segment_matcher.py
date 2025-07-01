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
        self.frame_timers = {}  # 每个frame_seq一个定时器



        rospy.Subscriber("/adv_robocup/sam2clip/sam_segment_mask", SegmentMask, self.segment_callback)
        rospy.Subscriber("/adv_robocup/sam2clip/clip_query", String, self.prompt_callback)  # 可以通过这个 topic 动态设置提示词
        self.result_pub = rospy.Publisher("/adv_robocup/sam2clip/clip_matched_object", Image, queue_size=10)
        self.image_cache = defaultdict(dict)  # 保存每个 segment 的图像 {frame_seq: {segment_id: image}}

        rospy.loginfo("CLIP Segment Matcher Initialized")
        rospy.logwarn("Please send prompts to /adv_robocup/sam2clip/clip_query")
        rospy.logwarn("Waiting for prompts...")
        rospy.spin()

    def prompt_callback(self, msg):
        new_prompt = msg.data
        if new_prompt != self.prompt:  # 只有当prompt真的改变时才处理
            self.prompt = new_prompt
            rospy.loginfo(f"New prompt received: {self.prompt}")
            
    def process_frame(self, frame):
        try:
            if frame not in self.segment_cache or not self.segment_cache[frame]:
                return

            best_segment = max(self.segment_cache[frame], key=lambda x: x[1])
            best_id, best_score = best_segment

            rospy.loginfo(f"[frame {frame}] \"{self.prompt}\" ID: {best_id}, score: {best_score:.4f}")

            if best_id in self.image_cache[frame]:
                self.result_pub.publish(self.image_cache[frame][best_id])
            else:
                rospy.logwarn(f"Segment ID {best_id} not found in image cache for frame {frame}")

            # 清理缓存和定时器
            del self.segment_cache[frame]
            del self.image_cache[frame]
            if frame in self.frame_timers:
                self.frame_timers[frame].shutdown()
                del self.frame_timers[frame]

        except Exception as e:
            rospy.logerr(f"Error in process_frame: {e}")


    def segment_callback(self, msg):
        try:
            frame = msg.frame_seq
            seg_id = msg.segment_id

            # 图像转换
            cv_img = self.bridge.imgmsg_to_cv2(msg.mask_image, desired_encoding="passthrough")
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            elif cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
            else:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            pil_img = PILImage.fromarray(cv_img)
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_feature = self.model.encode_image(img_tensor)
                text_feature = self.model.encode_text(clip.tokenize([self.prompt]).to(self.device))

                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
                similarity = (image_feature @ text_feature.T).item()

            self.segment_cache[frame].append((seg_id, similarity))
            self.image_cache[frame][seg_id] = msg.mask_image

            # 启动/重启定时器：100ms后触发该帧处理
            if frame in self.frame_timers:
                self.frame_timers[frame].shutdown()  # 停止旧定时器

            self.frame_timers[frame] = rospy.Timer(rospy.Duration(0.2), lambda event, f=frame: self.process_frame(f), oneshot=True)

        except Exception as e:
            rospy.logwarn(f"Waiting for prompt...")




if __name__ == "__main__":
    CLIPSegmentMatcher()
