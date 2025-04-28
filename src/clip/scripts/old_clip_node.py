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

        ################################################################################ parameters
        self.text_prompts = ["a man pointing at something", "a man sitting on the floor"]  # prompts to compare with
        self.num_segments_to_print = 10  # number of segments to print
        ################################################################################


        self.text_features = self.encode_text_prompts(self.text_prompts)

        rospy.Subscriber("/sam2ros/sam_segment", SegmentMask, self.mask_callback, queue_size=50)
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

        logits = image_features @ self.text_features.T
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()

        main_prompt_index = 0
        main_prompt_name = self.text_prompts[main_prompt_index]

        sorted_indices = np.argsort(probs[:, main_prompt_index])[::-1]


        num_to_print = min(self.num_segments_to_print, len(sorted_indices))

        rospy.loginfo(f"[Frame {self.current_frame_seq}] Ranking by '{main_prompt_name}' probability (Top {num_to_print}):")
        for rank, idx in enumerate(sorted_indices[:num_to_print], start=1):
            seg_id = id_list[idx]
            segment_probs = probs[idx]
            main_prob = segment_probs[main_prompt_index] * 100


        GREEN = "\033[92m"
        RESET = "\033[0m"
        for rank, idx in enumerate(sorted_indices[:num_to_print], start=1):
            seg_id = id_list[idx]
            segment_probs = probs[idx]
            main_score = logits[idx][main_prompt_index]  # score for the main prompt
            line = f"  {rank}. Segment ID: {seg_id}, {main_prompt_name} with score: {main_score:.4f}"
            if rank == 1:
                rospy.loginfo(f"{GREEN}{line}{RESET}") # first line with color
            else:
                rospy.loginfo(line)

            for prompt, prob in zip(self.text_prompts, segment_probs):
                rospy.loginfo(f"       {prompt}: {prob * 100:.2f}%")


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
