#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sam2ros_msgs.msg import SegmentMask
import torch
import clip
from sensor_msgs.msg import Image
from PIL import Image as PILImage




class CLIPNode:
    def __init__(self):
        rospy.init_node('clip_node', anonymous=True)

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "..", "CLIP_models", "ViT-L-14.pt") # ViT-L-14.pt # ViT-B-32.pt
        self.model, self.preprocess = clip.load(model_path, device=self.device)
        self.current_frame_seq = None
        self.current_segments = {}

        ################################################################################ parameters
        # json_path = os.path.join(script_dir, "..", "prompts", "clip_behavior_prompts.json")
        json_path = os.path.join(script_dir, "..", "prompts", "prompts_temp.json")

        with open(json_path, "r") as f:
            prompt_dict = json.load(f)
        
        all_prompts = []
        for prompts in prompt_dict.values():
            all_prompts.extend(prompts)

        self.text_prompts = all_prompts
        self.num_segments_to_print = 3  # number of segments to print
        ################################################################################


        self.text_features = self.encode_text_prompts(self.text_prompts)

        # rospy.Subscriber("/xtion/rgb/image_raw", Image, self.mask_callback, queue_size=10)
        rospy.Subscriber("/ultralytics/person_crop/image", Image, self.mask_callback, queue_size=10)
        rospy.loginfo(f"CLIP Node started, using device: {self.device}")
        self.loop()


    def encode_text_prompts(self, prompts):
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            return self.model.encode_text(text).float().cpu()


    def mask_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # 直接用 msg
            frame_seq = rospy.Time.now().to_nsec()  # 你原来用 frame_seq，这里没有，就用时间戳代替吧

            if self.current_frame_seq != frame_seq:
                self.current_frame_seq = frame_seq
                self.current_segments.clear()

            self.current_segments[frame_seq] = cv_image  # 用时间戳当作segment id

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
            pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = self.preprocess(pil_img).unsqueeze(0)
            image_tensors.append(tensor)
            id_list.append(seg_id)

        image_input = torch.cat(image_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float().cpu()

        logits = image_features @ self.text_features.T
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()

        rospy.loginfo(f"[Frame {self.current_frame_seq}] Top {self.num_segments_to_print} predictions per segment:")


        GREEN = "\033[92m"
        RESET = "\033[0m"
        for idx, seg_id in enumerate(id_list):
            segment_probs = probs[idx]
            # 找出 top5
            top_indices = np.argsort(segment_probs)[::-1][:self.num_segments_to_print]

            rospy.loginfo(f"  Segment ID {seg_id}:")
            for rank, prompt_idx in enumerate(top_indices, start=1):
                prompt = self.text_prompts[prompt_idx]
                prob = segment_probs[prompt_idx] * 100
                if prob > 80:  # 如果概率大于80%
                    # with green color
                    rospy.loginfo(f"    {rank}. {GREEN}{prompt}: {prob:.2f}%{RESET}")
                elif prob > 30:
                    rospy.logwarn(f"    {rank}. {prompt}: {prob:.2f}%")
                else:
                    rospy.loginfo(f"    {rank}. {prompt}: {prob:.2f}%")

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
            pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
