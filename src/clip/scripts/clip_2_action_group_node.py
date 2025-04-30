#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import clip
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import json
import os
import numpy as np

class ClipActionGroupNode:
    def __init__(self):
        self.need_update = False

        rospy.init_node('clip_action_group_node', anonymous=True)

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "..", "CLIP_models", "ViT-L-14.pt") # ViT-L-14.pt # ViT-B-32.pt
        self.model, self.preprocess = clip.load(model_path, device=self.device)

        script_dir = os.path.dirname(__file__)
        prompt_map_path = os.path.join(script_dir, "..", "prompts", "stage2_action_group_mapping.json")
        with open(prompt_map_path, 'r') as f:
            self.prompt_mapping = json.load(f)

        self.latest_motion_state = None
        self.current_prompts = []
        self.current_text_features = None

        rospy.Subscriber("/motion_state", String, self.motion_state_callback)
        rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.pub = rospy.Publisher("/action_group", String, queue_size=10)

        rospy.loginfo("[Stage2] Action Group Node started.")
        rospy.spin()

    def motion_state_callback(self, msg):
        self.latest_motion_state = msg.data.lower().strip()
        if self.latest_motion_state in self.prompt_mapping:
            self.current_prompts = self.prompt_mapping[self.latest_motion_state]
            self.current_text_features = self.encode_text(self.current_prompts)
            # rospy.loginfo(f"[Stage2] Loaded {len(self.current_prompts)} prompts for motion: {self.latest_motion_state}")
        else:
            self.current_prompts = []
            self.current_text_features = None
        
        self.need_update = True


    def encode_text(self, prompts):
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            return self.model.encode_text(text).float()

    def image_callback(self, msg):
        if not self.need_update:
            return
        self.need_update = False  # 只跑一次

        if not self.current_prompts or self.current_text_features is None:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_img = PILImage.fromarray(cv_img[..., ::-1])
            image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor).float()
                logits = image_features @ self.current_text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()

            best_idx = probs[0].argmax()
            best_action_group = self.current_prompts[best_idx]

            self.pub.publish(best_action_group)
            
            GREEN = "\033[92m"
            RESET = "\033[0m"
            # rospy.loginfo(f"[Frame {self.frame_seq}] Top predictions:")
            rospy.loginfo("Top predictions:")
            top_indices = np.argsort(probs[0])[::-1][:5]
            for rank, prompt_idx in enumerate(top_indices, start=1):
                prompt = self.current_prompts[prompt_idx]
                prob = probs[0][prompt_idx] * 100
                if prob > 80:
                    rospy.loginfo(f"  {rank}. {GREEN}{prompt}: {prob:.2f}%{RESET}")
                elif prob > 30:
                    rospy.logwarn(f"  {rank}. {prompt}: {prob:.2f}%")
                else:
                    rospy.loginfo(f"  {rank}. {prompt}: {prob:.2f}%")

        except Exception as e:
            rospy.logerr(f"[Stage2] Action group prediction failed: {e}")

if __name__ == '__main__':
    try:
        ClipActionGroupNode()
    except rospy.ROSInterruptException:
        pass
