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
import time
import numpy as np

class ClipMotionStateNode:
    def __init__(self):
        rospy.init_node('clip_motion_state_node', anonymous=True)

        self.detection_rate = rospy.get_param("~detection_rate", 2.0)  # 默认1Hz
        self.last_run_time = 0.0


        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "..", "CLIP_models", "ViT-L-14.pt") # ViT-L-14.pt # ViT-B-32.pt
        self.model, self.preprocess = clip.load(model_path, device=self.device)


        script_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(script_dir, "..", "prompts", "stage1_motion_prompt.json")
        with open(prompt_path, 'r') as f:
            self.prompts = json.load(f)

        self.text_features = self.encode_text(self.prompts)

        rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        # rospy.Subscriber("/ultralytics/person_crop/image", Image, self.image_callback, queue_size=1)
        self.pub = rospy.Publisher("/motion_state", String, queue_size=10)

        # self.frame_seq = 0

        rospy.loginfo("[Stage1] Motion State Node started with detection rate: {:.2f} Hz".format(self.detection_rate))
        rospy.spin()

    def encode_text(self, prompts):
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            return self.model.encode_text(text).float()

    def image_callback(self, msg):
        now = time.time()
        if now - self.last_run_time < 1.0 / self.detection_rate:
            return
        self.last_run_time = now

        try:
            # self.frame_seq += 1
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_img = PILImage.fromarray(cv_img[..., ::-1])
            image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor).float()
                logits = image_features @ self.text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()

            best_idx = probs[0].argmax()
            best_label = self.prompts[best_idx]

            self.pub.publish(best_label)

            GREEN = "\033[92m"
            RESET = "\033[0m"
            # rospy.loginfo(f"[Frame {self.frame_seq}] Top predictions:")
            rospy.loginfo("Top predictions:")
            top_indices = np.argsort(probs[0])[::-1][:5]
            for rank, prompt_idx in enumerate(top_indices, start=1):
                prompt = self.prompts[prompt_idx]
                prob = probs[0][prompt_idx] * 100
                if prob > 80:
                    rospy.loginfo(f"  {rank}. {GREEN}{prompt}: {prob:.2f}%{RESET}")
                elif prob > 30:
                    rospy.logwarn(f"  {rank}. {prompt}: {prob:.2f}%")
                else:
                    rospy.loginfo(f"  {rank}. {prompt}: {prob:.2f}%")

        except Exception as e:
            rospy.logerr(f"[Stage1] CLIP motion state failed: {e}")

if __name__ == '__main__':
    try:
        ClipMotionStateNode()
    except rospy.ROSInterruptException:
        pass
