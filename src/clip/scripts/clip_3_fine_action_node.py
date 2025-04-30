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

class ClipFineActionNode:
    def __init__(self):
        rospy.init_node('clip_fine_action_node', anonymous=True)

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "..", "CLIP_models", "ViT-L-14.pt") # ViT-L-14.pt # ViT-B-32.pt
        self.model, self.preprocess = clip.load(model_path, device=self.device)


        script_dir = os.path.dirname(__file__)
        prompt_map_path = os.path.join(script_dir, "..", "prompts", "stage3_fine_action_mapping.json")
        with open(prompt_map_path, 'r') as f:
            self.prompt_mapping = json.load(f)

        self.latest_action_group = None
        self.current_prompts = []
        self.current_text_features = None

        rospy.Subscriber("/action_group", String, self.action_group_callback)
        rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.pub = rospy.Publisher("/fine_behavior_result", String, queue_size=10)

        rospy.loginfo("[Stage3] Fine Action Node started.")
        rospy.spin()

    def action_group_callback(self, msg):
        group = msg.data.lower().strip()
        key = group.replace("a person is ", "").replace("a person ", "").strip()
        self.latest_action_group = key
        
        if key in self.prompt_mapping:
            self.current_prompts = self.prompt_mapping[key]
            self.current_text_features = self.encode_text(self.current_prompts)
            rospy.loginfo(f"[Stage3] Loaded {len(self.current_prompts)} prompts for action group: {key}")
        else:
            self.current_prompts = []
            self.current_text_features = None

    def encode_text(self, prompts):
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            return self.model.encode_text(text).float()

    def image_callback(self, msg):
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
            best_fine_action = self.current_prompts[best_idx]

            self.pub.publish(best_fine_action)
            rospy.loginfo(f"[Stage3] Predicted Fine Action: {best_fine_action}")

        except Exception as e:
            rospy.logerr(f"[Stage3] Fine action prediction failed: {e}")

if __name__ == '__main__':
    try:
        ClipFineActionNode()
    except rospy.ROSInterruptException:
        pass
