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

class ClipStage1Node:
    def __init__(self):
        rospy.init_node('clip_stage1_node', anonymous=True)

        # 加载模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "..", "CLIP_models", "ViT-L-14.pt") # ViT-L-14.pt # ViT-B-32.pt
        self.model, self.preprocess = clip.load(model_path, device=self.device)

        # 加载 prompts
        prompt_path = os.path.join(script_dir, "..", "prompts", "prompts_stage1.json")
        with open(prompt_path, 'r') as f:
            self.prompts = json.load(f)

        self.text_features = self.encode_text(self.prompts)

        # ROS订阅与发布
        self.bridge = CvBridge()
        rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.pub = rospy.Publisher("/clip_stage1_result", String, queue_size=10)

        rospy.loginfo("Clip Stage1 Node Started.")
        rospy.spin()

    def encode_text(self, prompts):
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            return self.model.encode_text(text).float()

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_img = PILImage.fromarray(cv_img[..., ::-1])
            image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor).float()
                logits = image_features @ self.text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()

            # 找到概率最高的类别
            best_idx = probs[0].argmax()
            best_action = self.prompts[best_idx]

            # 发布结果
            self.pub.publish(best_action)
            rospy.loginfo(f"[Stage1] Predicted: {best_action}")

        except Exception as e:
            rospy.logerr(f"Stage1 failed: {e}")

if __name__ == '__main__':
    try:
        ClipStage1Node()
    except rospy.ROSInterruptException:
        pass
