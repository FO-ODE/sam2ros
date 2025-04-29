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

class ClipStage2Node:
    def __init__(self):
        rospy.init_node('clip_stage2_node', anonymous=True)

        # 加载模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "..", "CLIP_models", "ViT-L-14.pt") # ViT-L-14.pt # ViT-B-32.pt
        self.model, self.preprocess = clip.load(model_path, device=self.device)

        # 加载二阶段 prompts 映射
        mapping_path = os.path.join(script_dir, "..", "prompts","prompts_stage2.json")
        with open(mapping_path, 'r') as f:
            self.stage2_prompts_mapping = json.load(f)

        self.current_prompts = []
        self.current_text_features = None

        # ROS订阅与发布
        self.bridge = CvBridge()
        rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        rospy.Subscriber("/clip_stage1_result", String, self.stage1_callback, queue_size=1)
        self.pub = rospy.Publisher("/clip_stage2_result", String, queue_size=10)

        self.latest_stage1_action = None

        rospy.loginfo("Clip Stage2 Node Started.")
        rospy.spin()

    def encode_text(self, prompts):
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)
            return self.model.encode_text(text).float()

    def stage1_callback(self, msg):
        action = msg.data
        self.latest_stage1_action = action
        rospy.loginfo(f"[Stage2] Received Stage1 Result: {action}")

        # 更新对应的二阶段prompts
        key = action.lower().replace("a person ", "").strip()
        if key in self.stage2_prompts_mapping:
            self.current_prompts = self.stage2_prompts_mapping[key]
            self.current_text_features = self.encode_text(self.current_prompts)
            rospy.loginfo(f"[Stage2] Loaded {len(self.current_prompts)} fine-grained prompts.")
        else:
            self.current_prompts = []
            self.current_text_features = None

    def image_callback(self, msg):
        if not self.current_prompts or self.current_text_features is None:
            return  # 没有细粒度prompt，不推理

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_img = PILImage.fromarray(cv_img[..., ::-1])
            image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor).float()
                logits = image_features @ self.current_text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()

            # 找到概率最高的细粒度动作
            best_idx = probs[0].argmax()
            best_fine_action = self.current_prompts[best_idx]

            self.pub.publish(best_fine_action)
            rospy.loginfo(f"[Stage2] Predicted Fine Action: {best_fine_action}")

        except Exception as e:
            rospy.logerr(f"Stage2 failed: {e}")

if __name__ == '__main__':
    try:
        ClipStage2Node()
    except rospy.ROSInterruptException:
        pass
