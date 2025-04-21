# -*- coding: utf-8 -*-
import torch
import clip
from PIL import Image
import json
import numpy as np
import os


script_dir = os.path.dirname(__file__)
json_path = os.path.join(script_dir, "prompts/clip_behavior_prompts.json")
with open(json_path, "r") as f:
    prompt_dict = json.load(f)


# 合并所有提示词
all_prompts = []
for prompts in prompt_dict.values():
    all_prompts.extend(prompts)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载图像
image = preprocess(Image.open("../test_images/workers.jpg")).unsqueeze(0).to(device)

# 编码
text = clip.tokenize(all_prompts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 输出 top-k 预测
topk = 5
top_indices = np.argsort(probs[0])[::-1][:topk]
print("\nTop Predictions:")
for i in top_indices:
    print(f"{all_prompts[i]}: {probs[0][i]:.4f}")
