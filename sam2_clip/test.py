import os
import torch
from ultralytics import SAM

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

model = SAM("sam2.1_b.pt")
model.to('cuda:0')

# stream=True need 遍历
for result in model.predict(
        source="test.mp4",
        imgsz=1024,
        stream=True,
        show=True,
        save=True):
    # print(result) 
    pass  
