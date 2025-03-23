import os
import torch
from ultralytics import SAM

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

model = SAM("SAM_models/sam2.1_l.pt")
model.to('cuda:0')

# stream=True need 遍历
for result in model.predict(
        # source="test.mp4",
        source="../test_images/goods.png",
        # points=[[100, 100], [200, 200], [300, 300]],
        # labels=[1, 1, 1],
        imgsz=1024,
        stream=True,
        show=True,
        save=True):
    # print(result) 
    pass  
