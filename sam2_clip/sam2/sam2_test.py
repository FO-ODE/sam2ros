# use sam2-env
import os
import torch
from ultralytics import SAM

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

model = SAM("SAM_models/sam2.1_b.pt")
model.to('cuda:0')


for result in model.pvvredict(
        # source="test.mp4",
        source="../test_images/goods.png",
        # points=[[100, 100], [200, 200], [300, 300]],
        # bboxes=[[100, 100, 500, 500]],
        
        # labels=[1, 1, 1],
        # imgsz=1024 by default, choose 512 is faster but with bad quality
        
        stream=True,
        show=True,
        save=True):

    # print(result) 
    pass  
