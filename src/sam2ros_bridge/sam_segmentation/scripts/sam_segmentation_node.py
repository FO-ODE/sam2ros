# -*- coding: utf-8 -*-
import os
import rospy
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import SAM
from ultralytics.engine.results import Results
from pathlib import Path
from sam2ros_msgs.msg import SegmentMask



def process_with_sam(input_image, model):
    
    start_time = time.time()
    results = model(input_image, verbose=False)[0] # [0]: first image processed by SAM
    # verbose=False: suppresses the output of the model in terminal
    time_used = round((time.time() - start_time) * 1000) # [ms]
    
    # print(type(results))       # <class 'ultralytics.engine.results.Results'>
    seg_vis = results.plot()
    masks = results.masks.data.cpu().numpy()  # shape: [N, H, W]


    objects = []
    for i, mask in enumerate(masks):

        binary_mask = (mask * 255).astype(np.uint8)
        masked_img = cv2.bitwise_and(input_image, input_image, mask=binary_mask)
        
        # 仅保留掩码附近区域，否则mask为原图大小
        y_indices, x_indices = np.where(binary_mask > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            continue
        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()
        cropped = masked_img[y1:y2+1, x1:x2+1]

        objects.append({
            'id': i + 1,
            'mask': binary_mask,    # binary
            'crop': cropped         # color
        })
        
        
    return seg_vis, objects, time_used



def display_with_subplots(objects, max_cols=5):
    crops = [obj['crop'] for obj in objects if obj['crop'].size > 0]
    
    num = len(crops)
    cols = min(num, max_cols)
    rows = (num + cols - 1) // cols

    plt.figure(figsize=(cols * 2, rows * 2))

    for i, crop in enumerate(crops):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"ID {i+1}")

    plt.tight_layout()
    # plt.show()
    plt.draw()
    plt.pause(1) # must have this line to update the plot



def display_segmented_objects_grid(objects, win_name="Segmented Objects", max_cols=4, spacing=10, bg_color=(0,0,0)):
    crops = [obj['crop'] for obj in objects if obj['crop'].size > 0]
    if not crops:
        rospy.logwarn("No objects to display.")
        return

    # 获取最大宽高用于对齐
    max_h = max(c.shape[0] for c in crops)
    max_w = max(c.shape[1] for c in crops)

    num = len(crops)
    cols = min(max_cols, num)
    rows = (num + cols - 1) // cols

    canvas_h = rows * max_h + (rows - 1) * spacing
    canvas_w = cols * max_w + (cols - 1) * spacing

    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    for idx, crop in enumerate(crops):
        row = idx // cols
        col = idx % cols

        x = col * (max_w + spacing)
        y = row * (max_h + spacing)

        # resize 
        ch, cw = crop.shape[:2]
        offset_y = (max_h - ch) // 2
        offset_x = (max_w - cw) // 2

        crop_annotated = crop.copy()
        cv2.rectangle(crop_annotated, (0, 0), (cw - 1, ch - 1), (0, 255, 0), 2)
        cv2.putText(crop_annotated, f"ID {idx+1}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        canvas[y + offset_y: y + offset_y + ch, x + offset_x: x + offset_x + cw] = crop_annotated

    cv2.imshow(win_name, canvas)
    cv2.waitKey(1)
    
    

class SamSegmentationNode:
    def __init__(self):
        rospy.init_node("sam_segmentation_node", anonymous=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../../SAM_models/sam2.1_b.pt")  
        model_path = os.path.abspath(model_path)
        # 'mobile_sam.pt',              'sam_b.pt',    'sam_l.pt',
        # 'sam2_t.pt',   'sam2_s.pt',   'sam2_b.pt',   'sam2_l.pt', 
        # 'sam2.1_t.pt', 'sam2.1_s.pt', 'sam2.1_b.pt', 'sam2.1_l.pt'
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.model = SAM(model_path)
        self.model.to(self.device)
        self.model_name = Path(model_path).stem
        self.frame_seq = 0

        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/adv_robocup/sam2clip/sam_segment_visual", Image, queue_size=10)  # for visualization
        self.mask_pub = rospy.Publisher("/adv_robocup/sam2clip/sam_segment_mask", SegmentMask, queue_size=50) # if queue_size=1, cannot publish all messages

        self.bridge = CvBridge()
        
        rospy.loginfo(f"SAM Segmentation Node Initialized, using device: {self.device}")
        rospy.loginfo(f"Using model: {self.model_name}")
        rospy.spin()


    def image_callback(self, msg):
        try:
            input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            segmented_image, segments, time_used = process_with_sam(input_image, self.model)
            self.frame_seq += 1
            rospy.loginfo(f"Frame: {self.frame_seq}, Detected targets: {len(segments)}, Timecost: {time_used}[ms]")

            h1, w1 = input_image.shape[:2]
            h2, w2 = segmented_image.shape[:2]
            if (h1, w1) != (h2, w2):
                segmented_image = cv2.resize(segmented_image, (w1, h1))
            

            ######################################################## 发布分割结果
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(segmented_image, "bgr8"))
            
            
            # segment_id=0, original image with mask
            msg_segmented_image = SegmentMask()
            msg_segmented_image.header = msg.header
            msg_segmented_image.mask_image = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
            msg_segmented_image.segment_id = 0
            msg_segmented_image.frame_seq = self.frame_seq
            self.mask_pub.publish(msg_segmented_image)

            # pubslish each segment
            for obj in segments:
                segment_msg = SegmentMask()
                segment_msg.header = msg.header
                segment_msg.mask_image = self.bridge.cv2_to_imgmsg(obj['crop'], "bgr8")
                segment_msg.segment_id = obj['id']
                segment_msg.frame_seq = self.frame_seq
                self.mask_pub.publish(segment_msg)



        except Exception as e:
            rospy.logerr(f"error: {e}")

if __name__ == "__main__":
    SamSegmentationNode()
