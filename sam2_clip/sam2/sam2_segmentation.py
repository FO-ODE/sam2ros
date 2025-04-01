# -*- coding: utf-8 -*-
# use sam2-env

# must run in the same terminal before running the script (done in the activate script of sam2-env)
# export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7" 

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import SAM
from pathlib import Path

from sam2ros_msgs.msg import SegmentMask

# # 系统ROS Python路径
# import sys
# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages') 


def process_with_sam2(input_image, model):

    results = model(input_image)[0] # [0]: first image processed by SAM2
    seg_vis = results.plot()
    masks = results.masks.data.cpu().numpy()  # shape: [N, H, W]


    objects = []
    for i, mask in enumerate(masks):
        # 处理掩码
        binary_mask = (mask * 255).astype(np.uint8)
        masked_img = cv2.bitwise_and(input_image, input_image, mask=binary_mask)
        
        # 仅保留掩码区域，否则mask为原图大小
        y_indices, x_indices = np.where(binary_mask > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            continue
        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()
        cropped = masked_img[y1:y2+1, x1:x2+1]

        objects.append({
            'id': i + 1,
            'mask': binary_mask,
            'crop': cropped 
        })
        
        
    return seg_vis, objects



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



class Sam2SegmentationNode:
    def __init__(self):
        rospy.init_node("sam2_segmentation_node", anonymous=True)
        
        model_path = "SAM_models/sam2.1_b.pt"
        self.model = SAM(model_path)
        self.model.to('cuda:0')
        model_name = Path(model_path).stem


        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/xtion/rgb/mask_segment", Image, queue_size=1)  # for RVIZ
        self.mask_pub = rospy.Publisher("/sam2ros/mask_segment", SegmentMask, queue_size=1) # queue_size=10

        self.bridge = CvBridge()
        self.segment_counter = 0 
        
        rospy.loginfo("SAM2 segmentation node has started!")
        rospy.loginfo(f"Using model: {model_name}")
        rospy.spin()



    def image_callback(self, msg):
        try:
            input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            segmented_image, segments = process_with_sam2(input_image, self.model)
            
            # 拼接原图 + 分割图（原始逻辑）
            h1, w1 = input_image.shape[:2]
            h2, w2 = segmented_image.shape[:2]
            if (h1, w1) != (h2, w2):
                segmented_image = cv2.resize(segmented_image, (w1, h1))
            combined_image = np.hstack((input_image, segmented_image))
            cv2.imshow("Original | Segmented", combined_image)


            # 原图作为 segment_id=0
            msg_original = SegmentMask()
            msg_original.header = msg.header
            msg_original.mask_image = self.bridge.cv2_to_imgmsg(input_image, "bgr8")
            msg_original.segment_id = 0
            self.mask_pub.publish(msg_original)

            # 发布每个裁剪结果
            for obj in segments:
                segment_msg = SegmentMask()
                segment_msg.header = msg.header
                segment_msg.mask_image = self.bridge.cv2_to_imgmsg(obj['crop'], "bgr8")
                segment_msg.segment_id = obj['id']
                self.mask_pub.publish(segment_msg)

            display_with_subplots(segments)
            # display_segmented_objects_grid(segments) # faster

        except Exception as e:
            rospy.logerr(f"error: {e}")

if __name__ == "__main__":
    Sam2SegmentationNode()
