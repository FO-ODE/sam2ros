# -*- coding: utf-8 -*-
# use sam2-env

# must run in the same terminal before running the script (done in the activate script of sam2-env)
# export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7" 

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import SAM
from pathlib import Path

from sam2ros_msgs.msg import SegmentMask

# # 系统ROS Python路径
# import sys
# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages') 


# def process_with_sam2(input_image):
#     # parameters
#     model_path = "SAM_models/sam2.1_b.pt"   # model_path = "SAM_models/mobile_sam.pt"
#     model_name = Path(model_path).stem
    
#     print(f"Using model: {model_name}")
    
#     model = SAM(model_path)
#     model.to('cuda:0')
#     # model.info()

#     segmented_image = model(input_image)
#     segmented_image = segmented_image[0].plot() # 
    
#     return segmented_image



def process_with_sam2(input_image):
    model_path = "SAM_models/sam2.1_b.pt"
    model_name = Path(model_path).stem
    model = SAM(model_path)
    model.to('cuda:0')
    print(f"Using model: {model_name}")
    
    results = model(input_image)[0]  # 拿到分割结果对象（不是 plot）
    
    masks = results.masks.data.cpu().numpy()  # shape: [N, H, W]
    
    objects = []
    for i, mask in enumerate(masks):
        # 二值掩码 -> uint8 图像
        binary_mask = (mask * 255).astype(np.uint8)
        # 裁剪
        masked_img = cv2.bitwise_and(input_image, input_image, mask=binary_mask)
        
        # 可选：找出边界框裁剪区域
        y_indices, x_indices = np.where(binary_mask > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            continue
        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()
        cropped = masked_img[y1:y2+1, x1:x2+1]

        # 添加编号和图像
        objects.append({
            'id': i + 1,
            
            'mask': binary_mask,
            'crop': cropped
        })
    
    return objects



# def display_segmented_objects(objects, win_name="Segmented Objects", max_per_row=5):
#     # 所有裁剪图像
#     crops = [obj['crop'] for obj in objects if obj['crop'].size > 0]

#     if not crops:
#         rospy.logwarn("No objects to display.")
#         return

#     # resize 所有图像为统一大小
#     thumb_size = (128, 128)
#     thumbs = [cv2.resize(crop, thumb_size) for crop in crops]
#     for i, thumb in enumerate(thumbs):
#         cv2.putText(thumb, f"ID {i+1}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

#     # 拼接为网格图像
#     rows = []
#     for i in range(0, len(thumbs), max_per_row):
#         row_imgs = thumbs[i:i + max_per_row]
#         # 若不够 max_per_row，则补空图
#         while len(row_imgs) < max_per_row:
#             row_imgs.append(np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8))
#         row = cv2.hconcat(row_imgs)
#         rows.append(row)

#     full_grid = cv2.vconcat(rows)
#     cv2.imshow(win_name, full_grid)
#     cv2.waitKey(1)


def display_segmented_objects(objects, win_name="Segmented Objects", row_max_width=800, spacing=10):
    crops = [obj['crop'] for obj in objects if obj['crop'].size > 0]

    if not crops:
        rospy.logwarn("No objects to display.")
        return

    rows = []
    current_row = []
    current_width = 0
    max_height_in_row = 0
    max_row_width = 0  # 用来记录最长行宽

    for i, crop in enumerate(crops):
        h, w = crop.shape[:2]
        if current_width + w + spacing > row_max_width and current_row:
            # 结束当前行
            row_canvas = np.zeros((max_height_in_row, current_width, 3), dtype=np.uint8)
            x_offset = 0
            for img in current_row:
                row_canvas[0:img.shape[0], x_offset:x_offset + img.shape[1]] = img
                x_offset += img.shape[1] + spacing
            rows.append(row_canvas)
            max_row_width = max(max_row_width, row_canvas.shape[1])

            # 开始新行
            current_row = [crop]
            current_width = w + spacing
            max_height_in_row = h
        else:
            current_row.append(crop)
            current_width += w + spacing
            max_height_in_row = max(max_height_in_row, h)

    # 添加最后一行
    if current_row:
        row_canvas = np.zeros((max_height_in_row, current_width, 3), dtype=np.uint8)
        x_offset = 0
        for img in current_row:
            row_canvas[0:img.shape[0], x_offset:x_offset + img.shape[1]] = img
            x_offset += img.shape[1] + spacing
        rows.append(row_canvas)
        max_row_width = max(max_row_width, row_canvas.shape[1])

    # 补齐每行的宽度（右边加黑边）
    for i in range(len(rows)):
        h, w = rows[i].shape[:2]
        if w < max_row_width:
            pad_width = max_row_width - w
            rows[i] = cv2.copyMakeBorder(rows[i], 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 拼接所有行
    full_image = cv2.vconcat(rows)
    cv2.imshow(win_name, full_image)
    cv2.waitKey(1)




class Sam2SegmentationNode:
    def __init__(self):
        rospy.init_node("sam2_segmentation_node", anonymous=True)

        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/xtion/rgb/mask_segment", Image, queue_size=1)  # for RVIZ
        self.mask_pub = rospy.Publisher("/sam2ros/mask_segment", SegmentMask, queue_size=1) # queue_size=10

        self.bridge = CvBridge()
        self.segment_counter = 0 
        
        rospy.loginfo("SAM2 segmentation node has started!")
        rospy.spin()
        
        
        



    def image_callback(self, msg):
        try:
            # input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # segmented_image = process_with_sam2(input_image)

            # # resize 到原图尺寸
            # # h1, w1 = input_image.shape[:2]
            # # h2, w2 = segmented_image.shape[:2]
            # # if (h1, w1) != (h2, w2):
            # #     segmented_image = cv2.resize(segmented_image, (w1, h1))

            # # combined_image = np.hstack((input_image, segmented_image))
            # # cv2.imshow("Original | Segmented", combined_image)
            
            # cv2.imshow("Original Image", input_image)
            # cv2.imshow("Segmented Image", segmented_image)
            # cv2.waitKey(1)

            # # 原图 segment_id=0
            # msg_original = SegmentMask()
            # msg_original.header = msg.header
            # msg_original.mask_image = self.bridge.cv2_to_imgmsg(input_image, "bgr8")
            # msg_original.segment_id = 0
            # self.mask_pub.publish(msg_original)


            # # 分割图 segment_id>0
            # self.segment_counter += 1  # 从1开始
            # msg_segmented = SegmentMask()
            # msg_segmented.header = msg.header
            # msg_segmented.mask_image = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
            # msg_segmented.segment_id = self.segment_counter
            # self.mask_pub.publish(msg_segmented)
            
            # self.image_pub.publish(msg_segmented.mask_image)
            
            input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            segments = process_with_sam2(input_image)

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
                
            display_segmented_objects(segments)








        except Exception as e:
            rospy.logerr(f"error: {e}")

if __name__ == "__main__":
    Sam2SegmentationNode()
