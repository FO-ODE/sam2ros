# -*- coding: utf-8 -*-
import os
import rospy
import cv2
import time
import threading
import numpy as np
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import SAM
from ultralytics.engine.results import Results
from pathlib import Path
from sam2ros_msgs.msg import SegmentMask


def process_with_sam(input_image, model):
    start_time = time.time()
    results = model(input_image, verbose=False)[0]
    time_used = round((time.time() - start_time) * 1000)  # ms
    seg_vis = results.plot()
    masks = results.masks.data.cpu().numpy()  # shape: (N, H, W)

    objects = []
    for i, mask in enumerate(masks):
        binary_mask = (mask > 0.5).astype(np.uint8)  # 0/1
        # binary_mask_visual = (binary_mask * 255).astype(np.uint8)  # for display/ROS

        # 生成裁剪图像
        masked_img = cv2.bitwise_and(input_image, input_image, mask=binary_mask)

        y_indices, x_indices = np.where(binary_mask > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            continue
        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()
        cropped = masked_img[y1:y2+1, x1:x2+1]

        objects.append({
            'id': i,
            'mask': binary_mask,      # 原图大小的可发布掩码,0/1
            'crop': cropped
        })

    return seg_vis, objects, time_used



class SamSegmentationNode:
    def __init__(self):
        rospy.init_node("sam_segmentation_node", anonymous=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../../SAM_models/sam2.1_b.pt")
        model_path = os.path.abspath(model_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SAM(model_path)
        self.model.to(self.device)
        self.model_name = Path(model_path).stem
        self.frame_seq = 0

        self.bridge = CvBridge()

        self.latest_image_msg = None
        self.image_lock = threading.Lock()

        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/adv_robocup/sam2clip/sam_segment_visual", Image, queue_size=10)
        self.crop_pub = rospy.Publisher("/adv_robocup/sam2clip/sam_segment_crop", SegmentMask, queue_size=50)
        self.mask_pub = rospy.Publisher("/adv_robocup/sam2clip/sam_segment_mask", SegmentMask, queue_size=50)
        self.orig_image_pub = rospy.Publisher("/adv_robocup/sam2clip/image_raw", Image, queue_size=1)

        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        rospy.loginfo(f"SAM Segmentation Node Initialized, using device: {self.device}")
        rospy.loginfo(f"Using model: {self.model_name}")
        rospy.spin()

    def image_callback(self, msg):
        with self.image_lock:
            self.latest_image_msg = msg  # 自动丢弃旧帧

    def processing_loop(self):
        rate = rospy.Rate(10)  # 控制最大处理频率
        while not rospy.is_shutdown():
            msg = None
            with self.image_lock:
                if self.latest_image_msg is not None:
                    msg = self.latest_image_msg
                    self.latest_image_msg = None  # 清空，表示已处理

            if msg is not None:
                self.process_image(msg)

            rate.sleep()

    def process_image(self, msg):
        try:
            self.orig_image_pub.publish(msg)  # 原始图像转发

            input_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            segmented_image, segments, time_used = process_with_sam(input_image, self.model)
            self.frame_seq += 1
            rospy.loginfo(f"Frame: {self.frame_seq}, Detected targets: {len(segments)}, Timecost: {time_used}[ms]")

            if segmented_image.shape[:2] != input_image.shape[:2]:
                segmented_image = cv2.resize(segmented_image, (input_image.shape[1], input_image.shape[0]))

            self.image_pub.publish(self.bridge.cv2_to_imgmsg(segmented_image, "bgr8"))


            for obj in segments:
                segment_msg = SegmentMask()
                segment_msg.header = msg.header

                # ✅ 保证是原图大小的掩码
                segment_msg.mask = self.bridge.cv2_to_imgmsg(obj['mask'], encoding="mono8")

                # ✅ 这是裁剪图像
                segment_msg.crop = self.bridge.cv2_to_imgmsg(obj['crop'], "bgr8")

                segment_msg.segment_id = obj['id']
                segment_msg.frame_seq = self.frame_seq

                self.crop_pub.publish(segment_msg)


        except Exception as e:
            rospy.logerr(f"Error in process_image: {e}")


if __name__ == "__main__":
    SamSegmentationNode()
