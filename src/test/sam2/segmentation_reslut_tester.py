#!/usr/bin/env python
# -*- coding: utf-8 -*-



######################################## test passed

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sam2ros_msgs.msg import SegmentMask

class MaskVisualizerNode:
    def __init__(self):
        rospy.init_node('SAM2_test_node', anonymous=True)

        self.bridge = CvBridge()
        self.current_frame_seq = None
        self.current_segments = {}

        rospy.Subscriber("/sam2ros/mask_segment", SegmentMask, self.mask_callback, queue_size=50) # if queue_size=1, cannot receive all messages
        rospy.loginfo("SAM2 testing Node started")
        self.loop()

    def mask_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg.mask_image, "bgr8")
            frame_seq = msg.frame_seq

            # 如果 frame_seq 变了，说明是新的一帧，清空旧内容
            if self.current_frame_seq != frame_seq:
                self.current_frame_seq = frame_seq
                self.current_segments.clear()

            self.current_segments[msg.segment_id] = cv_image

        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def loop(self):
        rate = rospy.Rate(5)  # 每秒刷新 5 次
        while not rospy.is_shutdown():
            self.display_segments()
            rate.sleep()

    def display_segments(self):
        if not self.current_segments:
            return

        segments = self.current_segments
        crops = [img for seg_id, img in sorted(segments.items()) if img is not None]
        ids = [seg_id for seg_id in sorted(segments.keys())]

        max_cols = 5
        cols = min(max_cols, len(crops))
        rows = (len(crops) + cols - 1) // cols

        max_h = max(img.shape[0] for img in crops)
        max_w = max(img.shape[1] for img in crops)

        spacing = 10
        canvas_h = rows * max_h + (rows - 1) * spacing
        canvas_w = cols * max_w + (cols - 1) * spacing
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for i, img in enumerate(crops):
            row = i // cols
            col = i % cols
            h, w = img.shape[:2]

            y = row * (max_h + spacing)
            x = col * (max_w + spacing)

            offset_y = (max_h - h) // 2
            offset_x = (max_w - w) // 2

            crop_display = img.copy()
            cv2.rectangle(crop_display, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)
            cv2.putText(crop_display, f"ID {ids[i]}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            canvas[y + offset_y:y + offset_y + h, x + offset_x:x + offset_x + w] = crop_display

        win_name = f"Segmented Crops (Frame seq: {self.current_frame_seq})"
        canvas_resized = cv2.resize(canvas, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow(win_name, canvas_resized)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        MaskVisualizerNode()
    except rospy.ROSInterruptException:
        pass
