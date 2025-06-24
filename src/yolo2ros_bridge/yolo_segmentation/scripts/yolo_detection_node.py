# -*- coding: utf-8 -*-
import os
import time
import ros_numpy
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from ultralytics import YOLO


class YoloDetectionNode:
    def __init__(self):
        rospy.init_node('yolo_detection_node', anonymous=True)
        rospy.loginfo("Yolo Detection Node Initialized")

        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../YOLO_models"))

        def load_model(name):
            path = os.path.join(model_dir, name)
            return YOLO(path).to("cuda"), os.path.splitext(name)[0]

        self.tracker_config = os.path.join(model_dir, "botsort.yaml")

        self.det_model, self.det_model_name = load_model("yolo11m.pt")
        self.seg_model, self.seg_model_name = load_model("yolo11m-seg.pt")
        self.pos_model, self.pos_model_name = load_model("yolo11m-pose.pt")

        self.frame_counter = 0
        self.det_skip_frames = 2
        self.seg_skip_frames = 2
        self.pos_skip_frames = 2

        self.last_gesture_pub_time = time.time()
        
        self.latest_depth_image = None
        self.depth_sub = rospy.Subscriber("/xtion/depth/image_raw", Image, self.depth_callback, queue_size=1)
        self.depth_bbox_pub = rospy.Publisher("/ultralytics/pose/selected_person/depth_bbox", Image, queue_size=1)


        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=1)
        self.seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=1)
        self.pos_image_pub = rospy.Publisher("/ultralytics/pose/image", Image, queue_size=1)
        self.gesture_pub = rospy.Publisher("/ultralytics/pose/gesture", String, queue_size=5)
        self.classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)
        self.person_crop_pub = rospy.Publisher("/ultralytics/person_crop/image", Image, queue_size=1)
        self.selected_person_pub = rospy.Publisher("/ultralytics/pose/selected_person", Image, queue_size=1)

        rospy.loginfo(f"Detection model: {self.det_model_name}")
        rospy.loginfo(f"Segmentation model: {self.seg_model_name}")
        rospy.loginfo(f"Pose model: {self.pos_model_name}")
        rospy.loginfo(f"Frame skipping - Det:{self.det_skip_frames}, Seg:{self.seg_skip_frames}, Pos:{self.pos_skip_frames}")
        rospy.spin()

    def detect_waving_gesture(self, keypoints, person_id):
        try:
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]

            if (nose[2] > 0.5 and left_shoulder[2] > 0.5 and 
                right_shoulder[2] > 0.5 and left_wrist[2] > 0.5 and right_wrist[2] > 0.5):

                head_y = nose[1]
                left_wrist_above_head = left_wrist[1] < head_y
                right_wrist_above_head = right_wrist[1] < head_y

                if left_wrist_above_head or right_wrist_above_head:
                    return True

        except Exception as e:
            rospy.logwarn(f"Error detecting gesture for person {person_id}: {e}")

        return False
    
    def depth_callback(self, msg):
        try:
            self.latest_depth_image = ros_numpy.numpify(msg)
        except Exception as e:
            rospy.logwarn(f"Failed to receive depth image: {e}")

    def image_callback(self, msg):
        self.frame_counter += 1
        input_image = ros_numpy.numpify(msg)

        if (self.det_image_pub.get_num_connections() or self.person_crop_pub.get_num_connections()) and \
           (self.frame_counter % self.det_skip_frames == 0):

            det_result = self.det_model.track(
                source=input_image,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
            det_annotated = det_result[0].plot(show=False)

            if self.det_image_pub.get_num_connections():
                self.det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

            if len(det_result[0].boxes) > 0:
                classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
                names = [det_result[0].names[i] for i in classes]
                self.classes_pub.publish(String(data=str(names)))

                if det_result[0].boxes.id is not None:
                    ids = det_result[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = [-1] * len(det_result[0].boxes.xyxy)

                if self.person_crop_pub.get_num_connections():
                    for box, cls, track_id in zip(det_result[0].boxes.xyxy.cpu().numpy(), 
                                                  classes, ids):
                        if det_result[0].names[cls] == "person":
                            x1, y1, x2, y2 = map(int, box)
                            cropped_person = input_image[y1:y2, x1:x2].copy()
                            if cropped_person.size > 0:
                                self.person_crop_pub.publish(ros_numpy.msgify(Image, cropped_person, encoding="rgb8"))
                                break

        if self.seg_image_pub.get_num_connections() and (self.frame_counter % self.seg_skip_frames == 0):
            seg_result = self.seg_model(input_image, verbose=False)
            seg_annotated = seg_result[0].plot(show=False)
            self.seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))

        if (self.pos_image_pub.get_num_connections() or self.gesture_pub.get_num_connections()) and \
           (self.frame_counter % self.pos_skip_frames == 0):

            pos_result = self.pos_model.track(
                source=input_image,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
            pos_annotated = pos_result[0].plot(show=False)

            waving_ids = []

            if pos_result[0].keypoints is not None and pos_result[0].boxes is not None and len(pos_result[0].boxes) > 0:
                keypoints = pos_result[0].keypoints.xy.cpu().numpy()
                confidences = pos_result[0].keypoints.conf.cpu().numpy()

                if pos_result[0].boxes.id is not None:
                    ids = pos_result[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = list(range(len(keypoints)))

                for i, (kpts, confs, person_id) in enumerate(zip(keypoints, confidences, ids)):
                    kpts_with_conf = [[k[0], k[1], c] for k, c in zip(kpts, confs)]
                    if self.detect_waving_gesture(kpts_with_conf, person_id):
                        waving_ids.append(person_id)

            current_time = time.time()
            if waving_ids and (current_time - self.last_gesture_pub_time > 1.0):
                ids_str = ", ".join([f"person {pid}" for pid in waving_ids])
                gesture_msg = String(data=f"Detected waving gesture from {ids_str}")
                self.gesture_pub.publish(gesture_msg)
                rospy.loginfo(gesture_msg.data)
                self.last_gesture_pub_time = current_time

            if self.pos_image_pub.get_num_connections():
                self.pos_image_pub.publish(ros_numpy.msgify(Image, pos_annotated, encoding="rgb8"))
                
            if waving_ids:
                waving_ids.sort()
                selected_id = waving_ids[0]

                # === 每秒发布一次手势信息 ===
                if current_time - self.last_gesture_pub_time > 1.0:
                    ids_str = ", ".join([f"person {pid}" for pid in waving_ids])
                    gesture_msg = String(data=f"Detected waving gesture from {ids_str}")
                    self.gesture_pub.publish(gesture_msg)
                    rospy.loginfo(gesture_msg.data)
                    self.last_gesture_pub_time = current_time

                # === 从姿态识别结果中找到 selected_id 对应的 box ===
                if pos_result[0].boxes.id is not None:
                    for box, pid in zip(pos_result[0].boxes.xyxy.cpu().numpy(),
                                        pos_result[0].boxes.id.cpu().numpy().astype(int)):
                        if pid == selected_id:
                            x1, y1, x2, y2 = map(int, box)
                            person_crop = input_image[y1:y2, x1:x2].copy()
                            if person_crop.size == 0:
                                rospy.logwarn("Selected person crop is empty")
                                break

                            # === 将裁剪图像送入 segmentation 模型 ===
                            seg_result = self.seg_model(person_crop, verbose=False)
                            seg_annotated = seg_result[0].plot(show=False)

                            # === 发布结果图像 ===
                            self.selected_person_pub.publish(
                                ros_numpy.msgify(Image, seg_annotated, encoding="rgb8")
                            )
                            break
                
                # === 发布对应深度图的 bounding box ===
                if self.latest_depth_image is not None:
                    import cv2
                    import numpy as np

                    # 复制深度图为可写 RGB 格式（转换为可视化用）
                    depth_vis = self.latest_depth_image.copy()
                    if len(depth_vis.shape) == 2:  # 单通道 -> RGB
                        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                    # 画出 bounding box
                    cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # 发布
                    depth_image_msg = ros_numpy.msgify(Image, depth_vis, encoding="rgb8")
                    self.depth_bbox_pub.publish(depth_image_msg)

if __name__ == '__main__':
    yolo_detection_node = YoloDetectionNode()
