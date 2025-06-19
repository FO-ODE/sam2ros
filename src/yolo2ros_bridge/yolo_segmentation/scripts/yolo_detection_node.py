# -*- coding: utf-8 -*-
import os
import time
import ros_numpy
import rospy
from std_msgs.msg import String, Header
from pathlib import Path
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

        # Load the tracker configuration
        self.tracker_config = os.path.join(model_dir, "botsort.yaml")  # or "sort.yaml"
        
        self.det_model, self.det_model_name = load_model("yolo11m.pt")
        self.seg_model, self.seg_model_name = load_model("yolo11m-seg.pt")
        self.pos_model, self.pos_model_name = load_model("yolo11m-pose.pt")
        
        # 帧跳跃控制变量
        self.frame_counter = 0
        self.det_skip_frames = 2  # 检测每x帧运行一次
        self.seg_skip_frames = 2  # 分割每x帧运行一次  
        self.pos_skip_frames = 2  # 姿态每x帧运行一次

        # 上一次处理的时间戳，用于性能监控
        self.last_det_time = 0
        self.last_seg_time = 0
        self.last_pos_time = 0
        
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback, queue_size=1)  # 减小队列大小
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=1)
        self.seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=1)
        self.pos_image_pub = rospy.Publisher("/ultralytics/pose/image", Image, queue_size=1)
        self.gesture_pub = rospy.Publisher("/ultralytics/pose/gesture", String, queue_size=5)
        self.classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)
        self.person_crop_pub = rospy.Publisher("/ultralytics/person_crop/image", Image, queue_size=1)

        rospy.loginfo(f"Detection model: {self.det_model_name}")
        rospy.loginfo(f"Segmentation model: {self.seg_model_name}")
        rospy.loginfo(f"Pose model: {self.pos_model_name}")
        rospy.loginfo(f"Frame skipping - Det:{self.det_skip_frames}, Seg:{self.seg_skip_frames}, Pos:{self.pos_skip_frames}")
        rospy.spin()
        
    def detect_waving_gesture(self, keypoints, person_id):
        """
        检测是否有挥手动作（手臂举过头顶）
        YOLO pose keypoints索引：
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        """
        try:
            # 获取关键点坐标 (x, y, confidence)
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            
            # 检查关键点是否可见（confidence > 0.5）
            if (nose[2] > 0.5 and left_shoulder[2] > 0.5 and 
                right_shoulder[2] > 0.5 and left_wrist[2] > 0.5 and right_wrist[2] > 0.5):
                
                # 计算头部位置（鼻子的y坐标）
                head_y = nose[1]
                
                # 检查左手腕是否举过头顶
                left_wrist_above_head = left_wrist[1] < head_y
                
                # 检查右手腕是否举过头顶
                right_wrist_above_head = right_wrist[1] < head_y
                
                # 如果任意一只手举过头顶，则认为是挥手动作
                if left_wrist_above_head or right_wrist_above_head:
                    gesture_msg = String()
                    gesture_msg.data = f"person_id:{person_id}, action:waving"
                    self.gesture_pub.publish(gesture_msg)
                    rospy.loginfo(f"Detected waving gesture from person {person_id}")
                    return True
                    
        except Exception as e:
            rospy.logwarn(f"Error detecting gesture for person {person_id}: {e}")
            
        return False

    def image_callback(self, msg):
        """Callback function to process image and publish annotated images."""
        """Detection, Segmentation, Pose"""
        
        # 增加帧计数器
        self.frame_counter += 1
        current_time = time.time()
        
        input_image = ros_numpy.numpify(msg)
        
        # 检测模块 - 每det_skip_frames帧运行一次
        if (self.det_image_pub.get_num_connections() or self.person_crop_pub.get_num_connections()) and \
           (self.frame_counter % self.det_skip_frames == 0):
            
            start_time = time.time()
            
            # Perform detection
            det_result = self.det_model.track(
                source=input_image,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
            det_annotated = det_result[0].plot(show=False)
            
            if self.det_image_pub.get_num_connections():
                self.det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))
            
            # Extract class names
            if len(det_result[0].boxes) > 0:
                classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
                names = [det_result[0].names[i] for i in classes]
                self.classes_pub.publish(String(data=str(names)))

                # Extract IDs
                if det_result[0].boxes.id is not None:
                    ids = det_result[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = [-1] * len(det_result[0].boxes.xyxy)

                # Extract person crops
                if self.person_crop_pub.get_num_connections():
                    for box, cls, track_id in zip(det_result[0].boxes.xyxy.cpu().numpy(), 
                                        classes,
                                        ids):
                        if det_result[0].names[cls] == "person":
                            x1, y1, x2, y2 = map(int, box)
                            cropped_person = input_image[y1:y2, x1:x2].copy()
                            if cropped_person.size > 0:
                                self.person_crop_pub.publish(ros_numpy.msgify(Image, cropped_person, encoding="rgb8"))
                                break  # 只发布第一个person crop
            
            self.last_det_time = time.time() - start_time
            rospy.logdebug(f"Detection processing time: {self.last_det_time:.3f}s")

        # 分割模块 - 每seg_skip_frames帧运行一次
        if self.seg_image_pub.get_num_connections() and (self.frame_counter % self.seg_skip_frames == 0):
            start_time = time.time()
            
            seg_result = self.seg_model(input_image)
            seg_annotated = seg_result[0].plot(show=False)
            self.seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))
            
            self.last_seg_time = time.time() - start_time
            rospy.logdebug(f"Segmentation processing time: {self.last_seg_time:.3f}s")
            
        # 姿态检测模块 - 每pos_skip_frames帧运行一次
        if (self.pos_image_pub.get_num_connections() or self.gesture_pub.get_num_connections()) and \
           (self.frame_counter % self.pos_skip_frames == 0):
            
            start_time = time.time()
            
            # 对pose model也应用tracking
            pos_result = self.pos_model.track(
                source=input_image,
                persist=True,
                tracker=self.tracker_config,
                verbose=False
            )
            
            # 直接使用YOLO自带的绘制，它会显示 "id:1 person 0.93" 格式
            pos_annotated = pos_result[0].plot(show=False)
            
            # 检测挥手手势
            if pos_result[0].keypoints is not None and pos_result[0].boxes is not None and len(pos_result[0].boxes) > 0:
                keypoints = pos_result[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
                confidences = pos_result[0].keypoints.conf.cpu().numpy()  # (N, 17)
                
                # 获取person的track IDs
                if pos_result[0].boxes.id is not None:
                    ids = pos_result[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = list(range(len(keypoints)))  # 如果没有ID，使用索引
                
                # 对每个检测到的person进行手势检测
                for i, (kpts, confs, person_id) in enumerate(zip(keypoints, confidences, ids)):
                    # 合并坐标和置信度 (17, 3) - (x, y, confidence)
                    kpts_with_conf = []
                    for j in range(len(kpts)):
                        kpts_with_conf.append([kpts[j][0], kpts[j][1], confs[j]])
                    
                    self.detect_waving_gesture(kpts_with_conf, person_id)
            
            if self.pos_image_pub.get_num_connections():
                self.pos_image_pub.publish(ros_numpy.msgify(Image, pos_annotated, encoding="rgb8"))
            
            self.last_pos_time = time.time() - start_time
            rospy.logdebug(f"Pose processing time: {self.last_pos_time:.3f}s")
        
        # 每100帧打印一次性能统计
        if self.frame_counter % 100 == 0:
            rospy.loginfo(f"Frame {self.frame_counter} - Processing times: Det:{self.last_det_time:.3f}s, Seg:{self.last_seg_time:.3f}s, Pos:{self.last_pos_time:.3f}s")


if __name__ == '__main__':
    yolo_detection_node = YoloDetectionNode()