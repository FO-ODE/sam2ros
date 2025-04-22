# -*- coding: utf-8 -*-
import os
import time
import ros_numpy
import rospy
from std_msgs.msg import String
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


        self.det_model, self.det_model_name = load_model("yolo11m.pt")
        self.seg_model, self.seg_model_name = load_model("yolo11m-seg.pt")
        self.pos_model, self.pos_model_name = load_model("yolo11m-pose.pt")
        
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
        self.seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)
        self.pos_image_pub = rospy.Publisher("/ultralytics/pose/image", Image, queue_size=5)
        self.classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)
        self.person_crop_pub = rospy.Publisher("/ultralytics/person_crop/image", Image, queue_size=5)


        rospy.loginfo(f"Detection model: {self.det_model_name}")
        rospy.loginfo(f"Segmentation model: {self.seg_model_name}")
        rospy.loginfo(f"Pose model: {self.pos_model_name}")
        rospy.spin()
        
        
    def image_callback(self, msg):
        """Callback function to process image and publish annotated images."""
        """Detection, Segmentation, Pose"""
        input_image = ros_numpy.numpify(msg)
        if self.det_image_pub.get_num_connections():
            det_result = self.det_model(input_image)
            det_annotated = det_result[0].plot(show=False)
            self.det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))
            
            # Extract class names
            classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
            names = [det_result[0].names[i] for i in classes]
            self.classes_pub.publish(String(data=str(names)))
            
            
            # Extract person crops
            for box, cls in zip(det_result[0].boxes.xyxy.cpu().numpy(), 
                                det_result[0].boxes.cls.cpu().numpy().astype(int)):

                if det_result[0].names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box)
                    cropped_person = input_image[y1:y2, x1:x2].copy()

                    if cropped_person.size == 0:
                        continue

                    self.person_crop_pub.publish(ros_numpy.msgify(Image, cropped_person, encoding="rgb8"))


        if self.seg_image_pub.get_num_connections():
            seg_result = self.seg_model(input_image)
            seg_annotated = seg_result[0].plot(show=False)
            self.seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))
            
            
        if self.pos_image_pub.get_num_connections():
            pos_result = self.pos_model(input_image)
            pos_annotated = pos_result[0].plot(show=False)
            self.pos_image_pub.publish(ros_numpy.msgify(Image, pos_annotated, encoding="rgb8"))
            
            
            
            
if __name__ == '__main__':
    yolo_detection_node = YoloDetectionNode()