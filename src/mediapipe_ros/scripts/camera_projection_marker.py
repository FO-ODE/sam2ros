#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Point
import math

class CameraProjectionMarker:
    def __init__(self):
        rospy.init_node('camera_projection_marker', anonymous=True)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 坐标系设置
        self.camera_frame = "xtion_rgb_optical_frame"
        self.base_frame = "base_link"
        
        # 相机内参
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None
        
        # 存储稳定的圆柱体markers
        self.stable_cylinders = []
        
        # 订阅相机信息和图像
        self.camera_info_sub = rospy.Subscriber("/xtion/rgb/camera_info", CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        
        # 订阅稳定的圆柱体markers
        self.stable_marker_sub = rospy.Subscriber("/stable_arm_marker", Marker, self.stable_marker_callback)
        
        # 发布处理后的图像
        self.marked_image_pub = rospy.Publisher("/camera_projection_marked", Image, queue_size=1)
        
        rospy.loginfo("Camera Projection Marker Node initialized")
        
    def camera_info_callback(self, msg):
        """处理相机内参信息"""
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)
        self.image_size = (msg.width, msg.height)
        
        rospy.loginfo("Camera intrinsics received:")
        rospy.loginfo(f"Camera matrix:\n{self.camera_matrix}")
        rospy.loginfo(f"Image size: {self.image_size}")
        
    def stable_marker_callback(self, msg):
        """处理稳定的圆柱体marker"""
        if msg.action == Marker.ADD:
            # 添加或更新圆柱体
            # 检查是否已存在相同ID的marker
            existing_index = None
            for i, cylinder in enumerate(self.stable_cylinders):
                if cylinder['id'] == msg.id and cylinder['ns'] == msg.ns:
                    existing_index = i
                    break
            
            cylinder_data = {
                'id': msg.id,
                'ns': msg.ns,
                'pose': msg.pose,
                'scale': msg.scale,
                'frame_id': msg.header.frame_id
            }
            
            if existing_index is not None:
                self.stable_cylinders[existing_index] = cylinder_data
            else:
                self.stable_cylinders.append(cylinder_data)
                rospy.loginfo(f"Added stable cylinder {msg.ns}:{msg.id}")
                
        elif msg.action == Marker.DELETE:
            # 删除特定圆柱体
            self.stable_cylinders = [c for c in self.stable_cylinders 
                                   if not (c['id'] == msg.id and c['ns'] == msg.ns)]
            rospy.loginfo(f"Removed stable cylinder {msg.ns}:{msg.id}")
            
        elif msg.action == Marker.DELETEALL:
            # 删除所有圆柱体
            self.stable_cylinders.clear()
            rospy.loginfo("Cleared all stable cylinders")
    
    def transform_pose_to_camera_frame(self, pose, source_frame):
        """将pose从源坐标系转换到相机坐标系"""
        try:
            # 创建PoseStamped
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = source_frame
            pose_stamped.header.stamp = rospy.Time(0)  # 使用最新变换
            pose_stamped.pose = pose
            
            # 转换到相机坐标系
            transformed_pose = self.tf_buffer.transform(pose_stamped, self.camera_frame, rospy.Duration(1.0))
            return transformed_pose.pose
            
        except Exception as e:
            rospy.logwarn(f"Transform failed: {e}")
            return None
    
    def cylinder_to_points(self, pose, scale, num_points=16):
        """将圆柱体转换为3D点集合（圆柱体表面的关键点）"""
        points = []
        
        # 圆柱体的高度和半径
        height = scale.z
        radius = scale.x / 2.0  # 假设x和y尺度相同
        
        # 生成圆柱体顶部和底部的圆周点
        for level in [0, height]:  # 顶部和底部
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                x = pose.position.x + radius * math.cos(angle)
                y = pose.position.y + radius * math.sin(angle)
                z = pose.position.z - height/2 + level
                points.append([x, y, z])
        
        # 添加圆柱体中心轴上的点
        for i in range(5):  # 中心轴上的5个点
            level = height * i / 4
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z - height/2 + level
            points.append([x, y, z])
            
        return np.array(points)
    
    def project_3d_to_2d(self, points_3d):
        """将3D点投影到2D图像平面"""
        if self.camera_matrix is None:
            return None
            
        # 使用cv2.projectPoints进行投影
        # 这里假设没有旋转和平移（因为点已经在相机坐标系中）
        rvec = np.zeros((3, 1))  # 无旋转
        tvec = np.zeros((3, 1))  # 无平移
        
        try:
            points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            return points_2d.reshape(-1, 2)
        except Exception as e:
            rospy.logwarn(f"Projection failed: {e}")
            return None
    
    def draw_projection_rays(self, image, points_2d):
        """在图像上绘制从相机中心到投影点的射线"""
        if points_2d is None or len(points_2d) == 0:
            return image
        
        # 相机中心在图像平面上的投影（主点）
        camera_center = (int(self.camera_matrix[0, 2]), int(self.camera_matrix[1, 2]))
        
        # 创建图像副本
        marked_image = image.copy()
        
        # 绘制射线
        for point in points_2d:
            pt = (int(point[0]), int(point[1]))
            
            # 检查点是否在图像范围内
            if (0 <= pt[0] < self.image_size[0] and 0 <= pt[1] < self.image_size[1]):
                # 绘制从相机中心到投影点的线
                cv2.line(marked_image, camera_center, pt, (0, 255, 0), 2)
                # 绘制投影点
                cv2.circle(marked_image, pt, 5, (0, 0, 255), -1)
        
        # 绘制相机中心
        cv2.circle(marked_image, camera_center, 8, (255, 0, 0), -1)
        
        # 添加文字说明
        cv2.putText(marked_image, "Camera Center", 
                   (camera_center[0] + 10, camera_center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return marked_image
    
    def draw_cylinder_outline(self, image, points_2d):
        """在图像上绘制圆柱体的轮廓"""
        if points_2d is None or len(points_2d) == 0:
            return image
            
        marked_image = image.copy()
        
        # 过滤出在图像范围内的点
        valid_points = []
        for point in points_2d:
            pt = (int(point[0]), int(point[1]))
            if (0 <= pt[0] < self.image_size[0] and 0 <= pt[1] < self.image_size[1]):
                valid_points.append(pt)
        
        if len(valid_points) < 3:
            return marked_image
        
        # 绘制凸包轮廓
        hull_points = cv2.convexHull(np.array(valid_points))
        cv2.drawContours(marked_image, [hull_points], -1, (255, 255, 0), 3)
        
        # 绘制所有投影点
        for pt in valid_points:
            cv2.circle(marked_image, pt, 3, (0, 255, 255), -1)
            
        return marked_image
    
    def image_callback(self, msg):
        """处理接收到的图像"""
        if self.camera_matrix is None:
            rospy.logdebug("Camera matrix not available yet")
            return
            
        try:
            # 转换ROS图像到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            marked_image = cv_image.copy()
            
            # 处理每个稳定的圆柱体
            for cylinder in self.stable_cylinders:
                # 将圆柱体pose转换到相机坐标系
                camera_pose = self.transform_pose_to_camera_frame(
                    cylinder['pose'], cylinder['frame_id'])
                
                if camera_pose is None:
                    continue
                
                # 检查圆柱体是否在相机前方
                if camera_pose.position.z <= 0:
                    continue  # 圆柱体在相机后方，跳过
                
                # 生成圆柱体的3D点
                points_3d = self.cylinder_to_points(camera_pose, cylinder['scale'])
                
                # 将3D点投影到2D
                points_2d = self.project_3d_to_2d(points_3d)
                
                if points_2d is not None:
                    # 方法1：绘制投影射线
                    marked_image = self.draw_projection_rays(marked_image, points_2d)
                    
                    # 方法2：绘制圆柱体轮廓（可选，取消注释启用）
                    # marked_image = self.draw_cylinder_outline(marked_image, points_2d)
            
            # 发布标记后的图像
            marked_msg = self.bridge.cv2_to_imgmsg(marked_image, "bgr8")
            marked_msg.header = msg.header
            self.marked_image_pub.publish(marked_msg)
            
        except Exception as e:
            rospy.logerr(f"Image processing failed: {e}")
    
    def run(self):
        """运行节点"""
        rospy.loginfo("Camera Projection Marker is running...")
        rospy.loginfo("Subscribe to /camera_projection_marked to see the marked images")
        rospy.spin()

if __name__ == '__main__':
    try:
        marker = CameraProjectionMarker()
        marker.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera Projection Marker node terminated")