#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import message_filters

class PointCloudToImageProjector:
    def __init__(self):
        rospy.init_node('pointcloud_to_image_projector', anonymous=True)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 相机内参
        self.camera_info = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 订阅相机信息
        self.camera_info_sub = rospy.Subscriber("/xtion/rgb/camera_info", 
                                              CameraInfo, self.camera_info_callback)
        
        # 订阅RGB图像和提取的点云，使用时间同步
        self.image_sub = message_filters.Subscriber("/xtion/rgb/image_raw", Image)
        self.pointcloud_sub = message_filters.Subscriber("/cylinder_internal_points", PointCloud2)
        
        # 时间同步器
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.pointcloud_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # 发布带有投影点的图像
        self.projected_image_pub = rospy.Publisher("/cylinder_points_on_image", Image, queue_size=1)
        
        # RGB和深度相机的frame_id
        self.rgb_frame_id = "xtion_rgb_optical_frame"
        self.depth_frame_id = "xtion_depth_optical_frame"  # 通常点云在深度相机坐标系中
        
        rospy.loginfo("PointCloud to Image Projector initialized")

    def camera_info_callback(self, msg):
        """处理相机内参信息"""
        self.camera_info = msg
        
        # 提取内参矩阵
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        
        # 提取畸变系数
        self.dist_coeffs = np.array(msg.D)
        
        rospy.loginfo("Camera info received:")
        rospy.loginfo(f"Image size: {msg.width} x {msg.height}")
        rospy.loginfo(f"Camera matrix:\n{self.camera_matrix}")
        rospy.loginfo(f"Distortion coefficients: {self.dist_coeffs}")

    def synchronized_callback(self, image_msg, pointcloud_msg):
        """处理同步的图像和点云数据"""
        if self.camera_matrix is None:
            rospy.logwarn("Camera info not received yet, skipping projection")
            return
        
        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # 投影点云到图像
            projected_image = self.project_pointcloud_to_image(cv_image, pointcloud_msg)
            
            # 发布结果图像
            if projected_image is not None:
                projected_msg = self.bridge.cv2_to_imgmsg(projected_image, "bgr8")
                projected_msg.header = image_msg.header
                self.projected_image_pub.publish(projected_msg)
                
        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")

    def transform_points_to_camera_frame(self, pointcloud_msg):
        """将点云转换到RGB相机坐标系"""
        try:
            # 如果点云已经在RGB相机坐标系中，直接返回
            if pointcloud_msg.header.frame_id == self.rgb_frame_id:
                return self.extract_points_from_pointcloud(pointcloud_msg)
            
            # 获取从点云坐标系到RGB相机坐标系的变换
            transform = self.tf_buffer.lookup_transform(
                self.rgb_frame_id,
                pointcloud_msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            # 提取点云中的所有点并转换
            points_3d = []
            for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                # 创建点的几何消息
                point_stamped = PointStamped()
                point_stamped.header = pointcloud_msg.header
                point_stamped.point.x = point[0]
                point_stamped.point.y = point[1]
                point_stamped.point.z = point[2]
                
                # 应用变换
                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                
                points_3d.append([
                    transformed_point.point.x,
                    transformed_point.point.y,
                    transformed_point.point.z
                ])
            
            return np.array(points_3d)
            
        except Exception as e:
            rospy.logerr(f"Failed to transform points to camera frame: {e}")
            return None

    def extract_points_from_pointcloud(self, pointcloud_msg):
        """从点云消息中提取3D点"""
        points_3d = []
        for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points_3d.append([point[0], point[1], point[2]])
        return np.array(points_3d)

    def project_3d_to_2d(self, points_3d):
        """将3D点投影到2D图像平面"""
        if len(points_3d) == 0:
            return np.array([])
        
        # 过滤掉在相机后方的点（z <= 0）
        valid_points = points_3d[points_3d[:, 2] > 0]
        
        if len(valid_points) == 0:
            rospy.logwarn("No valid points (all points behind camera)")
            return np.array([])
        
        # 使用OpenCV进行投影
        # 注意：这里假设没有旋转和平移（点已经在相机坐标系中）
        rvec = np.zeros(3)  # 无旋转
        tvec = np.zeros(3)  # 无平移
        
        # 投影3D点到2D
        image_points, _ = cv2.projectPoints(
            valid_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        # 重塑结果
        image_points = image_points.reshape(-1, 2)
        
        return image_points, valid_points

    def project_pointcloud_to_image(self, cv_image, pointcloud_msg):
        """将点云投影到图像上并可视化"""
        try:
            # 转换点云到相机坐标系
            points_3d = self.transform_points_to_camera_frame(pointcloud_msg)
            
            if points_3d is None or len(points_3d) == 0:
                rospy.logwarn("No valid 3D points to project")
                return cv_image
            
            rospy.loginfo(f"Projecting {len(points_3d)} 3D points to image")
            
            # 投影到2D
            projection_result = self.project_3d_to_2d(points_3d)
            
            if len(projection_result) == 0:
                rospy.logwarn("No points projected to image")
                return cv_image
            
            image_points, valid_3d_points = projection_result
            
            # 获取图像尺寸
            height, width = cv_image.shape[:2]
            
            # 在图像上绘制投影点
            result_image = cv_image.copy()
            valid_projections = 0
            
            for i, (u, v) in enumerate(image_points):
                # 检查点是否在图像范围内
                if 0 <= u < width and 0 <= v < height:
                    # 根据深度设置颜色（近的点为红色，远的点为蓝色）
                    depth = valid_3d_points[i][2]
                    
                    # 颜色映射：深度范围通常是0.5-5米
                    color_ratio = min(max((depth - 0.5) / 4.5, 0), 1)
                    color = (
                        int(255 * (1 - color_ratio)),  # B
                        int(128 * color_ratio),        # G  
                        int(255 * color_ratio)         # R
                    )
                    
                    # 绘制圆点
                    cv2.circle(result_image, (int(u), int(v)), 3, color, -1)
                    
                    # 可选：添加深度标注
                    if valid_projections % 10 == 0:  # 每10个点标注一次深度
                        cv2.putText(result_image, f"{depth:.2f}m", 
                                  (int(u) + 5, int(v) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    
                    valid_projections += 1
            
            rospy.loginfo(f"Successfully projected {valid_projections} points to image")
            
            # 添加信息文本
            info_text = f"Cylinder Points: {valid_projections}/{len(points_3d)}"
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return result_image
            
        except Exception as e:
            rospy.logerr(f"Error projecting pointcloud to image: {e}")
            return cv_image

    def get_pixel_coordinates_for_points(self, points_3d):
        """获取3D点对应的像素坐标（用于外部调用）"""
        """
        Args:
            points_3d: numpy array of shape (N, 3) containing 3D points
            
        Returns:
            list of tuples: [(u1, v1, depth1), (u2, v2, depth2), ...] 
                           pixel coordinates and depths for valid projections
        """
        if self.camera_matrix is None:
            rospy.logerr("Camera info not available")
            return []
        
        try:
            projection_result = self.project_3d_to_2d(points_3d)
            
            if len(projection_result) == 0:
                return []
            
            image_points, valid_3d_points = projection_result
            
            # 获取图像尺寸（如果有相机信息）
            if self.camera_info:
                width = self.camera_info.width
                height = self.camera_info.height
            else:
                # 默认图像尺寸
                width, height = 640, 480
            
            pixel_coordinates = []
            for i, (u, v) in enumerate(image_points):
                # 检查点是否在图像范围内
                if 0 <= u < width and 0 <= v < height:
                    depth = valid_3d_points[i][2]
                    pixel_coordinates.append((int(u), int(v), depth))
            
            return pixel_coordinates
            
        except Exception as e:
            rospy.logerr(f"Error getting pixel coordinates: {e}")
            return []

    def run(self):
        """运行节点"""
        rospy.loginfo("PointCloud to Image Projector is running...")
        rospy.loginfo("Waiting for:")
        rospy.loginfo("  - Camera info on /xtion/rgb/camera_info")
        rospy.loginfo("  - RGB images on /xtion/rgb/image_raw")
        rospy.loginfo("  - Point cloud on /cylinder_internal_points")
        rospy.loginfo("Publishing projected image on /cylinder_points_on_image")
        rospy.spin()

if __name__ == '__main__':
    try:
        projector = PointCloudToImageProjector()
        projector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("PointCloud to Image Projector node terminated")