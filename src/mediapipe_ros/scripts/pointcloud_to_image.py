#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from std_msgs.msg import Header

class PointCloudToPixelMapper:
    def __init__(self):
        rospy.init_node('pointcloud_to_pixel_mapper', anonymous=True)
        
        # 初始化cv_bridge
        self.bridge = CvBridge()
        
        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 相机内参
        self.camera_info = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 缓存最新的图像和点云
        self.latest_image = None
        self.latest_pointcloud = None
        
        # 订阅相机内参
        self.camera_info_sub = rospy.Subscriber("/xtion/rgb/camera_info", 
                                              CameraInfo, self.camera_info_callback)
        
        # 订阅RGB图像
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", 
                                        Image, self.image_callback)
        
        # 订阅圆柱体内部点云
        self.pointcloud_sub = rospy.Subscriber("/cylinder_internal_points", 
                                             PointCloud2, self.pointcloud_callback)
        
        # 发布带有投影点的图像
        self.projected_image_pub = rospy.Publisher("/cylinder_points_projected_image", 
                                                 Image, queue_size=10)
        
        # 发布像素坐标标记
        self.pixel_markers_pub = rospy.Publisher("/cylinder_pixel_markers", 
                                                MarkerArray, queue_size=10)
        
        rospy.loginfo("PointCloud to Pixel Mapper initialized")

    def camera_info_callback(self, msg):
        """处理相机内参信息"""
        self.camera_info = msg
        
        # 提取相机内参矩阵
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        
        # 提取畸变参数
        self.dist_coeffs = np.array(msg.D)
        
        rospy.loginfo("Camera intrinsics received:")
        rospy.loginfo(f"Camera matrix:\n{self.camera_matrix}")
        rospy.loginfo(f"Distortion coefficients: {self.dist_coeffs}")
        
        # 只需要接收一次相机内参
        self.camera_info_sub.unregister()

    def image_callback(self, msg):
        """处理RGB图像"""
        try:
            # 将ROS图像转换为OpenCV格式
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 如果有点云数据，进行投影
            if self.latest_pointcloud is not None and self.camera_matrix is not None:
                self.project_points_to_image()
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def pointcloud_callback(self, msg):
        """处理点云数据"""
        self.latest_pointcloud = msg
        
        # 如果有图像数据，进行投影
        if self.latest_image is not None and self.camera_matrix is not None:
            self.project_points_to_image()

    def project_points_to_image(self):
        """将3D点云投影到2D图像平面"""
        if (self.latest_pointcloud is None or self.latest_image is None or 
            self.camera_matrix is None):
            return
        
        try:
            # 提取点云中的3D坐标
            points_3d = []
            for point in pc2.read_points(self.latest_pointcloud, 
                                       field_names=("x", "y", "z"), 
                                       skip_nans=True):
                points_3d.append([point[0], point[1], point[2]])
            
            if not points_3d:
                rospy.logwarn("No valid 3D points found in pointcloud")
                return
            
            points_3d = np.array(points_3d, dtype=np.float32)
            rospy.loginfo(f"Processing {len(points_3d)} 3D points")
            
            # 使用相机内参将3D点投影到2D像素坐标
            # 注意：这里假设点云已经在相机坐标系中（xtion_rgb_optical_frame）
            pixels_2d = self.project_3d_to_2d(points_3d)
            
            if pixels_2d is not None and len(pixels_2d) > 0:
                # 在图像上绘制投影点
                projected_image = self.draw_projected_points(self.latest_image.copy(), pixels_2d)
                
                # 发布带有投影点的图像
                self.publish_projected_image(projected_image)
                
                # 发布像素坐标信息
                self.publish_pixel_coordinates(pixels_2d)
                
                rospy.loginfo(f"Successfully projected {len(pixels_2d)} points to image")
            else:
                rospy.logwarn("No valid 2D projections found")
                
        except Exception as e:
            rospy.logerr(f"Error in point projection: {e}")

    def project_3d_to_2d(self, points_3d):
        """使用相机内参将3D点投影到2D像素坐标"""
        try:
            # 过滤掉z坐标小于等于0的点（在相机后面的点）
            valid_indices = points_3d[:, 2] > 0.01  # 至少1cm距离
            if not np.any(valid_indices):
                rospy.logwarn("No points with positive Z coordinate found")
                return None
            
            valid_points_3d = points_3d[valid_indices]
            
            # 使用cv2.projectPoints进行投影
            # 注意：这里假设没有旋转和平移（因为点云已经在相机坐标系中）
            rvec = np.zeros((3, 1), dtype=np.float32)  # 无旋转
            tvec = np.zeros((3, 1), dtype=np.float32)  # 无平移
            
            pixels_2d, _ = cv2.projectPoints(valid_points_3d, 
                                           rvec, tvec, 
                                           self.camera_matrix, 
                                           self.dist_coeffs)
            
            # 重新整形为2D数组
            pixels_2d = pixels_2d.reshape(-1, 2)
            
            # 过滤掉超出图像边界的像素点
            image_height, image_width = self.latest_image.shape[:2]
            valid_pixels = []
            
            for i, pixel in enumerate(pixels_2d):
                u, v = int(pixel[0]), int(pixel[1])
                if 0 <= u < image_width and 0 <= v < image_height:
                    valid_pixels.append({
                        'pixel': (u, v),
                        '3d_point': valid_points_3d[i],
                        'depth': valid_points_3d[i][2]
                    })
            
            rospy.loginfo(f"Filtered to {len(valid_pixels)} valid pixels within image bounds")
            return valid_pixels
            
        except Exception as e:
            rospy.logerr(f"Error in 3D to 2D projection: {e}")
            return None

    def draw_projected_points(self, image, projected_points):
        """在图像上绘制投影的点"""
        try:
            for point_info in projected_points:
                u, v = point_info['pixel']
                depth = point_info['depth']
                
                # 根据深度设置颜色（近的点为红色，远的点为蓝色）
                # 深度范围通常在0.5m到5m之间
                normalized_depth = np.clip((depth - 0.5) / 4.5, 0, 1)
                color_b = int(255 * normalized_depth)
                color_r = int(255 * (1 - normalized_depth))
                color = (color_b, 0, color_r)  # BGR格式
                
                # 绘制圆点
                cv2.circle(image, (u, v), 3, color, -1)
                
                # 可选：绘制深度文本
                if len(projected_points) < 50:  # 只在点数较少时显示文本
                    cv2.putText(image, f"{depth:.2f}m", (u+5, v-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            return image
            
        except Exception as e:
            rospy.logerr(f"Error drawing projected points: {e}")
            return image

    def publish_projected_image(self, image):
        """发布带有投影点的图像"""
        try:
            # 转换为ROS图像消息
            ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = "xtion_rgb_optical_frame"
            
            # 发布图像
            self.projected_image_pub.publish(ros_image)
            
        except Exception as e:
            rospy.logerr(f"Error publishing projected image: {e}")

    def publish_pixel_coordinates(self, projected_points):
        """发布像素坐标作为标记（用于调试）"""
        try:
            marker_array = MarkerArray()
            
            for i, point_info in enumerate(projected_points):
                marker = Marker()
                marker.header.frame_id = "xtion_rgb_optical_frame"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "pixel_coordinates"
                marker.id = i
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                
                # 使用3D坐标作为位置
                marker.pose.position.x = point_info['3d_point'][0]
                marker.pose.position.y = point_info['3d_point'][1]
                marker.pose.position.z = point_info['3d_point'][2]
                
                marker.pose.orientation.w = 1.0
                
                # 设置尺寸
                marker.scale.z = 0.02  # 文本高度
                
                # 设置颜色
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                
                # 设置文本内容（像素坐标）
                u, v = point_info['pixel']
                marker.text = f"({u},{v})"
                
                # 短暂显示
                marker.lifetime = rospy.Duration(1.0)
                
                marker_array.markers.append(marker)
            
            # 发布标记
            self.pixel_markers_pub.publish(marker_array)
            
        except Exception as e:
            rospy.logerr(f"Error publishing pixel coordinate markers: {e}")

    def get_pixel_coordinates_list(self, projected_points):
        """获取像素坐标列表（供其他节点使用）"""
        pixel_list = []
        for point_info in projected_points:
            u, v = point_info['pixel']
            depth = point_info['depth']
            pixel_list.append({
                'u': u, 'v': v, 
                'depth': depth,
                'x': point_info['3d_point'][0],
                'y': point_info['3d_point'][1],
                'z': point_info['3d_point'][2]
            })
        return pixel_list

    def run(self):
        """运行节点"""
        rospy.loginfo("PointCloud to Pixel Mapper is running...")
        rospy.loginfo("Subscribing to:")
        rospy.loginfo("  - /xtion/rgb/camera_info")
        rospy.loginfo("  - /xtion/rgb/image_raw")
        rospy.loginfo("  - /cylinder_internal_points")
        rospy.loginfo("Publishing to:")
        rospy.loginfo("  - /cylinder_points_projected_image")
        rospy.loginfo("  - /cylinder_pixel_markers")
        
        rospy.spin()

if __name__ == '__main__':
    try:
        mapper = PointCloudToPixelMapper()
        mapper.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("PointCloud to Pixel Mapper node terminated")