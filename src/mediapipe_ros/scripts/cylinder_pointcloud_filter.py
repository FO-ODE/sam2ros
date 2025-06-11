#!/usr/bin/env python3
import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
import struct

class CylinderPointCloudExtractor:
    def __init__(self):
        rospy.init_node('cylinder_pointcloud_extractor', anonymous=True)
        
        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 订阅圆柱体marker
        self.marker_sub = rospy.Subscriber("/stable_arm_marker", Marker, self.marker_callback)
        
        # 订阅点云
        self.pointcloud_sub = rospy.Subscriber("/throttle_filtering_points/filtered_points", 
                                             PointCloud2, self.pointcloud_callback)
        
        # 发布提取的点云
        self.extracted_pc_pub = rospy.Publisher("/cylinder_internal_points", PointCloud2, queue_size=10)
        
        # 发布圆柱体两端的红色小球
        self.endpoint_markers_pub = rospy.Publisher("/cylinder_endpoints", MarkerArray, queue_size=10)
        
        # 存储当前圆柱体信息
        self.current_cylinder = None
        self.cylinder_frame_id = None
        
        # 缓存最新的点云数据
        self.latest_pointcloud = None
        
        rospy.loginfo("Cylinder PointCloud Extractor Node initialized")

    def marker_callback(self, msg):
        """处理接收到的圆柱体marker"""
        if msg.type != Marker.CYLINDER:
            return
        
        if msg.action == Marker.DELETE or msg.action == Marker.DELETEALL:
            rospy.loginfo("Cylinder marker deleted, clearing current cylinder")
            self.current_cylinder = None
            self.cylinder_frame_id = None
            # 清除端点标记
            self.clear_endpoint_markers()
            return
        
        # 提取圆柱体参数
        cylinder_info = self.extract_cylinder_info(msg)
        
        if cylinder_info is not None:
            self.current_cylinder = cylinder_info
            self.cylinder_frame_id = msg.header.frame_id
            
            # 打印圆柱体信息
            self.print_cylinder_info(cylinder_info, msg.header.frame_id)
            
            # 发布端点标记
            self.publish_endpoint_markers(cylinder_info, msg.header.frame_id)
            
            # 如果有缓存的点云，立即处理
            if self.latest_pointcloud is not None:
                self.process_pointcloud_extraction()

    def extract_cylinder_info(self, marker):
        """从marker中提取圆柱体信息"""
        try:
            # 圆柱体的中心位置
            center_x = marker.pose.position.x
            center_y = marker.pose.position.y
            center_z = marker.pose.position.z
            
            # 圆柱体的朝向（四元数）
            qx = marker.pose.orientation.x
            qy = marker.pose.orientation.y
            qz = marker.pose.orientation.z
            qw = marker.pose.orientation.w
            
            # 圆柱体的尺寸
            radius = marker.scale.x / 2.0  # scale.x 是直径
            height = marker.scale.z        # scale.z 是高度
            
            # 使用四元数直接计算轴线方向向量
            # 圆柱体的轴线沿着局部z轴方向，我们需要将局部z轴转换到全局坐标系
            from tf.transformations import quaternion_matrix
            
            # 构建旋转矩阵
            rotation_matrix = quaternion_matrix([qx, qy, qz, qw])
            
            # 局部z轴在全局坐标系中的方向（圆柱体的轴线方向）
            local_z_axis = np.array([0, 0, 1, 1])  # 齐次坐标
            global_axis_direction = np.dot(rotation_matrix, local_z_axis)
            axis_direction = global_axis_direction[:3]  # 取前三个分量
            
            # 归一化方向向量
            axis_direction = axis_direction / np.linalg.norm(axis_direction)
            
            # 计算圆柱体两端的坐标
            half_height = height / 2.0
            end1 = np.array([center_x, center_y, center_z]) - half_height * axis_direction
            end2 = np.array([center_x, center_y, center_z]) + half_height * axis_direction
            
            cylinder_info = {
                'center': np.array([center_x, center_y, center_z]),
                'radius': radius,
                'height': height,
                'axis_direction': axis_direction,
                'end1': end1,
                'end2': end2,
                'quaternion': np.array([qx, qy, qz, qw])
            }
            
            return cylinder_info
            
        except Exception as e:
            rospy.logerr(f"Error extracting cylinder info: {e}")
            return None

    def print_cylinder_info(self, cylinder_info, frame_id):
        """打印圆柱体信息"""
        rospy.loginfo("="*60)
        rospy.loginfo("CYLINDER INFORMATION:")
        rospy.loginfo(f"Frame ID: {frame_id}")
        rospy.loginfo(f"Radius: {cylinder_info['radius']:.4f} m")
        rospy.loginfo(f"Height: {cylinder_info['height']:.4f} m")
        rospy.loginfo(f"Center: [{cylinder_info['center'][0]:.4f}, {cylinder_info['center'][1]:.4f}, {cylinder_info['center'][2]:.4f}]")
        rospy.loginfo(f"End Point 1: [{cylinder_info['end1'][0]:.4f}, {cylinder_info['end1'][1]:.4f}, {cylinder_info['end1'][2]:.4f}]")
        rospy.loginfo(f"End Point 2: [{cylinder_info['end2'][0]:.4f}, {cylinder_info['end2'][1]:.4f}, {cylinder_info['end2'][2]:.4f}]")
        rospy.loginfo(f"Axis Direction: [{cylinder_info['axis_direction'][0]:.4f}, {cylinder_info['axis_direction'][1]:.4f}, {cylinder_info['axis_direction'][2]:.4f}]")
        rospy.loginfo("="*60)

    def publish_endpoint_markers(self, cylinder_info, frame_id):
        """发布圆柱体两端的红色小球标记"""
        try:
            marker_array = MarkerArray()
            
            # 创建两个球形标记
            for i, endpoint in enumerate([cylinder_info['end1'], cylinder_info['end2']]):
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "cylinder_endpoints"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # 设置位置
                marker.pose.position.x = endpoint[0]
                marker.pose.position.y = endpoint[1]
                marker.pose.position.z = endpoint[2]
                
                # 设置朝向（对于球体来说不重要，但保持一致性）
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # 设置尺寸（直径等于圆柱体的直径）
                diameter = cylinder_info['radius'] * 2.0
                marker.scale.x = diameter
                marker.scale.y = diameter
                marker.scale.z = diameter
                
                # 设置为红色
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                
                # 永久显示
                marker.lifetime = rospy.Duration(0)
                
                marker_array.markers.append(marker)
            
            # 发布标记数组
            self.endpoint_markers_pub.publish(marker_array)
            
            rospy.loginfo(f"Published endpoint markers: 2 red spheres with radius {cylinder_info['radius']:.4f}m")
            
        except Exception as e:
            rospy.logerr(f"Error publishing endpoint markers: {e}")

    def clear_endpoint_markers(self):
        """清除端点标记"""
        try:
            marker_array = MarkerArray()
            
            # 创建删除标记
            for i in range(2):
                marker = Marker()
                marker.header.frame_id = self.cylinder_frame_id if self.cylinder_frame_id else "base_link"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "cylinder_endpoints"
                marker.id = i
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)
            
            # 发布删除标记
            self.endpoint_markers_pub.publish(marker_array)
            rospy.loginfo("Cleared endpoint markers")
            
        except Exception as e:
            rospy.logerr(f"Error clearing endpoint markers: {e}")

    def pointcloud_callback(self, msg):
        """处理接收到的点云数据"""
        self.latest_pointcloud = msg
        
        # 如果有当前圆柱体，立即处理
        if self.current_cylinder is not None:
            self.process_pointcloud_extraction()

    def transform_pointcloud_to_cylinder_frame(self, pointcloud_msg):
        """将点云转换到圆柱体的坐标系"""
        if pointcloud_msg.header.frame_id == self.cylinder_frame_id:
            return pointcloud_msg
        
        try:
            # 等待变换可用
            transform = self.tf_buffer.lookup_transform(
                self.cylinder_frame_id,
                pointcloud_msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            # 注意：这里我们需要手动转换每个点
            # 因为tf2_geometry_msgs没有直接支持PointCloud2的转换
            return self.manually_transform_pointcloud(pointcloud_msg, transform)
            
        except Exception as e:
            rospy.logwarn(f"TF transformation failed: {e}")
            return None

    def manually_transform_pointcloud(self, pointcloud_msg, transform):
        """手动转换点云中的每个点"""
        try:
            # 提取变换矩阵
            t = transform.transform.translation
            r = transform.transform.rotation
            
            # 转换为变换矩阵
            from tf.transformations import quaternion_matrix, translation_matrix
            
            rotation_matrix = quaternion_matrix([r.x, r.y, r.z, r.w])
            translation_matrix_4x4 = translation_matrix([t.x, t.y, t.z])
            transform_matrix = np.dot(translation_matrix_4x4, rotation_matrix)
            
            # 读取原始点云数据
            points_list = []
            for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                # 应用变换
                point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
                transformed_point = np.dot(transform_matrix, point_homogeneous)
                points_list.append([transformed_point[0], transformed_point[1], transformed_point[2]])
            
            # 创建新的点云消息
            transformed_msg = PointCloud2()
            transformed_msg.header.frame_id = self.cylinder_frame_id
            transformed_msg.header.stamp = pointcloud_msg.header.stamp
            
            # 重新打包点云数据
            transformed_msg = pc2.create_cloud_xyz32(transformed_msg.header, points_list)
            
            return transformed_msg
            
        except Exception as e:
            rospy.logerr(f"Manual pointcloud transformation failed: {e}")
            return None

    def is_point_inside_cylinder(self, point, cylinder_info):
        """检查点是否在圆柱体内部"""
        # 点相对于圆柱体中心的向量
        point_vec = point - cylinder_info['center']
        
        # 计算点在圆柱体轴线上的投影
        axis_projection = np.dot(point_vec, cylinder_info['axis_direction'])
        
        # 检查点是否在圆柱体的高度范围内
        half_height = cylinder_info['height'] / 2.0
        if abs(axis_projection) > half_height:
            return False
        
        # 计算点到圆柱体轴线的距离
        # 向量在轴线方向上的分量
        axis_component = axis_projection * cylinder_info['axis_direction']
        # 垂直于轴线的分量
        perpendicular_component = point_vec - axis_component
        # 到轴线的距离
        distance_to_axis = np.linalg.norm(perpendicular_component)
        
        # 检查点是否在圆柱体的半径范围内
        return distance_to_axis <= cylinder_info['radius']

    def process_pointcloud_extraction(self):
        """处理点云提取"""
        if self.latest_pointcloud is None or self.current_cylinder is None:
            return
        
        try:
            # 如果点云不在圆柱体的坐标系中，进行坐标变换
            if self.latest_pointcloud.header.frame_id != self.cylinder_frame_id:
                transformed_pc = self.transform_pointcloud_to_cylinder_frame(self.latest_pointcloud)
                if transformed_pc is None:
                    rospy.logwarn("Failed to transform pointcloud, using original coordinates")
                    working_pc = self.latest_pointcloud
                else:
                    working_pc = transformed_pc
            else:
                working_pc = self.latest_pointcloud
            
            # 提取圆柱体内部的点
            internal_points = []
            total_points = 0
            
            for point in pc2.read_points(working_pc, field_names=("x", "y", "z"), skip_nans=True):
                total_points += 1
                point_array = np.array([point[0], point[1], point[2]])
                
                if self.is_point_inside_cylinder(point_array, self.current_cylinder):
                    internal_points.append(point)
            
            rospy.loginfo(f"Found {len(internal_points)} points inside cylinder out of {total_points} total points")
            
            # 如果有内部点云，发布它们
            if internal_points:
                self.publish_internal_pointcloud(internal_points, working_pc.header)
            else:
                rospy.logwarn("No points found inside the cylinder")
                
        except Exception as e:
            rospy.logerr(f"Error processing pointcloud extraction: {e}")

    def publish_internal_pointcloud(self, points, original_header):
        """发布圆柱体内部的点云"""
        try:
            # 创建新的点云消息
            header = original_header
            header.stamp = rospy.Time.now()
            
            # 使用圆柱体的坐标系作为发布的坐标系
            if self.cylinder_frame_id:
                header.frame_id = self.cylinder_frame_id
            
            # 创建点云
            internal_pc = pc2.create_cloud_xyz32(header, points)
            
            # 发布点云
            self.extracted_pc_pub.publish(internal_pc)
            
            rospy.logdebug(f"Published {len(points)} internal points in frame {header.frame_id}")
            
        except Exception as e:
            rospy.logerr(f"Error publishing internal pointcloud: {e}")

    def run(self):
        """运行节点"""
        rospy.loginfo("Cylinder PointCloud Extractor is running...")
        rospy.loginfo("Waiting for cylinder marker on /stable_arm_marker")
        rospy.loginfo("Waiting for pointcloud on /throttle_filtering_points/filtered_points")
        rospy.loginfo("Will publish extracted points on /cylinder_internal_points")
        rospy.loginfo("Will publish endpoint markers on /cylinder_endpoints")
        rospy.spin()

if __name__ == '__main__':
    try:
        extractor = CylinderPointCloudExtractor()
        extractor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Cylinder PointCloud Extractor node terminated")