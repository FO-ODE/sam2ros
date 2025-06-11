#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class CoordinateDebugger:
    def __init__(self):
        rospy.init_node('coordinate_debugger', anonymous=True)
        
        # 订阅者
        self.pc_sub = rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2, self.pc_callback)
        self.joints_sub = rospy.Subscriber("/pose_joints_3d", String, self.joints_callback)
        
        self.latest_pc = None
        
    def pc_callback(self, pc_msg):
        """分析点云数据的统计信息"""
        self.latest_pc = pc_msg
        
        # 读取所有点云数据
        points = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
        
        if points:
            points_array = np.array(points)
            
            # 计算统计信息
            x_min, x_max = np.min(points_array[:, 0]), np.max(points_array[:, 0])
            y_min, y_max = np.min(points_array[:, 1]), np.max(points_array[:, 1])
            z_min, z_max = np.min(points_array[:, 2]), np.max(points_array[:, 2])
            
            rospy.loginfo("=" * 50)
            rospy.loginfo("Point Cloud Statistics:")
            rospy.loginfo(f"Frame ID: {pc_msg.header.frame_id}")
            rospy.loginfo(f"Total points: {len(points)}")
            rospy.loginfo(f"Dimensions: {pc_msg.width} x {pc_msg.height}")
            rospy.loginfo(f"X range: {x_min:.3f} to {x_max:.3f} meters")
            rospy.loginfo(f"Y range: {y_min:.3f} to {y_max:.3f} meters") 
            rospy.loginfo(f"Z range: {z_min:.3f} to {z_max:.3f} meters")
            
            # 计算距离原点的距离
            distances = np.sqrt(np.sum(points_array**2, axis=1))
            rospy.loginfo(f"Distance range: {np.min(distances):.3f} to {np.max(distances):.3f} meters")
            rospy.loginfo(f"Mean distance: {np.mean(distances):.3f} meters")
            
            # 检查中心区域的点
            center_x = pc_msg.width // 2
            center_y = pc_msg.height // 2
            center_points = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), 
                                               skip_nans=True, uvs=[(center_x, center_y)]))
            if center_points:
                cp = center_points[0]
                rospy.loginfo(f"Center point ({center_x}, {center_y}): ({cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f})")
            
    def joints_callback(self, joints_msg):
        """分析关节数据"""
        joints_data = joints_msg.data
        rospy.loginfo("=" * 50)
        rospy.loginfo("Joints 3D Data:")
        rospy.loginfo(f"Data length: {len(joints_data)} characters")
        
        # 解析关节信息
        if joints_data:
            joints = joints_data.split("; ")
            for joint in joints[:3]:  # 只显示前3个关节
                rospy.loginfo(f"  {joint}")
        
    def run(self):
        rospy.loginfo("Starting coordinate debugger...")
        rospy.loginfo("This will analyze point cloud and joint coordinate data")
        rospy.loginfo("Make sure to have a person in front of the camera")
        rospy.spin()

if __name__ == '__main__':
    try:
        debugger = CoordinateDebugger()
        debugger.run()
    except rospy.ROSInterruptException:
        pass