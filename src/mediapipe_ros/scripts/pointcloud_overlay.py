#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import ros_numpy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
import tf.transformations as tf_trans

class PointCloudSegmenter:
    def __init__(self):
        rospy.init_node("pointcloud_segmenter")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pc_sub = rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2, self.pc_callback)
        self.marker_sub = rospy.Subscriber("/stable_arm_marker", Marker, self.marker_callback)
        self.pc_pub = rospy.Publisher("/segmented_points", PointCloud2, queue_size=1)

        self.target_frame = "xtion_rgb_optical_frame"
        self.marker = None

    def marker_callback(self, msg):
        self.marker = msg

    def pc_callback(self, msg):
        if self.marker is None:
            rospy.logwarn_throttle(5.0, "No marker received yet")
            return

        try:
            # 点云转换到 target_frame（相机坐标系）
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
            pc_np = self.transform_points(pc_np, transform)

            # 圆柱体中心（变换到相机坐标系）
            marker_pose = self.marker.pose.position
            marker_tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.marker.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            marker_center = np.array([
                marker_pose.x, marker_pose.y, marker_pose.z, 1.0
            ])
            T = self.transform_matrix(marker_tf.transform)
            marker_center = T @ marker_center
            marker_center = marker_center[:3]

            # 参数
            radius = self.marker.scale.x / 2.0
            cam_origin = np.array([0.0, 0.0, 0.0])
            v = marker_center - cam_origin
            v_norm = np.linalg.norm(v)

            # 点在相机到marker之间且到连线距离小于半径
            selected = []
            for p in pc_np:
                pv = p - cam_origin
                proj = np.dot(pv, v) / v_norm
                if proj < 0 or proj > v_norm:
                    continue
                closest_point = cam_origin + proj * v / v_norm
                dist_to_line = np.linalg.norm(p - closest_point)
                if dist_to_line <= radius:
                    selected.append(p)

            if not selected:
                rospy.loginfo_throttle(5.0, "No points in region")
                return

            # 转回 PointCloud2 并发布
            selected_np = np.array(selected)
            pc_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
                ros_numpy.point_cloud2.xyz_array_to_pointcloud2(selected_np, stamp=rospy.Time.now(), frame_id=self.target_frame)
            )
            self.pc_pub.publish(pc_msg)

        except Exception as e:
            rospy.logerr(f"TF or processing failed: {e}")

    def transform_matrix(self, tf):
        t = tf.translation
        q = tf.rotation
        trans = np.array([t.x, t.y, t.z])
        rot = np.array([q.x, q.y, q.z, q.w])
        T = tf_trans.quaternion_matrix(rot)
        T[0:3, 3] = trans
        return T

    def transform_points(self, points, tf):
        T = self.transform_matrix(tf.transform)
        points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
        points_trans = (T @ points_homo.T).T
        return points_trans[:, :3]

if __name__ == "__main__":
    PointCloudSegmenter()
    rospy.spin()
