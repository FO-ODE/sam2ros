#!/usr/bin/env python3
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Quaternion
from std_srvs.srv import Empty, EmptyResponse
import tf.transformations as tf_trans
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

class ArmStabilityDetector:
    def __init__(self):
        rospy.init_node('arm_stability_detector', anonymous=True)
        
        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 坐标系设置
        self.source_frame = "xtion_rgb_optical_frame"
        self.target_frame = "base_link"
        
        # 订阅手臂延伸marker
        self.arm_marker_sub = rospy.Subscriber("/arm_extension_marker", Marker, self.arm_marker_callback)
        
        # 发布稳定的绿色圆柱体
        self.stable_marker_pub = rospy.Publisher("/stable_arm_marker", Marker, queue_size=10)
        
        # 稳定性检测参数
        self.stability_duration = 5.0  # 需要保持稳定的时间（秒）
        self.position_threshold = 0.05  # 位置变化阈值（米）
        self.orientation_threshold = 0.1  # 朝向变化阈值（四元数差值）
        self.publish_rate = 10.0  # 绿色marker发布频率（Hz）
        
        # 状态变量
        self.last_marker = None
        self.stable_start_time = None
        self.is_tracking_stability = False
        self.green_marker_published = False
        self.green_marker_id_counter = 0
        self.green_markers = []  # 存储所有需要持续发布的绿色markers
        
        # 定时器用于持续发布绿色markers
        self.publish_timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), 
            self.publish_green_markers_callback
        )
        
        # 添加清除绿色markers的服务
        self.clear_service = rospy.Service('clear_stable_markers', Empty, self.clear_markers_service)
        
        rospy.loginfo("Arm Stability Detector Node initialized")
        rospy.loginfo("Use 'rosservice call /clear_stable_markers' to clear all green markers")
        rospy.loginfo(f"Transforming from {self.source_frame} to {self.target_frame}")
        
        # 等待TF变换可用
        self.wait_for_tf_transform()

    def quaternion_distance(self, q1, q2):
        """计算两个四元数之间的距离"""
        # 将四元数转换为numpy数组
        q1_array = np.array([q1.x, q1.y, q1.z, q1.w])
        q2_array = np.array([q2.x, q2.y, q2.z, q2.w])
        
        # 计算点积（考虑四元数的双重表示）
        dot_product = abs(np.dot(q1_array, q2_array))
        
        # 将点积限制在有效范围内
        dot_product = min(1.0, dot_product)
        
        # 计算角度距离
        angle_distance = 2.0 * np.arccos(dot_product)
        
        return angle_distance

    def is_marker_similar(self, marker1, marker2):
        """检查两个marker是否相似（位置和朝向接近）"""
        if marker1 is None or marker2 is None:
            return False
        
        # 检查位置差异
        pos_diff = np.sqrt(
            (marker1.pose.position.x - marker2.pose.position.x) ** 2 +
            (marker1.pose.position.y - marker2.pose.position.y) ** 2 +
            (marker1.pose.position.z - marker2.pose.position.z) ** 2
        )
        
        # 检查朝向差异
        orient_diff = self.quaternion_distance(marker1.pose.orientation, marker2.pose.orientation)
        
        # 判断是否相似
        position_similar = pos_diff < self.position_threshold
        orientation_similar = orient_diff < self.orientation_threshold
        
        rospy.logdebug(f"Position diff: {pos_diff:.4f}m, Orientation diff: {orient_diff:.4f}rad")
        rospy.logdebug(f"Position similar: {position_similar}, Orientation similar: {orientation_similar}")
        
        return position_similar and orientation_similar

    def wait_for_tf_transform(self):
        """等待TF变换可用"""
        rospy.loginfo(f"Waiting for TF transform from {self.source_frame} to {self.target_frame}...")
        
        try:
            # 等待变换可用（最多等待10秒）
            self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rospy.Time(0),  # 获取最新可用的变换
                rospy.Duration(10.0)
            )
            rospy.loginfo("TF transform is available!")
            
        except Exception as e:
            rospy.logwarn(f"TF transform not available: {e}")
            rospy.logwarn("The node will continue but transformations may fail")

    def transform_pose_to_base_link(self, pose_stamped):
        """将pose从源坐标系转换到base_link"""
        try:
            # 使用最新可用的变换时间（避免时间同步问题）
            pose_stamped.header.stamp = rospy.Time(0)
            
            # 执行变换
            transformed_pose = self.tf_buffer.transform(pose_stamped, self.target_frame, rospy.Duration(1.0))
            return transformed_pose
            
        except Exception as e:
            rospy.logerr(f"TF transformation failed: {e}")
            
            # 尝试使用当前时间进行变换
            try:
                pose_stamped.header.stamp = rospy.Time.now()
                transformed_pose = self.tf_buffer.transform(pose_stamped, self.target_frame, rospy.Duration(1.0))
                rospy.logwarn("Successfully transformed using current time")
                return transformed_pose
            except Exception as e2:
                rospy.logerr(f"Second transformation attempt failed: {e2}")
                return None

    def create_green_marker(self, reference_marker):
        """基于参考marker创建绿色的稳定marker（转换为 base_link 坐标系）"""
        green_marker = Marker()

        # 原始 pose + header 组合成 PoseStamped 进行变换
        original_pose = PoseStamped()
        original_pose.header = reference_marker.header
        original_pose.pose = reference_marker.pose

        # 坐标变换
        transformed_pose = self.transform_pose_to_base_link(original_pose)
        if transformed_pose is None:
            return None  # 变换失败则放弃生成 marker

        # 填充 header
        green_marker.header.frame_id = self.target_frame
        green_marker.header.stamp = rospy.Time.now()
        green_marker.ns = "stable_arm_extension"
        green_marker.id = self.green_marker_id_counter
        green_marker.type = Marker.CYLINDER
        green_marker.action = Marker.ADD

        # 使用变换后的 pose
        green_marker.pose = transformed_pose.pose

        # 复制尺寸
        green_marker.scale = reference_marker.scale
        green_marker.scale.x = reference_marker.scale.x * 2 # diameter
        green_marker.scale.y = reference_marker.scale.y * 2 # diameter

        # 设置为绿色
        green_marker.color.r = 0.0
        green_marker.color.g = 1.0
        green_marker.color.b = 0.0
        green_marker.color.a = 0.8

        # 永久显示
        green_marker.lifetime = rospy.Duration(0)

        self.green_marker_id_counter += 1

        return green_marker


    def arm_marker_callback(self, msg):
        """处理接收到的手臂延伸marker"""
        current_time = rospy.Time.now()
        
        # 检查是否是删除操作
        if msg.action == Marker.DELETE or msg.action == Marker.DELETEALL:
            rospy.logdebug("Received delete marker, resetting stability tracking")
            self.reset_stability_tracking()
            return
        
        # 如果这是第一个有效的marker
        if self.last_marker is None:
            rospy.loginfo("First valid marker received, starting stability tracking")
            self.last_marker = msg
            self.stable_start_time = current_time
            self.is_tracking_stability = True
            self.green_marker_published = False
            return
        
        # 检查当前marker与上一个marker是否相似
        if self.is_marker_similar(self.last_marker, msg):
            if not self.is_tracking_stability:
                # 开始新的稳定性追踪
                rospy.loginfo("Marker became stable, starting stability timer")
                self.stable_start_time = current_time
                self.is_tracking_stability = True
                self.green_marker_published = False
            else:
                # 继续稳定状态
                elapsed_time = (current_time - self.stable_start_time).to_sec()
                rospy.logdebug(f"Marker stable for {elapsed_time:.2f}s")
                
                # 检查是否已经稳定足够长时间且还未发布绿色marker
                if elapsed_time >= self.stability_duration and not self.green_marker_published:
                    rospy.loginfo(f"Marker stable for {elapsed_time:.2f}s, creating green marker")
                    green_marker = self.create_green_marker(msg)
                    if green_marker is not None:
                        self.green_markers.append(green_marker)
                        self.green_marker_published = True
                    else:
                        rospy.logwarn("Failed to create green marker due to TF transformation error")
        else:
            # marker发生了显著变化，重置稳定性追踪
            if self.is_tracking_stability:
                elapsed_time = (current_time - self.stable_start_time).to_sec()
                rospy.loginfo(f"Marker changed after {elapsed_time:.2f}s, resetting stability tracking")
            
            self.stable_start_time = current_time
            self.is_tracking_stability = True
            self.green_marker_published = False
        
        # 更新上一个marker
        self.last_marker = msg

    def publish_green_markers_callback(self, event):
        """定时器回调函数，持续发布所有绿色markers"""
        current_time = rospy.Time.now()
        
        for green_marker in self.green_markers:
            # 更新时间戳，但保持frame_id为base_link
            green_marker.header.stamp = current_time
            green_marker.header.frame_id = self.target_frame  # 确保frame_id是base_link
            # 发布marker
            self.stable_marker_pub.publish(green_marker)
        
        if len(self.green_markers) > 0:
            rospy.logdebug(f"Published {len(self.green_markers)} green markers in {self.target_frame}")

    def clear_markers_service(self, req):
        """ROS服务回调：清除所有绿色markers"""
        self.clear_all_green_markers()
        return EmptyResponse()

    def add_clear_markers_service(self):
        """可选：添加清除所有绿色markers的服务"""
        # 如果需要，可以在这里添加ROS服务来清除所有绿色markers
        pass

    def clear_all_green_markers(self):
        """清除所有绿色markers"""
        # 发送删除命令
        for green_marker in self.green_markers:
            delete_marker = Marker()
            delete_marker.header = green_marker.header
            delete_marker.header.stamp = rospy.Time.now()
            delete_marker.header.frame_id = self.target_frame  # 确保使用正确的frame_id
            delete_marker.ns = green_marker.ns
            delete_marker.id = green_marker.id
            delete_marker.action = Marker.DELETE
            self.stable_marker_pub.publish(delete_marker)
        
        # 清空列表
        self.green_markers.clear()
        rospy.loginfo("Cleared all green markers")

    def reset_stability_tracking(self):
        """重置稳定性追踪状态"""
        self.last_marker = None
        self.stable_start_time = None
        self.is_tracking_stability = False
        self.green_marker_published = False

    def run(self):
        """运行节点"""
        rospy.loginfo("Arm Stability Detector is running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ArmStabilityDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Arm Stability Detector node terminated")