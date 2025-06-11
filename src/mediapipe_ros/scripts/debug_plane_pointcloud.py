import rospy
from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2

class PointCloudToImage:
    def __init__(self):
        rospy.init_node('pointcloud_to_image', anonymous=True)
        self.camera_info = None
        rospy.Subscriber("/xtion/rgb/camera_info", CameraInfo, self.camera_info_cb)
        rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2, self.pc_cb)

    def camera_info_cb(self, msg):
        if self.camera_info is None:
            self.camera_info = msg

    def pc_cb(self, msg):
        if self.camera_info is None:
            rospy.logwarn("Waiting for camera info...")
            return
        
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        width = self.camera_info.width
        height = self.camera_info.height

        # 初始化深度图（或颜色图）
        depth_image = np.zeros((height, width), dtype=np.float32)

        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            if z <= 0.0:
                continue
            u = int((fx * x) / z + cx)
            v = int((fy * y) / z + cy)
            if 0 <= u < width and 0 <= v < height:
                # 深度图里记录最小的Z（防止多个点投到同一像素）
                if depth_image[v, u] == 0 or z < depth_image[v, u]:
                    depth_image[v, u] = z

        # 归一化显示
        depth_vis = (depth_image - np.nanmin(depth_image)) / (np.nanmax(depth_image) - np.nanmin(depth_image) + 1e-8)
        depth_vis = (depth_vis * 255).astype(np.uint8)

        # 显示图像（或保存）
        cv2.imshow("Projected Depth", depth_vis)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        PointCloudToImage()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
