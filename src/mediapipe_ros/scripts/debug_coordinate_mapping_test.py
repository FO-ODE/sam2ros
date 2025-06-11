#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class CoordinateMappingTest:
    def __init__(self):
        rospy.init_node('coordinate_mapping_test', anonymous=True)
        
        self.bridge = CvBridge()
        
        # 订阅者
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.pc_sub = rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2, self.pc_callback)
        
        self.latest_image = None
        self.latest_pc = None
        self.image_width = None
        self.image_height = None
        
        # 鼠标点击回调
        cv2.namedWindow("Image with Click Test")
        cv2.setMouseCallback("Image with Click Test", self.mouse_callback)
        
    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_height, self.image_width = self.latest_image.shape[:2]
        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")
            
    def pc_callback(self, msg):
        self.latest_pc = msg
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击事件 - 显示点击位置对应的3D坐标"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.latest_pc is None:
                rospy.logwarn("No point cloud data available")
                return
                
            rospy.loginfo(f"Clicked at image coordinates: ({x}, {y})")
            
            # 测试不同的映射方法
            self.test_mapping_methods(x, y)
            
    def test_mapping_methods(self, x_2d, y_2d):
        """测试不同的坐标映射方法"""
        if self.latest_pc is None:
            return
            
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Testing coordinate mapping for image point ({x_2d}, {y_2d})")
        rospy.loginfo(f"Image size: {self.image_width} x {self.image_height}")
        rospy.loginfo(f"Point cloud: {self.latest_pc.width} x {self.latest_pc.height}")
        
        # 方法1: 直接像素映射 (原始方法)
        if self.latest_pc.width > 1 and self.latest_pc.height > 1:
            pc_x = int((x_2d / self.image_width) * self.latest_pc.width)
            pc_y = int((y_2d / self.image_height) * self.latest_pc.height)
            
            rospy.loginfo(f"Method 1 - Direct pixel mapping: PC({pc_x}, {pc_y})")
            
            try:
                points = list(pc2.read_points(self.latest_pc, field_names=("x", "y", "z"), 
                                            skip_nans=True, uvs=[(pc_x, pc_y)]))
                if points:
                    p = points[0]
                    rospy.loginfo(f"  Result: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
                else:
                    rospy.loginfo("  Result: No valid point")
            except Exception as e:
                rospy.loginfo(f"  Error: {e}")
        
        # 方法2: 在点云周围区域搜索
        rospy.loginfo("Method 2 - Neighborhood search:")
        self.search_neighborhood(x_2d, y_2d)
        
        # 方法3: 基于深度的筛选
        rospy.loginfo("Method 3 - Depth-based filtering:")
        self.depth_based_search(x_2d, y_2d)
        
    def search_neighborhood(self, x_2d, y_2d, radius=5):
        """在图像坐标周围搜索有效的3D点"""
        valid_points = []
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                test_x = x_2d + dx
                test_y = y_2d + dy
                
                if (0 <= test_x < self.image_width and 0 <= test_y < self.image_height and
                    self.latest_pc.width > 1 and self.latest_pc.height > 1):
                    
                    pc_x = int((test_x / self.image_width) * self.latest_pc.width)
                    pc_y = int((test_y / self.image_height) * self.latest_pc.height)
                    
                    if pc_x < self.latest_pc.width and pc_y < self.latest_pc.height:
                        try:
                            points = list(pc2.read_points(self.latest_pc, field_names=("x", "y", "z"), 
                                                        skip_nans=True, uvs=[(pc_x, pc_y)]))
                            if points:
                                p = points[0]
                                if not (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2])):
                                    if 1.0 <= p[2] <= 4.0:  # 合理的深度范围
                                        valid_points.append((p, dx, dy))
                        except:
                            continue
        
        if valid_points:
            # 选择距离中心最近的点
            best_point = min(valid_points, key=lambda x: x[1]**2 + x[2]**2)
            p, dx, dy = best_point
            rospy.loginfo(f"  Found valid point at offset ({dx}, {dy}): ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        else:
            rospy.loginfo("  No valid points found in neighborhood")
            
    def depth_based_search(self, x_2d, y_2d):
        """基于深度值搜索最合理的点"""
        # 读取所有点云数据
        try:
            all_points = list(pc2.read_points(self.latest_pc, field_names=("x", "y", "z"), skip_nans=True))
            
            if not all_points:
                rospy.loginfo("  No valid points in entire point cloud")
                return
                
            # 转换为numpy数组
            points_array = np.array(all_points)
            
            # 筛选合理深度范围的点
            valid_mask = (points_array[:, 2] >= 1.0) & (points_array[:, 2] <= 4.0)
            valid_points = points_array[valid_mask]
            
            if len(valid_points) == 0:
                rospy.loginfo("  No points in reasonable depth range (1-4m)")
                return
                
            rospy.loginfo(f"  Found {len(valid_points)} points in reasonable depth range")
            
            # 找到最接近屏幕中心区域的点
            norm_x = (x_2d - self.image_width/2) / (self.image_width/2)
            norm_y = (y_2d - self.image_height/2) / (self.image_height/2)
            
            # 简化投影计算
            best_distance = float('inf')
            best_point = None
            
            for point in valid_points:
                if point[2] > 0.1:
                    proj_x = point[0] / point[2]
                    proj_y = -point[1] / point[2]  # Y轴翻转
                    
                    distance = np.sqrt((proj_x - norm_x)**2 + (proj_y - norm_y)**2)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_point = point
            
            if best_point is not None:
                rospy.loginfo(f"  Best match: ({best_point[0]:.3f}, {best_point[1]:.3f}, {best_point[2]:.3f})")
                rospy.loginfo(f"  Projection distance: {best_distance:.4f}")
            else:
                rospy.loginfo("  No suitable point found")
                
        except Exception as e:
            rospy.loginfo(f"  Error in depth-based search: {e}")
    
    def run(self):
        rospy.loginfo("Starting coordinate mapping test")
        rospy.loginfo("Click on the image to test coordinate mapping")
        rospy.loginfo("The real human coordinates should be around:")
        rospy.loginfo("  Nose: (-0.039, -0.56, 2.48)")
        rospy.loginfo("  Left shoulder: (0.098, -0.36, 2.69)")
        rospy.loginfo("  Right shoulder: (-0.172, -0.365, 2.46)")
        
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                # 在图像上绘制一些标记
                img_copy = self.latest_image.copy()
                
                # 绘制中心点
                center_x = self.image_width // 2
                center_y = self.image_height // 2
                cv2.circle(img_copy, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(img_copy, "Center", (center_x + 10, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 绘制说明
                cv2.putText(img_copy, "Click to test coordinate mapping", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Image with Click Test", img_copy)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
            rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        test = CoordinateMappingTest()
        test.run()
    except rospy.ROSInterruptException:
        pass