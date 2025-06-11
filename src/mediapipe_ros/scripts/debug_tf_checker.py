#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs

def check_transforms():
    rospy.init_node('tf_checker')
    
    # 创建TF buffer和listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    rospy.sleep(2.0)  # 等待TF数据
    
    print("=" * 60)
    print("TF Frame Analysis")
    print("=" * 60)
    
    # 检查常见的frame
    frames_to_check = [
        'map', 'odom', 'base_link', 'base_footprint',
        'xtion_rgb_frame', 'xtion_rgb_optical_frame', 
        'xtion_depth_frame', 'xtion_depth_optical_frame'
    ]
    
    available_frames = []
    
    for frame in frames_to_check:
        try:
            # 尝试获取从该frame到自身的变换（这会告诉我们frame是否存在）
            transform = tf_buffer.lookup_transform(frame, frame, rospy.Time(0), rospy.Duration(1.0))
            available_frames.append(frame)
            print(f"✓ Frame '{frame}' is available")
        except tf2_ros.LookupException:
            print(f"✗ Frame '{frame}' not found")
        except tf2_ros.ConnectivityException:
            print(f"✗ Frame '{frame}' not connected")
        except tf2_ros.ExtrapolationException:
            print(f"✗ Frame '{frame}' extrapolation error")
    
    print("\n" + "=" * 60)
    print("Available Frames:")
    for frame in available_frames:
        print(f"  - {frame}")
    
    # 检查关键变换
    print("\n" + "=" * 60)
    print("Key Transformations:")
    
    if 'xtion_rgb_optical_frame' in available_frames:
        print(f"\n📍 Using 'xtion_rgb_optical_frame' as reference")
        
        # 检查与其他重要frame的关系
        for target_frame in ['map', 'odom', 'base_link']:
            if target_frame in available_frames:
                try:
                    transform = tf_buffer.lookup_transform(
                        target_frame, 'xtion_rgb_optical_frame', 
                        rospy.Time(0), rospy.Duration(1.0)
                    )
                    t = transform.transform.translation
                    r = transform.transform.rotation
                    print(f"  {target_frame} -> xtion_rgb_optical_frame:")
                    print(f"    Translation: x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}")
                    print(f"    Rotation: x={r.x:.3f}, y={r.y:.3f}, z={r.z:.3f}, w={r.w:.3f}")
                except Exception as e:
                    print(f"  {target_frame} -> xtion_rgb_optical_frame: ❌ {e}")
    
    print("\n" + "=" * 60)
    print("RViz Configuration Recommendations:")
    print("=" * 60)
    
    if 'xtion_rgb_optical_frame' in available_frames:
        print("✅ RECOMMENDED: Set Fixed Frame to 'xtion_rgb_optical_frame'")
        print("   This matches your point cloud and joint data frame")
    elif 'map' in available_frames:
        print("⚠️  ALTERNATIVE: Set Fixed Frame to 'map'")
        print("   Make sure TF between map and camera is correct")
    elif 'base_link' in available_frames:
        print("⚠️  ALTERNATIVE: Set Fixed Frame to 'base_link'")
        print("   Make sure TF between robot and camera is correct")
    else:
        print("❌ WARNING: No suitable fixed frame found!")
        print("   You may need to publish TF transforms")
    
    print("\n📋 Debugging Steps:")
    print("1. In RViz, set Fixed Frame to 'xtion_rgb_optical_frame'")
    print("2. Add PointCloud2 display for your point cloud topic")
    print("3. Add MarkerArray display for /pose_skeleton_markers")
    print("4. Both should appear in the same coordinate space")
    print("\n🔧 If markers still appear wrong:")
    print("- Check if camera is mounted upside down or rotated")
    print("- Verify camera calibration")
    print("- Check if depth and RGB are properly aligned")

if __name__ == '__main__':
    try:
        check_transforms()
    except rospy.ROSInterruptException:
        pass