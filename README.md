# FP: SAM2 + Clip

## Containers

### Ultralytics + SAM

```bash
rocker --nvidia --x11 --privileged \
    --volume /home/zby/ros/FP_workspace/src:/catkin_ws/src \
    --network host \
    --name ultralytics_container \
    foode258/ultralytics_ros_base:env1.3
```

### CLIP

```bash
rocker --nvidia --x11 --privileged \
    --volume /home/zby/ros/FP_workspace/src:/catkin_ws/src \
    --network host \
    --name clip_container \
    foode258/clip_ros_base:env1.0
```

### Tiago

```bash
rocker --nvidia --x11 --privileged \
    --network host \
    --name tiago_container \
    palroboticssl/tiago_tutorials:noetic
```

## Useful Commands

```bash
sudo kill

rqt

rosbag play -l <filename.bag>

rosnode cleanup

rosrun smach_viewer smach_viewer.py

rosservice list

roslaunch tiago_moveit_config moveit_rviz.launch config:=true
```

## Tiago Robot

### Connect

**Tiago Webmanager**: <http://192.168.1.200:8080>

```bash
# to asure the IP
ifconfig

# Ethernet
export ROS_MASTER_URI=http://192.168.1.200:11311 \
export ROS_IP=10.68.0.131

# WLAN
export ROS_MASTER_URI=http://192.168.1.200:11311 \
export ROS_IP=192.168.1.103
```

### Connection Test

```bash
ping tiago-46c
ping http://192.168.1.200

ssh pal@tiago-46c

rostopic pub /test_topic std_msgs/String "data: 'test'"
rostopic echo /test_topic
```

### Timer Synchronization

```bash
# sudo apt install ntpdate
sudo ntpdate 192.168.1.200

# check the date
watch -n 0.1 date
```

### Tiago Navigation & Map

```bash
rosrun map_server map_server src/carry_my_luggage/ics_map/map.yaml

rosrun rviz rviz -d'rospack find tiago_2dnav'/config/rviz/navigation.rviz

rosservice call /pal_map_manager/change_map "input: '2025-01-30_124458'"
```

### Tiago RViz

if you are using `rosbag`

```bash
# 先生成带gripper的URDF
rosrun xacro xacro `rospack find tiago_description`/robots/tiago.urdf.xacro end_effector:=pal-gripper > /tmp/tiago_gripper.urdf

# 然后启动display
roslaunch urdf_tutorial display.launch model:=/tmp/tiago_gripper.urdf
```

## Others

### Rosbag Record

Topics to be recorded in: `Tutorial4_ws2425.pdf`

Record the rosbag in ssh, example:

```bash
# please record the bag in /tmp
# pal@tiago-46c:/tmp$ rosbag record

rosbag record \
/tf \
/tf_static \
/clock \
/xtion/rgb/image_raw \
/xtion/rgb/camera_info \
/xtion/depth/camera_info \
/xtion/depth_registered/camera_info \
/xtion/depth_registered/image \
/throttle_filtering_points/filtered_points \
```

Copy the file from remote back to host, example:

```bash
scp pal@tiago-46c:/tmp/2024-12-18-15-59-40.bag ~/Documents

# you will see in terminal:
zby@ub2004:~$ scp pal@tiago-46c:/tmp/2024-12-18-15-52-52.bag ~/Documents
pal@tiago-46c's password:
2024-12-18-15-52-52.bag                       100%  390MB   2.0MB/s   03:18  
```

### Camera Info

```bash
header: 
  seq: 3696
  stamp: 
    secs: 1749598854
    nsecs: 447469523
  frame_id: "xtion_rgb_optical_frame"
height: 480
width: 640
distortion_model: "plumb_bob"
D: [0.05885243042898163, -0.08343080029493587, 0.004227265808599358, -0.001760150057091595, 0.0]
K: [579.7253645876941, 0.0, 329.1726287752489, 0.0, 581.0305537009599, 252.8879914424264, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [586.369140625, 0.0, 327.6738649184117, 0.0, 0.0, 586.5962524414062, 254.1431469189192, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---
```

## Usage

### For Posture Recognition & Pointing at Objects

in `ultralytics_container`

```bash
rosrun mediapipe_ros arm_extension.py

rosrun mediapipe_ros arm_stability_detector.py 

rosrun mediapipe_ros cylinder_pointcloud_filter.py

rosrun mediapipe_ros pointcloud_to_image.py
```

### For Object Segmentation & Zero Shot Detection

in `ultralytics_container`

```bash
rosrun sam_segmentation sam_segmentation_node.py
```

in `clip_container`

```bash
rosrun clip clip_segment_matcher.py 

rosrun clip clip_query_input_node.py
```
