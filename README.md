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

### Tiago Navigation & Map

```bash
rosrun map_server map_server src/carry_my_luggage/ics_map/map.yaml

rosrun rviz rviz -d'rospack find tiago_2dnav'/config/rviz/navigation.rviz

rosservice call /pal_map_manager/change_map "input: '2025-01-30_124458'"
```

### Tiago RViz

if you are using rosbag

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
/xtion/rgb/image_raw \
/xtion/rgb/camera_info \
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
