#!/bin/bash
source /opt/ros/noetic/setup.bash

if [ -f /root/catkin_ws/devel/setup.bash ]; then
    source /root/catkin_ws/devel/setup.bash
fi

exec "$@"
