FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# 1. 安装基础工具和 Python3
RUN apt update && apt install -y \
    lsb-release gnupg2 curl wget git locales \
    python3 python3-pip python3-venv python3-dev \
    build-essential

# 2. 配置 ROS Noetic 源
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# 3. 安装 ROS Noetic
RUN apt update && apt install -y ros-noetic-desktop-full \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool \
    python3-catkin-tools

RUN rosdep init && rosdep update

# 4. 安装 Python 包
RUN pip3 install --upgrade pip
RUN pip3 install rospkg catkin_pkg opencv-python numpy
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install git+https://github.com/openai/CLIP.git

# 5. ROS 环境变量
ENV ROS_DISTRO=noetic
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# 6. 工作目录
WORKDIR /catkin_ws

# 7. 拷贝源码
# COPY ./catkin_ws/src ./src

# 8. 构建 catkin_ws
# RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /root/catkin_ws && catkin_make"

# 9. 启动脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
