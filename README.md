# Environment Setup

clip-env python3.12

sam2-env python3.8

## in sam2-env

nano /home/zby/anaconda3/envs/sam2-env/etc/conda/activate.d/env_vars.sh
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7"

nano /home/zby/anaconda3/envs/sam2-env/etc/conda/deactivate.d/env_vars.sh
unset LD_PRELOAD

## pub segmented image

when using RVIZ, use Image not Camera

<!-- ## in clip-env -->

<!-- nano /home/zby/anaconda3/envs/clip-env/etc/conda/activate.d/env_vars.sh
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7"

nano /home/zby/anaconda3/envs/clip-env/etc/conda/deactivate.d/env_vars.sh
unset LD_PRELOAD -->

## nomore use clip-env, created a new venv (clip_ros_env)

activate the venv: source ~/clip_ros_env/bin/activate

in venv: export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH