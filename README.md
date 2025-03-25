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