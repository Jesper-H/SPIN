#!/bin/bash
PYTHONPATH=/data/jesper/workspace/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages:/usr/lib/python3/dist-packages
PATH=/opt/ros/noetic/bin:/usr/local/cuda/bin:/opt/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/data/jesper/.local/bin:/data/jesper/.local/bin

export PYTHONPATH
export PATH

python3 ./robotcom.py
