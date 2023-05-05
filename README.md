# SPIN
Semantic Perception (SPIN) focus on sensor fusion and robot integration. Robot used in this project is a UR3E with onrobot eyes camera and robotiq 2f140 gripper.

## Quickstart
1. **Clone this repo**: git clone https://github.com/Jesper-H/SPIN.git

1. **Install ros**: [ubuntu link](https://wiki.ros.org/noetic/Installation/Ubuntu) 

1. **Install catkin and wstool** as described [here](https://ros-planning.github.io/moveit_tutorials/doc/getting_started/getting_started.html). Set up "workspace" as your workspace, download MoveIt source and compile it.

1. **Run environment**: roslaunch ur3e_robotiq140_eyes_moveit_config env.launch use_cam:=false

1. **Activate external control**. As instructed [here](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/install_urcap_e_series.md) 

1. **Run code**: python3 pointcloudtools/main.py
