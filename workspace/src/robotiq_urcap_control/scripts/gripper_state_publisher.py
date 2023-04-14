#!/usr/bin python3

import argparse
import rospy

from robotiq_urcap_control.gripper import RobotiqGripper
# import getch
import time
import numpy as np

import sensor_msgs.msg


arg_fmt = argparse.RawDescriptionHelpFormatter
parser = argparse.ArgumentParser(
    formatter_class=arg_fmt, description='read code for description')
parser.add_argument(
    '--ip', required=True, type=str, help='robot IP')
args = parser.parse_args(rospy.myargv()[1:])

rospy.init_node("gripper_position")

global gripper
while True:
  try:
    gripper = RobotiqGripper(args.ip)
    gripper.connect()
  except:
    print('Failed to connect to gripper at ip:', args.ip)
    time.sleep(2)
  else:
    break
pub = rospy.Publisher('joint_states', sensor_msgs.msg.JointState, queue_size=5)

while not rospy.is_shutdown():
  pos = gripper.get_current_position()
  pos *= .77 / 255 # rescale to moveit format (unknown number)
  msg = sensor_msgs.msg.JointState()
  msg.header.stamp = rospy.Time.now()
  msg.name.append('finger_joint')
  msg.position = [pos]
  msg.velocity = [0]
  msg.effort = [0]
  pub.publish(msg)
  rospy.sleep(0.1)

