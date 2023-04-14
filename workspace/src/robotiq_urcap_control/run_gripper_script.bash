#!/bin/bash
IP="192.168.1.137"
pack_path=$(rospack find robotiq_urcap_control)
python3 $pack_path/scripts/gripper_state_publisher.py --ip $IP
