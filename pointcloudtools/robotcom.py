import sys

import numpy as np
import rospy
import moveit_commander
import yaml
import moveit_msgs.msg
import geometry_msgs.msg
import socket
import time
from robotiq_urcap_control.gripper import RobotiqGripper
from copy import deepcopy


class Grip(RobotiqGripper):

    def __init__(self, robot_ip='192.168.1.137'):
        super(Grip, self).__init__(robot_ip, gripper_sid=None)
        self.connect()
        self.speed = 255
        self.force = 0

    def obj_info(self):
        """
        0: 'Fingers in motion, no object detected',
        1: 'Opened grasp, object detected',
        2: 'Closed grasp, object detected',
        3: 'Fingers at specified location, no object detected'
        """
        return self.get_status().gOBJ

    def open(self, speed: int = None, force: int = None):
        speed = speed if speed else self.speed
        force = force if force else self.force
        self.move_and_wait_for_pos(0, speed, force)

    def close(self, speed: int = None, force: int = None):
        speed = speed if speed else self.speed
        force = force if force else self.force
        self.move_and_wait_for_pos(255, speed, force)

    def set(self, pos: float, speed: int = None, force: int = None):
        speed = speed if speed else self.speed
        force = force if force else self.force
        pos = 255 - int(pos / 140.0 * 255)
        self.move_and_wait_for_pos(pos, speed, force)


class RobotCom:
    def __init__(self):
        self.group = self._start_group_commander()
        self.gripper = Grip()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.eef_link = self.group.get_end_effector_link()
        self.robot = moveit_commander.RobotCommander()

        self.clean_table()
        self.force = np.zeros(3)
        self.poll_force()
        self.gripper_links = [
            'left_inner_finger',
            'left_inner_finger_pad',
            'left_inner_knuckle',
            'left_outer_finger',
            'left_outer_knuckle',
            'right_inner_finger',
            'right_inner_finger_pad',
            'right_inner_knuckle',
            'right_outer_finger',
            'right_outer_knuckle',
            'robotiq_arg2f_base_link',
            'robotiq_coupler', ]
        # TODO assert valid_start_state, f'robot state out of bounds {self.get_joint()}'

    def clean_table(self, height: float = 0.05):
        for obj in self.scene.get_known_object_names():
            self.scene.remove_world_object(obj)
        self.add_box([0, 0, -height / 2 - 0.001, 0, 0, 0, 1], [2, 2, height], 'table')

    def add_box(self, pose: list, size: list, name: str):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.robot.get_planning_frame()
        (box_pose.pose.position.x,
         box_pose.pose.position.y,
         box_pose.pose.position.z,
         box_pose.pose.orientation.x,
         box_pose.pose.orientation.y,
         box_pose.pose.orientation.z,
         box_pose.pose.orientation.w,) = pose
        self.scene.add_box(name, box_pose, size=size)
        
        start = time.time()
        timeout = 0
        while name not in self.scene.get_known_object_names():
            time.sleep(0.1)
            if time.time() - start > timeout:
                raise TimeoutError(f'box: {name}, did not publish within {timeout} seconds')

    def _start_group_commander(self, group_name:str='arm'):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('robot_communicator', anonymous=True)
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.group.set_max_velocity_scaling_factor(1.0)
        self.group.set_max_acceleration_scaling_factor(1.0)
        return self.group

    def plan(self, pause: bool = False):
        plan = self.group.plan()
        flag, plan, _, _ = plan
        assert flag, 'Planning failed'
        if pause:
            input('check plan on Rviz')
        return plan

    def set_joints(self, joint, stop: bool = True):
        self.group.set_joint_value_target(joint)
        p = self.plan()
        assert self.group.execute(p, wait=True), 'Planning failed'
        if stop:
            self.group.stop()
        return self.get_joint()

    def set_pose(self, pose, stop: bool = True):
        pose_goal = geometry_msgs.msg.Pose()
        (pose_goal.position.x,
         pose_goal.position.y,
         pose_goal.position.z,
         pose_goal.orientation.x,
         pose_goal.orientation.y,
         pose_goal.orientation.z,
         pose_goal.orientation.w) = pose
        self.group.set_pose_target(pose_goal)
        p = self.plan()
        assert self.group.execute(p, wait=True), 'Planning failed'
        if stop:
            self.group.stop()
        self.group.clear_pose_targets()
        return self.get_pose()

    def move_gently(self, max_force: float, direction: list, speed: float, target_collision: str = 'table'):
        """ max_force: in newton, direction: [x,y,z], speed: in meters per second """
        rate = 10
        snow_white = rospy.Rate(rate)
        abs_dir = np.linalg.norm(direction)
        direction = np.array(direction)
        dir_norm = direction / abs_dir
        step = direction / abs_dir
        step *= speed / rate
        abs_step = np.linalg.norm(step)
        peak_force = 0
        initial_force = self.get_force()
        self._attach2world(target_collision, self.gripper_links)

        while True:
            force = self.get_force()
            force -= initial_force
            force = force.dot(dir_norm)  # projected force
            peak_force = max(force, peak_force)
            if peak_force > max_force:
                break
            abs_dir -= abs_step
            if abs_dir <= 0:
                break

            pose = self.get_pose()
            pose[:3] = [p + s for p, s in zip(pose, step)]
            self.set_pose(pose, stop=False)
            snow_white.sleep()
        self.group.stop()
        self._detach_from_world(target_collision)
        return float(peak_force)

    def attach(self, object_name: str):
        i = self.gripper.obj_info()
        gripped = i == 1 or i == 2
        if not gripped:
            return gripped
        self.scene.attach_box(self.eef_link, object_name, touch_links=self.gripper_links)
        self.gripper_links += [object_name]
        return gripped

    def _attach2world(self, object_name: str, ignore_collision: list):
        self.scene.attach_box('base_link', object_name, touch_links=ignore_collision)

    def _detach_from_world(self, object_name: str):
        self.scene.remove_attached_object('base_link', object_name)

    def detach(self, object_name: str):
        if object_name not in self.gripper_links:
            print('Warning, requested detachment object not found')
        self.scene.remove_attached_object(self.eef_link, name=object_name)
        self.gripper_links = [gl for gl in self.gripper_links if gl != object_name]

    def _set_force(self, msg):
        f = msg.wrench.force
        f = np.array([f.x, f.y, f.z])
        self.force += (f - self.force) / 3  # low pass filtering

    def get_force(self):
        return deepcopy(self.force)

    def poll_force(self):
        rospy.Subscriber('/wrench', geometry_msgs.msg.WrenchStamped, self._set_force)

    def get_joint(self):
        return self.group.get_current_joint_values()

    def get_pose(self):
        poser = self.group.get_current_pose()
        t = poser.pose.position
        r = poser.pose.orientation
        return [t.x, t.y, t.z, r.x, r.y, r.z, r.w]

    def open_socket(self):
        host = '127.0.0.1'
        port = 65432
        word2func = {
            'set_joints': self.set_joints,
            'set_pose': self.set_pose,
            'get_joint': self.get_joint,
            'get_pose': self.get_pose,
            'set_gripper': self.gripper.set,
            'add_box': self.add_box,
            'move_gently': self.move_gently,
            'attach': self.attach,
            'detach': self.detach,
            'exit': exit,
        }

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen()
            s.setblocking(True)
            conn, addr = s.accept()
            with conn:
                def send_message(message):
                    msg = yaml.safe_dump(message)  # serialise
                    conn.sendall(msg.encode())  # encode and send

                while True:
                    data = conn.recv(1024).decode()
                    print('received', data)
                    command, args, kwargs = yaml.safe_load(data)
                    response = word2func[command](*args, **kwargs)
                    send_message(response)


if __name__ == "__main__":
    rob = RobotCom()
    rob.open_socket()
