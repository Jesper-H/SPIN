from numpy import pi
from func import *
import time
from func import two_cameras, align, print_devices, rotation_matrix
from scipy.spatial.transform import Rotation
# from func import RobotClient
from robotcom import *
import subprocess
import tf.transformations as T

# subprocess.Popen('./robotcom.sh')
start = time.time()
wally = Camera('config_static_camera.yaml')
wally.save_point_cloud()
# wrist = Camera('config_wrist_camera.yaml')

# cl1 = o3d.io.read_point_cloud("911222060096.ply")
cl2 = o3d.io.read_point_cloud("030522071855.ply")
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.06636, [0, 0, 0])

print("pcl time:", time.time() - start)

# Calibrations
x = 0.02  # D415 distance to depth sensor
y = 0.119 - 0.0143  # onrobot frame - measured distance to lens
z = 0.0262001  # from old calibration. Only know value is less than 0.0345
rx, ry, rz, w = -0.5, -0.5, -0.5, 0.5  # clean angles from spreadsheet
manual_calib = [x, y, z, rx, ry, rz, w]
wrist_calib = [0.0312988, 0.102542, 0.0262001, -0.505622, -0.499951, -0.49172, 0.5026]  # [x y z rx ry rz w]
static_calib = [0.64129, 0.53881, 0.501983, 0.386933, 0.85708, -0.300722, -0.158947]
static_calib = [0.660612, 0.544702, 0.497378, -0.383301, -0.15792, 0.853749, -0.315047]
temp_tool0 = [0.25859, 0.11178, 0.39565, -0.16066, 0.92911, -0.18783, -0.27506]

"""
# transform wrist camera
# r = Rotation.from_quat(wrist_calib[3:]).as_matrix()
r = T.quaternion_matrix(wrist_calib[3:])[:3,:3]
t = np.asarray([wrist_calib[:3]]).T
tf_cam2onrobot_frame = rotation_matrix(x=pi / 2, y=-pi / 2)
r = r @ tf_cam2onrobot_frame
# r2 = Rotation.from_quat(temp_tool0[3:]).as_matrix()
r2 = T.quaternion_matrix(temp_tool0[3:])[:3,:3]
t2 = np.asarray([temp_tool0[:3]]).T

cl1.rotate(r, [0, 0, 0])
cl1.translate(t)
cl1.rotate(r2, [0, 0, 0])
cl1.translate(t2)
"""

# transform static camera
# r = Rotation.from_quat(static_calib[3:]).as_matrix()
r = T.quaternion_matrix(static_calib[3:])[:3,:3]
t = np.asarray([static_calib[:3]]).T
tf_cam2base_frame = rotation_matrix(x=-pi/2, z=pi/2)
# cl2.translate([-0.015, 0, 0]) # adjust 15mm to align with rgb
cl2.rotate(tf_cam2base_frame.T, [0, 0, 0])
cl2.rotate(r, [0, 0, 0])
cl2.translate(t)


# cl2 += align(cl2, cl1)
start = time.time()
box_list = segment(cl2)  # Segment and get bounding box
assert box_list, 'no target box detected'
print('sagment time:', time.time() - start)
*box_list, target_box = sorted(box_list, key=lambda box: box.volume())
target_box.color = [.0, 1.0, .0]
#o3d.visualization.draw_geometries([cl2, target_box, coordinate_frame])


robot = RobotCom()
time.sleep(0.1) # TODO, assert box is published instead of this lazy hack
box_pose, box_size = bounding_box_to_pose2(target_box)
robot.add_box(box_pose, box_size, 'target_box')
for i, box in enumerate(box_list):
    robot.add_box(*bounding_box_to_pose2(box), f'box_{i}')
o3d.visualization.draw_geometries([cl2, target_box, coordinate_frame])

def pick_n_place(client):
    """grab target box and place on marked spot"""
    target_pose = bounding_box_to_pose(target_box)
    target_elevation = target_pose[2]
    target_pose[2] += target_elevation + 0.26
    mark = [0.35722, 0.22812, 0.27, -0.4283235994640392, -0.9034616683200799, 0.016461124976137623, 0.004993934619837831]
    mark[2] += target_elevation
    end = [0.35722, 0.22812, 0.36, -0.4283235994640392, -0.9034616683200799, 0.016461124976137623, 0.004993934619837831]

    robot.gripper.set(140)
    robot.set_pose(target_pose)
    target_pose[2] -= min(target_elevation, 0.065) + 0.01  # 65mm grip surface
    robot.set_pose(target_pose)
    robot.gripper.set(0)
    assert client(['attach', ['target_box'], {}]), 'failed to grasp target'
    target_pose[2] += min(target_elevation, 0.065) + 0.01
    robot.set_pose(target_pose)
    robot.set_pose(mark)
    force = client(['move_gently', [2.0, [0.0, 0.0, -0.03], 0.01], {}])
    client(['detach', ['target_box'], {}])
    robot.gripper.set(140)

    client(['set_pose', [end], {}])
    client.close()


def pose4object(bounding_box):
    """ Suggests a suitable robot grasping poses based on the normals of the input bounding box """
    t_vec = bounding_box.center # translation vector to box center
    distance = np.linalg.norm(t_vec)
    gripper_length = 0.247
    robot_length = 0.5
    gripper_width = 0.140 - 0.02
    epsilon = 0.001
    plane_normal = np.array([0, 0, 1]) # surface normal of grasping plane
    min_plane = 0.1 # lower bound
    max_plane = target_box.center[2]+gripper_length-epsilon # upper bound
    cos_of_vectors = lambda u,v: (u@v) / (u@u * v@v)**0.5
    
    grasp_plane = max_plane
    if np.linalg.norm(plane_normal*grasp_plane+t_vec) > robot_length:
      # if max plane (bird eye view) pose is too far away:
      # calculate dynamic grasping plane based on distance (far away -> low grasp angle)
      # solve square area with heron's formula then divide by base to find plane hight
      s = (robot_length + gripper_length + distance) / 2 # center of triangle
      a = 4*s*(s-robot_length)*(s-gripper_length)*(s-distance) # triangle²*2² -> square²
      dynamic_plane = a**0.5 / distance if a > 0 else 0
      # get component orthogonal to plane using sin² + cos² = 1
      dynamic_plane *= (1-cos_of_vectors(plane_normal,t_vec)**2)**0.5
      grasp_plane = max(min_plane, min(dynamic_plane, max_plane)) # clamp
    
    x, y, z = bounding_box_to_normals(bounding_box)
    points = []
    for box_normal in [x, y, z]:
        if abs(cos_of_vectors(plane_normal, box_normal)) > .5: 
            continue  # skip gripping angles past 60 degrees
        side_length = np.linalg.norm(box_normal)
        if side_length >= gripper_width : 
            continue  # skip sides greater than gripper width
        
        n1 = plane_normal*grasp_plane  # gripping plane
        n1 = n1 - plane_normal * (plane_normal @ t_vec.T) # hight relative to object
        n2 = np.cross(n1, box_normal) # vector parallel to plane
        n2 /= np.linalg.norm(n2) # ...normalized
        length_of_vector = (gripper_length**2 - n1 @ n1.T) ** 0.5  # l = +-sqrt( (r²-n1²))
        n2 = length_of_vector * n2  # de-normalize
        point1 = n1 + n2 + t_vec
        point2 = n1 - n2 + t_vec
        points += [point1, point2]
    if not points:
        return []

    rotatoes = []
    for p in points:
        nz = t_vec - p                   # normal for grippers z-axis
        nx = np.cross(plane_normal, nz)  # normal for grippers x-axis
        ny = np.cross(nz, nx)            # normal for grippers y-axis
        nx, ny, nz = [n / np.linalg.norm(n) for n in [nx, ny, nz]]
        rotation = rotation_matrix_from_vector_matrix(nx, ny, nz)
        rotatoes += [rotation]
    
    sorted_list = [*sorted(zip(points, rotatoes), key=lambda x: np.linalg.norm(x[0]))]
    points, rotatoes = zip(*sorted_list)
    rotations = [np.eye(4) for _ in range(len(rotatoes))]
    for r4,r3 in zip(rotations, rotatoes): # cast from R³ -> R⁴
      r4[:3,:3] = r3
    
    r_vec = [T.quaternion_from_matrix(r) for r in rotations] # cast matrix to quat form
    poses = [[*p, *r] for p, r in zip(points, r_vec)] # append rotation to translation
    poses = [[*map(float, p)] for p in poses] # cast to float for compatability
    return poses


def pose2mesh(pose):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05, -np.array(pose[:3]))
    r = T.quaternion_matrix(pose[3:])[:3,:3]
    mesh.rotate(r, pose[:3])
    return mesh

    
grasp_poses = pose4object(target_box)
assert grasp_poses, 'No poses found'
grasp_pose, *_ = grasp_poses
mesh_list = [pose2mesh(pose) for pose in grasp_poses]
mesh_list[0].vertex_colors = o3d.utility.Vector3dVector(np.eye(3))

o3d.visualization.draw_geometries([cl2, *box_list, target_box, coordinate_frame, *mesh_list])

robot.gripper.set(140)
robot.set_pose(grasp_pose)
robot.gripper.set(0)
assert robot.attach('target_box'), 'failed to grasp target'
grasp_pose[2] += 0.1
robot.set_pose(grasp_pose)
robot.detach('target_box')
robot.gripper.set(140)
# pick_n_place(client)
