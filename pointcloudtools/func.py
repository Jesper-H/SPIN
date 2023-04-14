from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import socket
import time
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import yaml
from compas.geometry import local_axes, world_to_local_coordinates_numpy, local_to_world_coordinates_numpy, \
    transform_points_numpy, Frame, Transformation
from compas.geometry.bbox.bbox import bounding_box
from compas.numerical import pca_numpy
from numpy import asarray, amax, amin
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
from copy import deepcopy
import tf.transformations as T


def print_devices():
    ctx = rs.context()
    if len(ctx.devices) <= 0:
        print("No Intel Device connected")
        return
    for d in ctx.devices:
        print('Found device:',
              d.get_info(rs.camera_info.name),
              d.get_info(rs.camera_info.serial_number))


class Camera:
    """Camera handler class. Loads configs and does minor functions"""

    def __init__(self, config: str):
        with open(config) as cfg:
            self.cfg = yaml.safe_load(cfg)
        self.context = rs.context()
        self.pipeline, self.config = self.open_stream()
        if 'json_file_path' in self.cfg.keys():
            self.read_json_settings()  # use advanced settings

    def open_stream(self):
        """Opens a stream on this camera object"""
        # create device config
        config = rs.config()
        config.enable_device(str(self.cfg['id']))
        config.enable_stream(rs.stream.depth, self.cfg['depth_width'], self.cfg['depth_height'],
                             rs.format.z16, self.cfg['depth_fps'])
        config.enable_stream(rs.stream.color, self.cfg['color_width'], self.cfg['color_height'],
                             rs.format.rgb8, self.cfg['color_fps'])

        # open a stream
        pipeline = rs.pipeline(self.context)
        pipeline.start(config)
        return pipeline, config

    def read_json_settings(self):
        """Loads advanced mode json config into active camera"""

        """
        def get_advanced_device(dev_id):
            "Polls all connected devices for specified id that also supports advanced mode"
            ds5_product_ids = {"0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03",
                               "0B07", "0B3A", "0B5C"}
            devices = self.context.query_devices()
            devices = [d for d in devices if d.get_info(rs.camera_info.product_id) in ds5_product_ids]
            assert devices, 'no advanced mode compatible devices found'
            devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == dev_id]
            assert devices, f'camera id {dev_id} not found'
            return rs.rs400_advanced_mode(devices[0])"""

        def make_device_advanced():
            """Returns connected device as advanced device. Assumes device supports it"""
            dev = self.pipeline.get_active_profile().get_device()
            return rs.rs400_advanced_mode(dev)

        # device = get_advanced_device(self.cfg['id'])
        device = make_device_advanced()
        while not device.is_enabled():
            if not device.is_enabled():
                print('failed to set advanced mode, retrying...')
            device.toggle_advanced_mode(True)
            time.sleep(5)  # wait for reboot
            # The 'dev' object will become invalid, so we need to initialize it again
            device = make_device_advanced()
            if device.is_enabled():
                print('advanced mode active')

        with open('defaultConfig.json') as default_cfg:
            default = json.load(default_cfg)  # grab default settings

        with open(self.cfg['json_file_path']) as adv_cfg:
            json_cfg = json.load(adv_cfg)
            default['parameters'].update(json_cfg['parameters'])  # update settings
            default['device'].update(json_cfg['device'])
            json_cfg = default
            # The C++ JSON parser requires double-quotes for the json object, so we need
            # to replace the single quote of the pythonic json to double-quotes
            json_cfg = str(json_cfg).replace("'", '\"')
            device.load_json(json_cfg)

    def stream(self):
        """Starts a stream on active camera"""
        # make window
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RealSense', 640, 480)

        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Initialize colorizer class
            colorizer = rs.colorizer()
            # Convert images to numpy arrays, using colorizer to generate appropriate colors
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Stack both images horizontally
            images = np.hstack((color_image, depth_image))

            # Show images
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    def save_point_cloud(self):
        frame = self.pipeline.wait_for_frames()
        aligner = rs.align(rs.stream.color)
        aligned_frame = aligner.process(frame)
        depth_frame = aligned_frame.get_depth_frame()
        color_frame = aligned_frame.get_color_frame()
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        name = self.cfg['id']
        print(f'Saving to {name}... ', end='')
        points.export_to_ply(f'{name}.ply', color_frame)
        print('done')

    def get_depth_scale(self):
        profile = self.pipeline.get_active_profile()
        return profile.get_device().first_depth_sensor().get_depth_scale()

    def get_ply(self):
        """Broken function, missing intrinsics"""
        frame = self.pipeline.wait_for_frames()
        depth_data = frame.get_depth_frame().data
        depth_data = np.asarray(depth_data)

        w, h = depth_data.shape
        index = [(i % w, i // h) for i in range(w * h)]
        xyz = [(float(x), float(y), depth_data[x, y]) for x, y in index if depth_data[x, y]]
        xyz = np.asarray(np.array(xyz))
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(xyz)
        return pcl


def two_cameras():
    wrist = Camera('config_wrist_camera.yaml')
    wall = Camera('config_static_camera.yaml')
    time.sleep(2)  # let a few frames pass
    wrist.save_point_cloud()
    wall.save_point_cloud()


def align(cl1, cl2):
    """
    Using ICP to align the two roughly aligned points clouds
    :param cl1: point cloud
    :param cl2: point cloud
    :return: one aligned point cloud
    """
    trans = np.eye(4)
    threshold = 0.02
    reg = o3d.pipelines.registration.registration_icp(cl2, cl1, threshold,
                                                      trans,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                      o3d.pipelines.registration.ICPConvergenceCriteria(10000))
    cl2.transform(reg.transformation)
    return cl2


def segment(pc):
    """ Code to segment and return 6D pose of objects on the robots table """
    # crop cloud to save performance
    min_bound = np.array([-0.75, -0.75, -0.005])
    max_bound = np.array([0.75, 0.75, 0.75])
    vec = o3d.utility.Vector3dVector([min_bound, max_bound])
    big_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vec)
    big_box.color = [0., 0., 0.]
    pc = pc.crop(big_box)

    # remove the robot base
    points = np.array(pc.points)
    robo_index = np.where(np.linalg.norm(points, axis=1) < 0.14)[0]
    pc = pc.select_by_index(robo_index, invert=True)

    # remove table
    plane_model, table_index = pc.segment_plane(0.01, 3, 100)
    table_cloud = pc.select_by_index(table_index)
    table = o3d.geometry.AxisAlignedBoundingBox.create_from_points(table_cloud.points)
    table.color = [.0, .0, 1.0]
    pc = pc.select_by_index(table_index, invert=True)  # Remove table

    # remove outliers
    points = np.array(pc.points)
    min_samples = max(50, int(points.shape[0] * 0.001))
    db = DBSCAN(eps=0.009, min_samples=min_samples).fit(points)

    # Get bounding boxes of clusters
    table.translate([0., 0., 0.01])  # nudge table up to intersect with items on table
    box_list = []
    for label in np.unique(db.labels_):
        if label == -1:  # if outlier skip
            continue

        cluster = pc.select_by_index(np.where(db.labels_ == label)[0])

        intersecting_table = table.get_point_indices_within_bounding_box(cluster.points)
        if not intersecting_table:
            continue

        bbox = oriented_bounding_box_numpy(cluster.points)  # get fit-ed bounding box
        bb = o3d.utility.Vector3dVector(bbox)
        bb = o3d.geometry.OrientedBoundingBox.create_from_points(bb)
        bb.color = [1.0, .0, .0]
        box_list += [bb]

    # drop low volume objects
    min_volume = 0.01 ** 3  # one cube centimeter
    box_list = [b for b in box_list if b.volume() > min_volume]

    return box_list


def oriented_bounding_box_numpy(points):
    r"""Compute the oriented minimum bounding box of a set of points in 3D space.

    Parameters
    ----------
    points : array_like[point]
        XYZ coordinates of the points.

    Returns
    -------
    list[[float, float, float]]
        XYZ coordinates of 8 points defining a box.

    Raises
    ------
    AssertionError
        If the input data is 2D.
    QhullError
        If the data is essentially 2D.

    Notes
    -----
    The *oriented (minimum) bounding box* (OBB) of a given set of points
    is computed using the following procedure:

    1. Compute the convex hull of the points.
    2. For each of the faces on the hull:

       1. Compute face frame.
       2. Compute coordinates of other points in face frame.
       3. Find "peak-to-peak" (PTP) values of point coordinates along local axes.
       4. Compute volume of box formed with PTP values.

    3. Select the box with the smallest volume.

    Examples
    --------
    Generate a random set of points with
    :math:`x \in [0, 10]`, :math:`y \in [0, 1]` and :math:`z \in [0, 3]`.
    Add the corners of the box such that we now the volume is supposed to be :math:`30.0`.

    points = np.random.rand(10000, 3)
    bottom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    top = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    points = np.concatenate((points, bottom, top))
    points[:, 0] *= 10
    points[:, 2] *= 3

    Rotate the points around an arbitrary axis by an arbitrary angle.

    from compas.geometry import Rotation
    >>> from compas.geometry import transform_points_numpy
    R = Rotation.from_axis_and_angle([1.0, 1.0, 0.0], 0.3 * 3.14159)
    points = transform_points_numpy(points, R)

    Compute the volume of the oriented bounding box.

    from compas.geometry import length_vector, subtract_vectors, close
    bbox = oriented_bounding_box_numpy(points)
    a = length_vector(subtract_vectors(bbox[1], bbox[0]))
    b = length_vector(subtract_vectors(bbox[3], bbox[0]))
    c = length_vector(subtract_vectors(bbox[4], bbox[0]))
    close(a * b * c, 30.)
    """
    points = asarray(points)
    n, dim = points.shape

    assert 2 < dim, "The point coordinates should be at least 3D: %i" % dim

    points = points[:, :3]

    # noinspection PyBroadException
    try:
        ConvexHull(points)
    except Exception:
        return oabb_numpy(points)

    hull = ConvexHull(points)
    volume = None
    bbox = []

    # this can be vectorised!
    for simplex in hull.simplices:
        a, b, c = points[simplex]
        uvw = local_axes(a, b, c)
        xyz = points[hull.vertices]
        frame = [a, uvw[0], uvw[1]]
        rst = world_to_local_coordinates_numpy(frame, xyz)
        rmin, smin, tmin = amin(rst, axis=0)
        rmax, smax, tmax = amax(rst, axis=0)
        dr = rmax - rmin
        ds = smax - smin
        dt = tmax - tmin
        v = dr * ds * dt

        if volume is None or v < volume:
            bbox = [
                [rmin, smin, tmin],
                [rmax, smin, tmin],
                [rmax, smax, tmin],
                [rmin, smax, tmin],
                [rmin, smin, tmax],
                [rmax, smin, tmax],
                [rmax, smax, tmax],
                [rmin, smax, tmax],
            ]
            bbox = local_to_world_coordinates_numpy(frame, bbox)
            volume = v

    return bbox


def oabb_numpy(points):
    """Oriented bounding box of a set of points.

    Parameters
    ----------
    points : array_like[point]
        XYZ coordinates of the points.

    Returns
    -------
    list[[float, float, float]]
        XYZ coordinates of 8 points defining a box.

    """
    origin, (xaxis, yaxis, zaxis), values = pca_numpy(points)
    frame = Frame(origin, xaxis, yaxis)
    world = Frame.worldXY()
    x = Transformation.from_frame_to_frame(frame, world)
    points = transform_points_numpy(points, x)
    bbox = bounding_box(points)
    bbox = transform_points_numpy(bbox, x.inverse())
    return bbox


def rotation_matrix(x: float = None, y: float = None, z: float = None):
    from functools import reduce
    import operator
    from numpy import sin, cos

    tf = []
    if x:
        xx = [[1, 0, 0],
              [0, cos(x), -sin(x)],
              [0, sin(x), cos(x)]]
        tf += [np.asarray(xx)]
    if y:
        yy = [[cos(y), 0, sin(y)],
              [0, 1, 0],
              [-sin(y), 0, cos(y)]]
        tf += [np.asarray(yy)]
    if z:
        zz = [[cos(z), -sin(z), 0],
              [sin(z), cos(z), 0],
              [0, 0, 1]]
        tf += [np.asarray(zz)]
    return reduce(operator.matmul, tf, np.eye(3))


def bounding_box_to_pose(bounding_box):
    points = bounding_box.get_box_points()
    points = np.array(points)
    (p, *points) = points
    diff = [p - pp for pp in points]
    normal = min(diff, key=lambda x: np.linalg.norm(x))
    normal /= np.linalg.norm(normal)
    normal = np.array(normal)

    tool0_axis = np.array([1.0, .0, .0])  # part of tool we want aligned with box
    r_vec = rotation_matrix_from_vectors(normal, tool0_axis)
    # r_vec = Rotation.from_matrix(r_vec).as_quat()
    r_vec = bird_eye_pose(normal)
    t_vec = bounding_box.get_center()
    pose = [*t_vec, *r_vec]
    return [float(p) for p in pose]


def bird_eye_pose(normal):
    normal[2] = 0 # project to z plane
    normal /= np.linalg.norm(normal)
    axis = np.array([0., 1., 0.]) # reference axis for rotation (y)
    axis /= np.linalg.norm(axis)
    dot = normal.dot(axis)
    angle = np.arccos(dot)
    r_vec = np.eye(4)
    r_vec[:3,:3] = rotation_matrix(y=np.pi, z=angle+np.pi/2)
    # r_vec = Rotation.from_matrix(r_vec).as_quat()
    r_vec = T.quaternion_from_matrix(r_vec)
    return r_vec


def bounding_box_to_normals(bounding_box):
    points = bounding_box.get_box_points()
    points = np.array(points)
    p, *points = points
    diff = [p - pp for pp in points]
    diff = [*sorted(diff, key=np.linalg.norm)]
    x, y, z1, z2, *_ = diff # normals should be in this order after sort
    z_normal = np.cross(x, y) # we get one "fake" z so check orthogonality
    z_normal /= np.linalg.norm(z_normal)
    threshold = 0.9 # 90% match is enough, other vector is multiplied with cos(90)
    z = z1 if abs(z1.dot(z_normal)) > np.linalg.norm(z1) * threshold else z2
    z *= np.sign(z@z_normal.T) # flip z if wrong direction
    return x, y, z


def bounding_box_to_pose2(bounding_box):
    x, y, z = bounding_box_to_normals(bounding_box)
    rotation = np.eye(4)
    rotation[:3,:3] = rotation_matrix_from_vector_matrix(x, y, z)
    # r_vec = Rotation.from_matrix(rotation).as_quat()
    r_vec = T.quaternion_from_matrix(rotation)
    t_vec = bounding_box.get_center()
    size = [float(np.linalg.norm(a)) for a in [x, y, z]]
    pose = [*t_vec, *r_vec]
    pose = [float(p) for p in pose]
    return pose, size


def rotation_matrix_from_vector_matrix(x, y, z):
    """ takes 3 orthogonal vectors and computes the rotation needed to align with them """
    x_axis, y_axis, z_axis = np.eye(3)
    rotation = np.eye(3)
    z *= np.sign(np.cross(x,y)@z) # flip z sign if needed
    for robot_axis, box_axis in zip([x_axis, y_axis, z_axis], [x, y, z]):
        box_axis = box_axis@rotation # update axis so we don't repeat old rotations
        rot = rotation_matrix_from_vectors(robot_axis, box_axis)
        rotation = rotation@rot
    aligned = box_axis@robot_axis.T > (1e-6-1) * np.linalg.norm(box_axis) * np.linalg.norm(robot_axis)
    assert aligned, 'solution not found, check axis order or signs'
    return rotation


def rotation_matrix_from_vectors(target_vec, source_vec):
    """ in: 2 numpy vectors (1,3). out: numpy rotation matrix (3,3) """
    target_vec = target_vec / np.linalg.norm(target_vec)
    source_vec = source_vec / np.linalg.norm(source_vec)
    dot = target_vec.dot(source_vec.T)
    if dot < -0.9999: # avoid division by zero
        return -np.eye(3)
    x = np.cross(target_vec, source_vec)
    z = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])  # Skew-symmetric cross product matrix
    return np.eye(3) + z + z @ z / (1 + dot)


class RobotClient:
    """ Small tcp client in case robot communicator is hosted externally """
    def __init__(self, host: str = "127.0.0.1", port: int = 65432):
        self.host = host  # The server's hostname or IP address
        self.port = port  # The port used by the server
        self.s = self._connect()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        return s

    def __call__(self, *args, **kwargs):
        assert len(args) == 1, 'method takes only one argument'
        return self.send_message(*args)

    def send_message(self, msg: tuple):
        """ 
        Passes a message in the form (function name, args, kwargs) to server.
        Inputs should be only base python types.
        """
        assert type(msg) == tuple or type(msg) == list, 'message must be (str, [], {}) format'
        msg = yaml.dump(msg)  # serialise
        self.s.sendall(msg.encode())  # encode and send
        data = self.s.recv(1024).decode()
        return yaml.safe_load(data)

    def close(self):
        self.send_message(['exit', [], {}])
        self.s.close()
