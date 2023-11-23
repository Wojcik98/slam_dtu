import ctypes
import struct
import time
from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import Point, Pose, PoseWithCovariance, Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points, read_points_numpy
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import copy


class GlobalLocalization(Node):
    def __init__(self):
        super().__init__("global_localization")

        # read map path from parameter
        self.declare_parameter("map_path", "")
        self.map_path = (
            self.get_parameter("map_path").get_parameter_value().string_value
        )
        # read visualize from parameter
        self.declare_parameter("visualize", False)
        self.visualize = (
            self.get_parameter("visualize").get_parameter_value().bool_value
        )
        # read distance_threshold from parameter
        self.declare_parameter("distance_threshold", 2.0)
        self.distance_threshold = (
            self.get_parameter("distance_threshold").get_parameter_value().double_value
        )
        # read angle_threshold from parameter
        self.declare_parameter("angle_threshold", np.deg2rad(60))
        self.angle_threshold = (
            self.get_parameter("angle_threshold").get_parameter_value().double_value
        )
        # read fitness_threshold from parameter
        self.declare_parameter("fitness_threshold", 0.7)
        self.fitness_threshold = (
            self.get_parameter("fitness_threshold").get_parameter_value().double_value
        )

        self.non_3d_angle_threshold = np.deg2rad(5)

        if self.visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map: o3d.geometry.PointCloud = o3d.io.read_point_cloud(self.map_path)
        self.map.estimate_normals()

        self.rgbd = o3d.geometry.PointCloud()

        if self.visualize:
            self.vis.add_geometry(self.map)
            self.vis.add_geometry(self.rgbd)

        self.pose_estimate = np.array([[0.0], [0.0], [0.0]])
        self.latest_pcd_msg: Optional[PointCloud2] = None

        # initialize subscribers
        self.odom_sub = self.create_subscription(Odometry, "odom_in", self.odom_cb, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, "points", self.pointcloud_cb, 10
        )

        # initialize publishers
        self.pose_pub = self.create_publisher(Odometry, "odom_out", 10)

        self.timer = self.create_timer(1.0, self.timer_cb)

    def odom_cb(self, msg: Odometry) -> None:
        """Save the latest pose estimate."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

        self.pose_estimate = np.array([[x], [y], [theta]])

    def pointcloud_cb(self, msg: PointCloud2) -> None:
        """Save the latest pointcloud."""
        self.latest_pcd_msg = msg

        if self.visualize:
            self.vis.poll_events()
            self.vis.update_renderer()

    def timer_cb(self) -> None:
        self.get_logger().info(f"===========================================")

        if self.latest_pcd_msg is None:
            return

        start = time.time()
        abs_start = start

        try:
            rgbd_to_base_link = self.get_transform_matrix(
                "base_link", self.latest_pcd_msg.header.frame_id, rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform base_link to "
                f"{self.latest_pcd_msg.header.frame_id}: {ex}"
            )
            return

        # self.get_logger().info(
        #     f"    getting transform took {time.time() - start:.3f} seconds"
        # )
        start = time.time()

        # convert pointcloud message to numpy open3d pointcloud
        # raw_data = read_points(
        #     self.latest_pcd_msg, skip_nans=True, field_names=("x", "y", "z")
        # )  # 1D (n, ) np.array holding tuples of (x, y, z, rgb)
        points = read_points_numpy(
            self.latest_pcd_msg, skip_nans=True, field_names=("x", "y", "z")
        )  # 1D (n, ) np.array holding tuples of (x, y, z, rgb)

        # self.get_logger().info(
        #     f"    reading pointcloud took {time.time() - start:.3f} seconds"
        # )
        start = time.time()

        # transform pointcloud to base_link frame
        # points = np.hstack((points, np.ones((points.shape[0], 1))))
        # points = rgbd_to_base_link @ points.T
        # points = points[:3, :].T

        # self.get_logger().info(
        #     f"    transforming pointcloud took {time.time() - start:.3f} seconds"
        # )
        # start = time.time()

        self.rgbd.points = o3d.utility.Vector3dVector(points)
        # self.rgbd.colors = o3d.utility.Vector3dVector(colors)

        # self.get_logger().info(
        #     f"    constructing took {time.time() - start:.3f} seconds"
        # )
        start = time.time()

        self.rgbd.transform(rgbd_to_base_link)
        # self.get_logger().info(
        #     f"    transforming pointcloud took {time.time() - start:.3f} seconds"
        # )
        start = time.time()

        # downsample rgbd
        rgbd = self.rgbd.voxel_down_sample(voxel_size=0.05)

        # self.get_logger().info(
        #     f"    downsampling took {time.time() - start:.3f} seconds"
        # )
        start = time.time()

        rgbd.estimate_normals()

        # self.get_logger().info(
        #     f"    estimating normals took {time.time() - start:.3f} seconds"
        # )
        start = time.time()

        # initial guess
        T_init = np.eye(4)
        T_init[:3, :3] = Rotation.from_euler("z", self.pose_estimate[2, 0]).as_matrix()
        T_init[:2, 3] = self.pose_estimate[:2, 0]

        # crop map to square around initial guess
        # square_size = 100.0
        # map = self.map.crop(
        #     o3d.geometry.AxisAlignedBoundingBox(
        #         min_bound=(
        #             self.pose_estimate[0, 0] - square_size / 2,
        #             self.pose_estimate[1, 0] - square_size / 2,
        #             -np.inf,
        #         ),
        #         max_bound=(
        #             self.pose_estimate[0, 0] + square_size / 2,
        #             self.pose_estimate[1, 0] + square_size / 2,
        #             np.inf,
        #         ),
        #     )
        # )
        # map = self.map

        # crop floor
        floor_height = 0.1
        map = self.map.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(
                    -np.inf,
                    -np.inf,
                    -np.inf,
                ),
                max_bound=(
                    np.inf,
                    np.inf,
                    floor_height,
                ),
            )
        )

        rgbd.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(
                    -np.inf,
                    -np.inf,
                    -np.inf,
                ),
                max_bound=(
                    np.inf,
                    np.inf,
                    floor_height,
                ),
            )
        )

        # self.get_logger().info(f"    cropping took {time.time() - start:.3f} seconds")
        start = time.time()

        # registration
        reg = o3d.pipelines.registration.registration_icp(
            rgbd,
            map,
            max_correspondence_distance=0.5,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=10
            ),
        )
        # reg = o3d.pipelines.registration.registration_colored_icp(
        #     rgbd,
        #     map,
        #     max_correspondence_distance=0.5,
        #     init=T_init,
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        #         max_iteration=10
        #     ),
        # )

        if self.visualize:
            self.rgbd.transform(reg.transformation)
            self.vis.update_geometry(self.rgbd)
            self.vis.poll_events()
            self.vis.update_renderer()

        # self.get_logger().info(
        #     f"ICP fitness: {reg.fitness}, inlier_rmse: {reg.inlier_rmse}"
        # )
        # self.get_logger().info(f"Initial guess:\n{T_init}")
        # self.get_logger().info(f"Transformation:\n{reg.transformation}")

        # self.get_logger().info(
        #     f"    registration took {time.time() - start:.3f} seconds"
        # )
        start = time.time()

        position = reg.transformation[:2, 3]
        initial_position = T_init[:2, 3]
        distance = np.linalg.norm(position - initial_position)

        angles = Rotation.from_matrix(reg.transformation[:3, :3].copy()).as_euler("xyz")
        z_angle = angles[2]
        initial_z_angle = Rotation.from_matrix(T_init[:3, :3]).as_euler("xyz")[2]
        angle_diff = np.abs(z_angle - initial_z_angle)

        if (
            reg.fitness > self.fitness_threshold
            and distance < self.distance_threshold
            and angle_diff < self.angle_threshold
            and np.abs(angles[0]) < self.non_3d_angle_threshold
            and np.abs(angles[1]) < self.non_3d_angle_threshold
        ):
            # self.get_logger().info("Publishing registration")
            self.publish(reg, distance)
        else:
            self.get_logger().error("Not publishing registration")

        # self.get_logger().info(f"    publishing took {time.time() - start:.3f} seconds")

        self.get_logger().info(f"Total time: {time.time() - abs_start:.3f} seconds")

        if reg.fitness > self.fitness_threshold:
            self.get_logger().info(f"Registration fitness: {100 * reg.fitness:.3f} %")
        else:
            self.get_logger().warn(f"Registration fitness: {100 * reg.fitness:.3f} %")

        if distance < self.distance_threshold:
            self.get_logger().info(f"Registration distance: {distance:.3f} m")
        else:
            self.get_logger().warn(f"Registration distance: {distance:.3f} m")

        if angle_diff < self.angle_threshold:
            self.get_logger().info(f"Registration angle: {angle_diff:.3f} rad")
        else:
            self.get_logger().warn(f"Registration angle: {angle_diff:.3f} rad")

        if np.abs(angles[0]) < self.non_3d_angle_threshold:
            self.get_logger().info(f"Registration angle x: {angles[0]:.3f} rad")
        else:
            self.get_logger().warn(f"Registration angle x: {angles[0]:.3f} rad")

        if np.abs(angles[1]) < self.non_3d_angle_threshold:
            self.get_logger().info(f"Registration angle y: {angles[1]:.3f} rad")
        else:
            self.get_logger().warn(f"Registration angle y: {angles[1]:.3f} rad")

        # self.get_logger().warn(f"Registration distance: {distance:.3f} m")
        # self.get_logger().warn(f"Registration angle: {angle_diff:.3f} rad")
        # self.get_logger().info(f"Registration: {reg}")
        self.get_logger().info(f"Registration RMSE: {reg.inlier_rmse:.3f}")

        # stop = time.time()
        # elapsed = stop - start

    # def get_points_and_colors(self, msg: PointCloud2) -> tuple[np.ndarray, np.ndarray]:
    #     """Convert a pointcloud message to numpy arrays of points and colors."""
    #     raw_data = read_points(msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))
    #     points = np.array(list(np.array(list(point)) for point in raw_data))
    #     colors_raw = points[:, 3].squeeze()
    #     points = points[:, :3]
    #     colors = np.zeros((colors_raw.shape[0], 3))

    #     for i in range(colors_raw.shape[0]):  # TODO optimize
    #         s = struct.pack(">f", colors_raw[i])
    #         gle = struct.unpack(">l", s)[0]
    #         rgb = ctypes.c_uint32(gle).value
    #         r = (rgb & 0x00FF0000) >> 16
    #         g = (rgb & 0x0000FF00) >> 8
    #         b = rgb & 0x000000FF
    #         colors[i, :] = np.array([r, g, b]) / 255.0

    #     return points, colors

    def publish(
        self,
        registration: o3d.pipelines.registration.RegistrationResult,
        distance: float,
    ) -> None:
        """Publish the registration result as an Odometry message."""
        transformation = registration.transformation
        fitness = registration.fitness

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        x = transformation[0, 3]
        y = transformation[1, 3]
        R = transformation[:3, :3].copy()
        theta = Rotation.from_matrix(R).as_euler("xyz")[2]
        q = Rotation.from_euler("z", theta).as_quat()

        # Calculate covariance matrix
        # error = 0.1 * (1 / (fitness + 1e-6))
        error = distance
        cov = np.diag([1, 1, 0, 0, 0, 1]) * error
        self.get_logger().info(f"Error: {error}")

        pose = PoseWithCovariance(
            pose=Pose(
                position=Point(x=x, y=y, z=0.0),
                orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
            ),
            covariance=cov.ravel(),
        )

        # publish
        self.pose_pub.publish(
            Odometry(
                header=header,
                child_frame_id="base_link",
                pose=pose,
            )
        )

    def get_transform_matrix(
        self, source: str, target: str, time: Optional[rclpy.time.Time] = None
    ) -> np.ndarray:
        if time is None:
            time = rclpy.time.Time()

        tf_msg = self.tf_buffer.lookup_transform(source, target, time)

        translation = tf_msg.transform.translation
        rotation = tf_msg.transform.rotation

        T = np.eye(4)
        T[:3, 3] = np.array([translation.x, translation.y, translation.z])
        T[:3, :3] = Rotation.from_quat(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        ).as_matrix()

        return T

    def __del__(self):
        self.vis.destroy_window()


def main(args=None):
    rclpy.init(args=args)

    global_localization_node = GlobalLocalization()
    rclpy.spin(global_localization_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
