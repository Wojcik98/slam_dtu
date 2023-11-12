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
from sensor_msgs_py.point_cloud2 import read_points
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class GlobalLocalization(Node):
    def __init__(self):
        super().__init__("global_localization")

        # read map path from parameter
        self.declare_parameter("map_path", "")
        self.map_path = (
            self.get_parameter("map_path").get_parameter_value().string_value
        )
        self.declare_parameter("visualize", False)
        self.visualize = (
            self.get_parameter("visualize").get_parameter_value().bool_value
        )

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
        if self.latest_pcd_msg is None:
            return

        start = time.time()

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

        # convert pointcloud message to numpy open3d pointcloud
        raw_data = read_points(
            self.latest_pcd_msg, skip_nans=True, field_names=("x", "y", "z")
        )  # 1D (n, ) np.array holding tuples of (x, y, z)
        points = np.array(
            list(np.array(list(point)) for point in raw_data)
        )  # 2D (n, 3) np.array holding [x, y, z] in each row

        # transform pointcloud to base_link frame
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = rgbd_to_base_link @ points.T
        points = points[:3, :].T

        self.rgbd.points = o3d.utility.Vector3dVector(points)
        self.rgbd.estimate_normals()

        # initial guess
        T_init = np.eye(4)
        T_init[:3, :3] = Rotation.from_euler("z", self.pose_estimate[2, 0]).as_matrix()
        T_init[:2, 3] = self.pose_estimate[:2, 0]

        # registration
        reg = o3d.pipelines.registration.registration_icp(
            self.rgbd,
            self.map,
            max_correspondence_distance=0.3,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

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

        self.publish(reg.transformation)

        stop = time.time()
        elapsed = stop - start
        self.get_logger().info(f"Registration time: {elapsed} seconds")

    def publish(self, transformation: np.ndarray) -> None:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        x = transformation[0, 3]
        y = transformation[1, 3]
        R = transformation[:3, :3].copy()
        theta = Rotation.from_matrix(R).as_euler("xyz")[2]
        q = Rotation.from_euler("z", theta).as_quat()

        pose = PoseWithCovariance(
            pose=Pose(
                position=Point(x=x, y=y, z=0.0),
                orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
            ),
            covariance=np.zeros(36).ravel(),
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
