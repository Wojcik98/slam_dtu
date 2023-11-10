import time
from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    Quaternion,
    Twist,
    TwistWithCovariance,
    Vector3,
)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from std_msgs.msg import Header


class GlobalLocalization(Node):
    def __init__(self):
        super().__init__("global_localization")

        # read map path from parameter
        self.declare_parameter("map_path", "")
        self.map_path = (
            self.get_parameter("map_path").get_parameter_value().string_value
        )

        self.map: o3d.geometry.PointCloud = o3d.io.read_point_cloud(self.map_path)
        self.map.estimate_normals()

        self.pose_estimate = np.array([[0.0], [0.0], [0.0]])
        self.latest_pcd_msg: Optional[PointCloud2] = None

        self.threshold = 0.05

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

    def timer_cb(self) -> None:
        if self.latest_pcd_msg is None:
            return

        start = time.time()

        # convert pointcloud message to numpy open3d pointcloud
        points = read_points(
            self.latest_pcd_msg, skip_nans=True, field_names=("x", "y", "z")
        )  # 1D (n, ) np.array holding tuples of (x, y, z)
        points = np.array(
            list(np.array(list(point)) for point in points)
        )  # 2D (n, 3) np.array holding [x, y, z] in each row

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.estimate_normals()

        # initial guess
        T_init = np.eye(4)
        T_init[:3, :3] = Rotation.from_euler("z", self.pose_estimate[2, 0]).as_matrix()
        T_init[:2, 3] = self.pose_estimate[:2, 0]

        # registration
        # reg = o3d.pipelines.registration.registration_icp(
        #     pcd,
        #     self.map,
        #     self.threshold,
        #     T_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        # )
        reg = o3d.pipelines.registration.registration_icp(
            pcd,
            self.map,
            self.threshold,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
        )

        self.get_logger().info(
            f"ICP fitness: {reg.fitness}, inlier_rmse: {reg.inlier_rmse}"
        )
        self.get_logger().info(f"Transformation:\n{reg.transformation}")

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


def main(args=None):
    rclpy.init(args=args)

    global_localization_node = GlobalLocalization()
    rclpy.spin(global_localization_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
