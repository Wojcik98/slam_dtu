import numpy as np
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
from std_msgs.msg import Header

from typing import Optional


class OdomMetrics(Node):
    def __init__(self):
        super().__init__("odom_metrics")

        self.name = "" # TODO get from param

        self.gt_last: Optional[Odometry] = None
        self.dist_hist = []
        self.cov_hist = []

        # initialize subscribers
        self.gt_sub = self.create_subscription(
            Odometry, "gt", self.gt_cb, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "odom", self.odom_cb, 10
        )

    def gt_cb(self, msg: Odometry) -> None:
        """Callback function for the ground truth data."""
        self.gt_last = msg

    def odom_cb(self, msg: Odometry) -> None:
        """Callback function for the evaluated odometry."""
        if self.gt_last is None:
            return

        gt_pos = self.gt_last.pose.position
        odom_pos = msg.pose.position

        dist = ((gt_pos.x - odom_pos.x)**2 + (gt_pos.y - odom_pos.y)**2)**0.5

    def __del__(self) -> None:
        # TODO save data
        super().__del__()   # or something



def main(args=None):
    rclpy.init(args=args)

    metrics_node = OdomMetrics()
    rclpy.spin(metrics_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
