from typing import Optional

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


class VelocityComplementaryFilter(Node):
    """A complementary filter combining two velocity measurements."""

    def __init__(self):
        super().__init__("velocity_complementary_filter_node")

        # state vector: [vx, vy, omegaz]
        self.x_km1_km1 = np.zeros((3, 1))
        self.P_km1_km1 = np.eye(3)

        # process noise covariance matrix
        self.Q = 1e-15 * np.eye(3)

        # measurement noise covariance matrix
        self.R = 1e-15 * np.eye(3)

        # measurement matrix
        self.H = np.eye(3)

        self.last_primary_twist: Optional[np.ndarray] = None

        self.last_time: Optional[Time] = None

        # initialize subscribers
        self.primary_sub = self.create_subscription(
            Odometry, "primary_odom", self.primary_callback, 10
        )
        self.secondary_sub = self.create_subscription(
            Odometry, "secondary_odom", self.secondary_callback, 10
        )

        # initialize publishers
        self.filter_pub = self.create_publisher(Odometry, "output", 10)

    def primary_callback(self, msg: Odometry) -> None:
        twist = np.array(
            [
                [msg.twist.twist.linear.x],
                [msg.twist.twist.linear.y],
                [msg.twist.twist.angular.z],
            ]
        )
        self.last_primary_twist = twist

        twist += self.x_km1_km1

        msg_out = Odometry(
            header=msg.header,
            pose=PoseWithCovariance(),
            twist=TwistWithCovariance(
                twist=Twist(
                    linear=Vector3(x=twist[0, 0], y=twist[1, 0], z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=twist[2, 0]),
                ),
                covariance=msg.twist.covariance,  # TODO sum covariance matrices?
            ),
        )
        self.filter_pub.publish(msg_out)

    def secondary_callback(self, msg: Odometry) -> None:
        """Callback function for the velocity subscriber."""
        if self.last_primary_twist is None:
            return

        new_twist = np.array(
            [
                [msg.twist.twist.linear.x],
                [msg.twist.twist.linear.y],
                [msg.twist.twist.angular.z],
            ]
        )

        # if self.last_time is None:
        #     self.last_time = Time(
        #         seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec
        #     )
        #     return

        # time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        # dt = (time - self.last_time).nanoseconds * 1e-9
        # self.last_time = time

        # TODO extract Kalman filter to separate class

        # predict step
        # TODO how can we predict? add acceleration term?
        x_k_km1 = self.x_km1_km1
        F = np.eye(3)
        P_k_km1 = F @ self.P_km1_km1 @ F.T + self.Q

        # update
        z = new_twist - self.last_primary_twist
        z_hat = self.H @ x_k_km1
        y = z - z_hat

        S = self.H @ P_k_km1 @ self.H.T + self.R
        K = P_k_km1 @ self.H.T @ np.linalg.inv(S)

        x_k_k = x_k_km1 + K @ y
        P_k_k = (np.eye(3) - K @ self.H) @ P_k_km1

        self.x_km1_km1 = x_k_k
        self.P_km1_km1 = P_k_k


def main(args=None):
    rclpy.init(args=args)

    complementary_filter_node = VelocityComplementaryFilter()
    rclpy.spin(complementary_filter_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
