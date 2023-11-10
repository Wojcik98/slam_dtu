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


# ros2 run localization_dtu mikf_node vo_odom:=/rtabmap/odom wheel_odom:=/model/rover/odometry output:=/mikf_odom


class MultipleInputKalmanFilter(Node):
    """Extended Kalman filter combining two velocity measurements."""

    def __init__(self):
        super().__init__("mikf_node")

        # state vector: [x, y, theta, v_x, v_y, omega_z]
        self.x_km1_km1 = np.zeros((6, 1))
        self.P_km1_km1 = np.zeros((6, 6))

        # process noise covariance matrix
        self.Q = 1e-5 * np.eye(6)

        # measurement noise covariance matrix
        self.R_vo = 1e-5 * np.eye(3)
        self.R_wheels = 1e-3 * np.eye(3)

        # measurement matrix
        self.H = np.hstack((np.zeros((3, 3)), np.eye(3)))

        self.last_time: Optional[Time] = None

        # initialize subscribers
        self.vo_sub = self.create_subscription(
            Odometry, "vo_odom", self.vo_callback, 10
        )
        self.wheels_sub = self.create_subscription(
            Odometry, "wheel_odom", self.wheels_callback, 10
        )

        # initialize publishers
        self.filter_pub = self.create_publisher(Odometry, "output", 10)

    def vo_callback(self, msg: Odometry) -> None:
        """Visual odometry measurement callback."""
        dt = self.get_dt(msg.header)
        if dt is None:
            return

        twist = np.array(
            [
                [msg.twist.twist.linear.x],
                [msg.twist.twist.linear.y],
                [msg.twist.twist.angular.z],
            ]
        )

        # predict step
        x_k_km1 = self.f(self.x_km1_km1, dt)
        F = self.F(self.x_km1_km1, dt)
        P_k_km1 = F @ self.P_km1_km1 @ F.T + self.Q

        # update
        z = twist
        z_hat = self.H @ x_k_km1
        y = z - z_hat

        S = self.H @ P_k_km1 @ self.H.T + self.R_vo
        K = P_k_km1 @ self.H.T @ np.linalg.inv(S)

        x_k_k = x_k_km1 + K @ y
        P_k_k = (np.eye(6) - K @ self.H) @ P_k_km1

        self.publish(x_k_k, P_k_k, msg.header)

        self.x_km1_km1 = x_k_k
        self.P_km1_km1 = P_k_k

    def wheels_callback(self, msg: Odometry) -> None:
        """Wheel odometry measurement callback."""
        dt = self.get_dt(msg.header)
        if dt is None:
            return

        twist = np.array(
            [
                [msg.twist.twist.linear.x],
                [msg.twist.twist.linear.y],
                [msg.twist.twist.angular.z],
            ]
        )

        # predict step
        x_k_km1 = self.f(self.x_km1_km1, dt)
        F = self.F(self.x_km1_km1, dt)
        P_k_km1 = F @ self.P_km1_km1 @ F.T + self.Q

        # update
        z = twist
        z_hat = self.H @ x_k_km1
        y = z - z_hat

        S = self.H @ P_k_km1 @ self.H.T + self.R_wheels
        K = P_k_km1 @ self.H.T @ np.linalg.inv(S)

        x_k_k = x_k_km1 + K @ y
        P_k_k = (np.eye(6) - K @ self.H) @ P_k_km1

        self.publish(x_k_k, P_k_k, msg.header)

        self.x_km1_km1 = x_k_k
        self.P_km1_km1 = P_k_k

    def publish(self, x_k_k: np.ndarray, P_k_k: np.ndarray, header: Header) -> None:
        # pose
        pose_covariance = np.zeros((6, 6))
        pose_covariance[:2, :2] = P_k_k[:2, :2]
        pose_covariance[5, 5] = P_k_k[2, 2]
        q = Rotation.from_euler("z", x_k_k[2, 0]).as_quat()

        pose = PoseWithCovariance(
            pose=Pose(
                position=Point(x=x_k_k[0, 0], y=x_k_k[1, 0], z=0.0),
                orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
            ),
            covariance=pose_covariance.ravel(),
        )

        # twist
        twist_covariance = np.zeros((6, 6))
        twist_covariance[:2, :2] = P_k_k[3:5, 3:5]
        twist_covariance[5, 5] = P_k_k[5, 5]

        twist = TwistWithCovariance(
            twist=Twist(
                linear=Vector3(x=x_k_k[3, 0], y=x_k_k[4, 0], z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=x_k_k[5, 0]),
            ),
            covariance=twist_covariance.ravel(),
        )

        # publish
        self.filter_pub.publish(
            Odometry(
                header=header,
                pose=pose,
                twist=twist,
            )
        )

    def f(self, x_km1_km1: np.ndarray, dt: float) -> np.ndarray:
        """The state transition function for the omnidirectional drive model."""
        x, y, th, vx, vy, omz = x_km1_km1[:, 0]

        dth = omz * dt

        dx = (vx * np.cos(th + 0.5 * omz * dt) - vy * np.sin(th + 0.5 * omz * dt)) * dt
        dy = (vx * np.sin(th + 0.5 * omz * dt) + vy * np.cos(th + 0.5 * omz * dt)) * dt

        return np.array(
            [
                [x + dx],
                [y + dy],
                [th + dth],
                [vx],
                [vy],
                [omz],
            ]
        )

    def F(self, x_km1_km1: np.ndarray, dt: float) -> np.ndarray:
        """The Jacobian of the state transition function for the omnidirectional drive
        model."""
        x, y, th, vx, vy, omz = x_km1_km1[:, 0]

        dx_dth = (
            -vx * np.sin(th + 0.5 * omz * dt) - vy * np.cos(th + 0.5 * omz * dt)
        ) * dt
        dy_dth = (
            vx * np.cos(th + 0.5 * omz * dt) - vy * np.sin(th + 0.5 * omz * dt)
        ) * dt

        dx_dvx = np.cos(th + 0.5 * omz * dt) * dt
        dy_dvx = np.sin(th + 0.5 * omz * dt) * dt

        dx_dvy = -np.sin(th + 0.5 * omz * dt) * dt
        dy_dvy = np.cos(th + 0.5 * omz * dt) * dt

        # fmt: off
        dx_domz = (
            -vx * np.sin(th + 0.5 * omz * dt)
            - vy * np.cos(th + 0.5 * omz * dt)
        ) * 0.5 * dt**2
        dy_domz = (
            vx * np.cos(th + 0.5 * omz * dt)
            - vy * np.sin(th + 0.5 * omz * dt)
        ) * 0.5 * dt**2
        # fmt: on

        return np.array(
            [
                [1, 0, dx_dth, dx_dvx, dx_dvy, dx_domz],
                [0, 1, dy_dth, dy_dvx, dy_dvy, dy_domz],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    def get_dt(self, header: Header) -> Optional[float]:
        if self.last_time is None:
            self.last_time = Time(
                seconds=header.stamp.sec, nanoseconds=header.stamp.nanosec
            )
            return None

        time = Time(seconds=header.stamp.sec, nanoseconds=header.stamp.nanosec)
        dt = (time - self.last_time).nanoseconds * 1e-9
        self.last_time = time
        return dt


def main(args=None):
    rclpy.init(args=args)

    mikf_node = MultipleInputKalmanFilter()
    rclpy.spin(mikf_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
