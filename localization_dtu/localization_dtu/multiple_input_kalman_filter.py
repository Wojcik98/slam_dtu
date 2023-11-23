from typing import Optional
from dataclasses import dataclass

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
from rclpy.time import Time, Duration
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header


# ros2 run localization_dtu mikf_node vo_odom:=/rtabmap/odom wheel_odom:=/model/rover/odometry output:=/mikf_odom


@dataclass
class KalmanBufferRecord:
    x_km1_km1: np.ndarray
    P_km1_km1: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    H: np.ndarray
    z: np.ndarray
    last_time: Time
    time: Time


class MultipleInputKalmanFilter(Node):
    """Extended Kalman filter combining multiple odometry measurements."""

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
        self.R_pose = 1e-3 * np.eye(3)

        # measurement matrix
        self.H = np.hstack((np.zeros((3, 3)), np.eye(3)))
        self.H_pose = np.hstack((np.eye(3), np.zeros((3, 3))))

        self.last_time: Optional[Time] = None

        # buffer
        self.kalman_buffer: list[KalmanBufferRecord] = []
        self.kalman_buffer_time = Duration(seconds=20.0)

        # initialize subscribers
        self.vo_sub = self.create_subscription(
            Odometry, "vo_odom", self.vo_callback, 10
        )
        self.wheels_sub = self.create_subscription(
            Odometry, "wheel_odom", self.wheels_callback, 10
        )
        self.pose_sub = self.create_subscription(
            Odometry, "pose_odom", self.pose_callback, 10
        )

        # initialize publishers
        self.filter_pub = self.create_publisher(Odometry, "output", 10)

    def vo_callback(self, msg: Odometry) -> None:
        """Visual odometry measurement callback."""
        last_time = self.last_time
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

        time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        record = KalmanBufferRecord(
            self.x_km1_km1,
            self.P_km1_km1,
            self.Q,
            self.R_vo,
            self.H,
            twist,
            last_time,
            time,
        )
        self.add_to_buffer(record)

        x_k_k, P_k_k = self.kalman_filter(
            self.x_km1_km1, self.P_km1_km1, self.Q, self.R_vo, self.H, twist, dt
        )

        self.publish(x_k_k, P_k_k, msg.header)

        self.x_km1_km1 = x_k_k
        self.P_km1_km1 = P_k_k

    def wheels_callback(self, msg: Odometry) -> None:
        """Wheel odometry measurement callback."""
        last_time = self.last_time
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

        time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        record = KalmanBufferRecord(
            self.x_km1_km1,
            self.P_km1_km1,
            self.Q,
            self.R_wheels,
            self.H,
            twist,
            last_time,
            time,
        )
        self.add_to_buffer(record)

        x_k_k, P_k_k = self.kalman_filter(
            self.x_km1_km1, self.P_km1_km1, self.Q, self.R_wheels, self.H, twist, dt
        )

        self.publish(x_k_k, P_k_k, msg.header)

        self.x_km1_km1 = x_k_k
        self.P_km1_km1 = P_k_k

    def pose_callback(self, msg: Odometry) -> None:
        """Global pose odometry measurement callback."""
        # pose estimate reading
        theta = Rotation.from_quat(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        ).as_euler("xyz")[2]
        pose = np.array(
            [
                [msg.pose.pose.position.x],
                [msg.pose.pose.position.y],
                [theta],
            ]
        )

        # get relevant records from buffer
        buffer = self.get_buffer_slice(
            Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        )
        if len(buffer) == 0:
            return

        last_time = buffer[0].last_time
        time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        # dt = (time - last_time).nanoseconds * 1e-9

        # read covariance
        pose3d_covariance = np.array(msg.pose.covariance).reshape((6, 6))
        pose_covariance = np.zeros((3, 3))
        pose_covariance[:2, :2] = pose3d_covariance[:2, :2]
        pose_covariance[2, 2] = pose3d_covariance[5, 5]

        # print covariance
        self.get_logger().info(f"pose_covariance: {pose_covariance}")

        record = KalmanBufferRecord(
            buffer[0].x_km1_km1,
            buffer[0].P_km1_km1,
            self.Q,
            pose_covariance,
            self.H_pose,
            pose,
            last_time,
            time,
        )
        buffer[0].last_time = time
        buffer.insert(0, record)

        # run the filter
        for i in range(len(buffer)):
            dt = (buffer[i].time - buffer[i].last_time).nanoseconds * 1e-9

            x_k_k, P_k_k = self.kalman_filter(
                buffer[i].x_km1_km1,
                buffer[i].P_km1_km1,
                buffer[i].Q,
                buffer[i].R,
                buffer[i].H,
                buffer[i].z,
                dt,
            )

            if i < len(buffer) - 1:
                buffer[i + 1].x_km1_km1 = x_k_k
                buffer[i + 1].P_km1_km1 = P_k_k
            else:
                self.x_km1_km1 = x_k_k
                self.P_km1_km1 = P_k_k

    def kalman_filter(
        self,
        x_km1_km1: np.ndarray,
        P_km1_km1: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        H: np.ndarray,
        z: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman filter implementation."""
        # predict step
        x_k_km1 = self.f(x_km1_km1, dt)
        F = self.F(x_km1_km1, dt)
        P_k_km1 = F @ P_km1_km1 @ F.T + Q

        # update
        z_hat = H @ x_k_km1
        y = z - z_hat

        S = H @ P_k_km1 @ H.T + R
        K = P_k_km1 @ H.T @ np.linalg.inv(S)

        x_k_k = x_k_km1 + K @ y
        P_k_k = (np.eye(6) - K @ H) @ P_k_km1

        return x_k_k, P_k_k

    def add_to_buffer(self, record: KalmanBufferRecord) -> None:
        """Add a record to the buffer and remove old records."""
        self.kalman_buffer.append(record)

        try:
            time_threshold = record.time - self.kalman_buffer_time
        except ValueError:  # time_threshold is negative
            time_threshold = Time(seconds=0, nanoseconds=0)

        self.kalman_buffer = [
            record for record in self.kalman_buffer if record.time > time_threshold
        ]

    def get_buffer_slice(self, time: Time) -> list[KalmanBufferRecord]:
        """Get all records in the buffer after the given time."""
        return [record for record in self.kalman_buffer if record.time > time]

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
