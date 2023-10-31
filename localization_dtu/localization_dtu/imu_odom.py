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
from sensor_msgs.msg import Imu
from std_msgs.msg import Header


class ImuOdometry(Node):
    """2D odometry from IMU data."""

    def __init__(self):
        super().__init__("imu_odom")

        # state vector: [x, y, theta, vx, vy, omegaz, ax, ay]
        self.n = 8  # number of variables
        self.x_km1_km1 = np.zeros((self.n, 1))
        self.P_km1_km1 = np.zeros((self.n, self.n))

        # process noise covariance matrix
        self.Q = 1e-5 * np.eye(self.n)

        # measurement noise covariance matrix (TODO: can we get this from the IMU?)
        self.R = 1e-5 * np.eye(3)

        # measurement matrix (3 x 8)
        self.H = np.hstack((np.zeros((3, 5)), np.eye(3)))

        self.last_time = None

        # initialize subscriber
        self.sub = self.create_subscription(Imu, "imu", self.callback, 10)

        # initialize publisher
        self.pub = self.create_publisher(Odometry, "odom", 10)

    def f(self, x_km1_km1: np.ndarray, dt: float) -> np.ndarray:
        """The state transition function."""
        x, y, th, vx, vy, omz, ax, ay = x_km1_km1[:, 0]
        # x = x_km1_km1[0, 0]
        # y = x_km1_km1[1, 0]
        # th = x_km1_km1[2, 0]
        # vx = x_km1_km1[3, 0]
        # vy = x_km1_km1[4, 0]
        # omz = x_km1_km1[5, 0]
        # ax = x_km1_km1[6, 0]
        # ay = x_km1_km1[7, 0]

        dvx = ax * dt
        dvy = ay * dt
        dth = omz * dt

        dx = (
            (vx + 0.5 * ax * dt) * np.cos(th + 0.5 * omz * dt)
            - (vy + 0.5 * ay * dt) * np.sin(th + 0.5 * omz * dt)
        ) * dt
        dy = (
            (vx + 0.5 * ax * dt) * np.sin(th + 0.5 * omz * dt)
            + (vy + 0.5 * ay * dt) * np.cos(th + 0.5 * omz * dt)
        ) * dt

        return np.array(
            [
                [x + dx],
                [y + dy],
                [th + dth],
                [vx + dvx],
                [vy + dvy],
                [omz],
                [ax],
                [ay],
            ]
        )

    def F(self, x_km1_km1: np.ndarray, dt: float) -> np.ndarray:
        """The Jacobian of the state transition function."""
        x, y, th, vx, vy, omz, ax, ay = x_km1_km1[:, 0]

        dx_dth = (
            -(vx + 0.5 * ax * dt) * np.sin(th + 0.5 * omz * dt)
            - (vy + 0.5 * ay * dt) * np.cos(th + 0.5 * omz * dt)
        ) * dt
        dy_dth = (
            (vx + 0.5 * ax * dt) * np.cos(th + 0.5 * omz * dt)
            - (vy + 0.5 * ay * dt) * np.sin(th + 0.5 * omz * dt)
        ) * dt

        dx_dvx = np.cos(th + 0.5 * omz * dt) * dt
        dy_dvx = np.sin(th + 0.5 * omz * dt) * dt

        dx_dvy = -np.sin(th + 0.5 * omz * dt) * dt
        dy_dvy = np.cos(th + 0.5 * omz * dt) * dt

        # fmt: off
        dx_domz = (
            -(vx + 0.5 * ax * dt) * np.sin(th + 0.5 * omz * dt)
            - (vy + 0.5 * ay * dt) * np.cos(th + 0.5 * omz * dt)
        ) * 0.5 * dt**2
        dy_domz = (
            (vx + 0.5 * ax * dt) * np.cos(th + 0.5 * omz * dt)
            - (vy + 0.5 * ay * dt) * np.sin(th + 0.5 * omz * dt)
        ) * 0.5 * dt**2
        # fmt: on

        dx_dax = np.cos(th + 0.5 * omz * dt) * 0.5 * dt**2
        dy_dax = np.sin(th + 0.5 * omz * dt) * 0.5 * dt**2

        dx_day = -np.sin(th + 0.5 * omz * dt) * 0.5 * dt**2
        dy_day = np.cos(th + 0.5 * omz * dt) * 0.5 * dt**2

        return np.array(
            [
                [1, 0, dx_dth, dx_dvx, dx_dvy, dx_domz, dx_dax, dx_day],
                [0, 1, dy_dth, dy_dvx, dy_dvy, dy_domz, dy_dax, dy_day],
                [0, 0, 1, 0, 0, dt, 0, 0],  # git
                [0, 0, 0, 1, 0, 0, dt, 0],  # git
                [0, 0, 0, 0, 1, 0, 0, dt],  # git
                [0, 0, 0, 0, 0, 1, 0, 0],  # git
                [0, 0, 0, 0, 0, 0, 1, 0],  # git
                [0, 0, 0, 0, 0, 0, 0, 1],  # git
            ]
        )

    def callback(self, msg: Imu) -> None:
        if self.last_time is None:
            self.last_time = Time(
                seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec
            )
            return

        time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
        dt = (time - self.last_time).nanoseconds * 1e-9
        self.last_time = time

        # predict step
        x_k_km1 = self.f(self.x_km1_km1, dt)
        F = self.F(self.x_km1_km1, dt)
        P_k_km1 = F @ self.P_km1_km1 @ F.T + self.Q

        # update
        z = np.array(
            [
                [msg.angular_velocity.x],
                [msg.linear_acceleration.x],
                [msg.linear_acceleration.y],
            ]
        )
        z_hat = self.H @ x_k_km1
        y = z - z_hat

        S = self.H @ P_k_km1 @ self.H.T + self.R
        K = P_k_km1 @ self.H.T @ np.linalg.inv(S)

        x_k_k = x_k_km1 + K @ y
        P_k_k = (np.eye(self.n) - K @ self.H) @ P_k_km1

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
        self.pub.publish(
            Odometry(
                header=header,
                pose=pose,
                twist=twist,
            )
        )


def main(args=None):
    rclpy.init(args=args)

    imu_odom_node = ImuOdometry()
    rclpy.spin(imu_odom_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
