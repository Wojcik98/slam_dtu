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


class Localization(Node):
    def __init__(self):
        super().__init__("localization")

        # state vector: [x, y, theta, v, omega]
        self.x_km1_km1 = np.zeros((5, 1))
        self.P_km1_km1 = np.zeros((5, 5))

        # process noise covariance matrix
        self.Q = 0.001 * np.eye(5)

        # measurement noise covariance matrix
        self.R = 0.001 * np.eye(2)

        # measurement matrix
        self.H = np.hstack((np.zeros((2, 3)), np.eye(2)))

        self.last_time = None

        # initialize subscribers
        self.twist_sub = self.create_subscription(
            Odometry, "twist", self.velocity_cb, 10
        )

        # initialize publishers
        self.filter_pub = self.create_publisher(Odometry, "output", 10)

    def f(self, x_km1_km1: np.ndarray, dt: float) -> np.ndarray:
        """The state transition function."""
        dvx = x_km1_km1[3, 0] * np.cos(x_km1_km1[2, 0] + dt * x_km1_km1[4, 0] / 2) * dt
        dvy = x_km1_km1[3, 0] * np.sin(x_km1_km1[2, 0] + dt * x_km1_km1[4, 0] / 2) * dt
        return np.array(
            [
                [x_km1_km1[0, 0] + dvx],
                [x_km1_km1[1, 0] + dvy],
                [x_km1_km1[2, 0] + x_km1_km1[4, 0] * dt],
                [x_km1_km1[3, 0]],
                [x_km1_km1[4, 0]],
            ]
        )

    def F(self, x_km1_km1: np.ndarray, dt: float) -> np.ndarray:
        """The Jacobian of the state transition function."""
        return np.array(
            [
                [
                    1,
                    0,
                    -x_km1_km1[3, 0]
                    * np.sin(x_km1_km1[2, 0] + dt * x_km1_km1[4, 0] / 2)
                    * dt,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    x_km1_km1[3, 0]
                    * np.cos(x_km1_km1[2, 0] + dt * x_km1_km1[4, 0] / 2)
                    * dt,
                    0,
                    0,
                ],
                [0, 0, 1, 0, dt],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )

    def velocity_cb(self, msg: Odometry) -> None:
        """Callback function for the velocity subscriber."""
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
        z = np.array([[msg.twist.twist.linear.x], [msg.twist.twist.angular.z]])
        z_hat = self.H @ x_k_km1
        y = z - z_hat

        S = self.H @ P_k_km1 @ self.H.T + self.R
        K = P_k_km1 @ self.H.T @ np.linalg.inv(S)

        x_k_k = x_k_km1 + K @ y
        # self.get_logger().info(f"Eye shape: {(K @ self.H).shape}")
        P_k_k = (np.eye(5) - K @ self.H) @ P_k_km1

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
        twist_covariance[0, 0] = P_k_k[3, 3]
        twist_covariance[5, 5] = P_k_k[4, 4]

        twist = TwistWithCovariance(
            twist=Twist(
                linear=Vector3(x=x_k_k[3, 0], y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=x_k_k[4, 0]),
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


def main(args=None):
    rclpy.init(args=args)

    localization_node = Localization()
    rclpy.spin(localization_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
