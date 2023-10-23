import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry


class Localization(Node):
    def __init__(self):
        super().__init__("localization")
        self.subscription = self.create_subscription(
            Odometry, "velocity", self.velocity_cb, 10
        )

    def velocity_cb(self, msg: Odometry) -> None:
        self.get_logger().info(f"Received velocity: {msg.twist.twist.linear.x}")


def main(args=None):
    rclpy.init(args=args)

    localization_node = Localization()
    rclpy.spin(localization_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
