import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from slam_interfaces.msg import FullOdometry


class OdometryTranslate(Node):
    """Translate from custom message type FullOdometry to standard nav_msgs/Odometry."""

    def __init__(self):
        super().__init__("odometry_translate")

        # initialize subscriber
        self.sub = self.create_subscription(FullOdometry, "input", self.callback, 10)

        # initialize publisher
        self.pub = self.create_publisher(Odometry, "output", 10)

    def callback(self, msg: FullOdometry) -> None:
        out = Odometry(
            header=msg.header,
            child_frame_id=msg.child_frame_id,
            pose=msg.pose,
            twist=msg.twist,
        )

        # publish
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)

    odometry_translate_node = OdometryTranslate()
    rclpy.spin(odometry_translate_node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
