import rclpy
from rclpy.node import Node

from agents.navigation.mpc.mpc_node import mpcControlNode


def main(args=None):
    rclpy.init(args=args)

    try:
        mpc_node = mpcControlNode()

        try:
            # rclpy.on_shutdown(stanley_control_node.stop_vehicle)
            rclpy.spin(mpc_node)
        except Exception as e:
            print(f"MPC Control Node cannot be spinned! Reason: \n {e}")
            # Destroy the node explicitly
            # (optional - otherwise it will be done automatically
            # when the garbage collector destroys the node object)
            mpc_node.destroy_node()
            rclpy.shutdown()

    except Exception as e:
        print(f"MPC Control Node cannot be created! Reason: \n {e}")


if __name__ == "__main__":
    main()
