#!/usr/bin/env python3

import numpy as np    
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
import math
from custom_msgs.msg import DeltaOdom

def yaw_to_quaternion(yaw):
    """Convert a yaw angle (in radians) into a Quaternion message."""
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q

class EKFNode(Node):
    def __init__(self):
        super().__init__("EKFNode")


        self.subscription_odom = self.create_subscription(
            Odometry, "/calc_odom", self.odom_callback, 10
        )

        self.subscription_real_odom = self.create_subscription(
            Odometry, "/odom", self.real_odom_callback, 10
        )

        self.last_odom = (0,0,0)  # Store previous odometry state
        self.read_odom = False

        self.real_path_pub = self.create_publisher(Path, "/real_robot_path", 10)
        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"

        self.calc_path_pub = self.create_publisher(Path, "/calc_robot_path", 10)
        self.calc_path_msg = Path()
        self.calc_path_msg.header.frame_id = "map"

        self.counter = 0
        self.limit = 1

        self.delta_pub = self.create_publisher(
            DeltaOdom, "/delta", 10
        )

    def real_odom_callback(self, data: Odometry):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = data.pose.pose   # geometry_msgs/Pose

        self.real_path_msg.poses.append(pose_stamped)
        self.real_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.real_path_pub.publish(self.real_path_msg)

    def odom_callback(self, data: Odometry):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = data.pose.pose   # geometry_msgs/Pose

        self.calc_path_msg.poses.append(pose_stamped)
        self.calc_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.calc_path_pub.publish(self.calc_path_msg)

        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        # Extract quaternion (w, x, y, z) format
        q_w = data.pose.pose.orientation.w
        q_x = data.pose.pose.orientation.x
        q_y = data.pose.pose.orientation.y
        q_z = data.pose.pose.orientation.z
        
        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2]  # Extract yaw

        if self.read_odom:
            # Compute translation difference
            dx = x - self.last_odom[0]
            dy = y - self.last_odom[1]
            delta_t = np.sqrt(dx**2 + dy**2)

            if delta_t > 1e-6:
                delta_rot1 = np.arctan2(dy, dx) - self.last_odom[2]
                delta_rot2 = theta - self.last_odom[2] - delta_rot1
            else:
                # No translation → assume in-place rotation
                delta_rot1 = 0.0
                delta_rot2 = theta - self.last_odom[2]
            
            # Normalize angles
            delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
            delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))

            msg = DeltaOdom()
            msg.dr1 = delta_rot1
            msg.dr2 = delta_rot2
            msg.dt = delta_t
            self.delta_pub.publish(msg)

        self.last_odom = (x, y, theta)
        if self.read_odom == False:
            self.read_odom = True
        

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    
    try:
        rclpy.spin(node)  # Keep ROS 2 running
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
