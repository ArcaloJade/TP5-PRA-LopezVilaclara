#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Quaternion
import math
from custom_msgs.msg import DeltaOdom, Belief

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

        self.ekf_path_pub = self.create_publisher(Path, "/ekf_robot_path", 10)
        self.ekf_path_msg = Path()
        self.ekf_path_msg.header.frame_id = "map"

        self.counter = 0
        self.limit = 1

        # Create a publisher for PointCloud2
        self.pointcloud_pub1 = self.create_publisher(
            PointCloud2, "/particle_cloud1", 10
        )

        # Create a publisher for PointCloud2
        self.pointcloud_pub2 = self.create_publisher(
            PointCloud2, "/particle_cloud2", 10
        )

        self.delta_pub = self.create_publisher(
            DeltaOdom, "/delta", 10
        )

        self.publish_belief = self.create_publisher(
            Belief, "/belief", 10
        )
        self.subscription_belief = self.create_subscription(
            Belief, "/belief", self.belief_callback, 10
        )

        self.pub_pose_with_cov = self.create_publisher(
            PoseWithCovarianceStamped, "/pose_with_covariance", 10
        )

        msg = Belief()
        msg.mu.x = 0.0
        msg.mu.y = 0.0
        msg.mu.theta = 0.0
        msg.covariance = [1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, (np.pi/4)**2]
        self.publish_belief.publish(msg)

    def belief_callback(self, msg):
        self.mu = np.array([msg.mu.x, msg.mu.y, msg.mu.theta])
        self.covariance = np.array([[msg.covariance[0], msg.covariance[1], msg.covariance[2]],
                                    [msg.covariance[3], msg.covariance[4], msg.covariance[5]],
                                    [msg.covariance[6], msg.covariance[7], msg.covariance[8]]])

        # Build PoseWithCovarianceStamped for RViz
        pose_msg = PoseWithCovarianceStamped()
        # Belief doesnt have a header
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = self.mu[0]
        pose_msg.pose.pose.position.y = self.mu[1]
        pose_msg.pose.pose.position.z = 0.0

        # Convert theta -> quaternion with your helper
        pose_msg.pose.pose.orientation = yaw_to_quaternion(self.mu[2])

        # Map 3x3 covariance into 6x6 covariance
        cov6 = np.zeros((6, 6))
        cov6[0:2, 0:2] = self.covariance[0:2, 0:2]   # x,y
        cov6[0:2, 5]   = self.covariance[0:2, 2]     # x,y with yaw
        cov6[5, 0:2]   = self.covariance[2, 0:2]     # yaw with x,y
        cov6[5, 5]     = self.covariance[2, 2]       # yaw

        pose_msg.pose.covariance = cov6.flatten().tolist()

        # Publish for RViz visualization
        self.pub_pose_with_cov.publish(pose_msg)

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose_msg.pose.pose   # geometry_msgs/Pose
        self.ekf_path_msg.poses.append(pose_stamped)
        self.ekf_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.ekf_path_pub.publish(self.ekf_path_msg) 

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

    def create_pointcloud2(self, points):
        """Create a PointCloud2 message from numpy array."""
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "calc_odom"  # or whatever your frame is
        
        if len(points.shape) == 3:
            points = points.reshape(-1, 3)
        
        msg.height = 1
        msg.width = len(points)
        
        # Define the point fields (x, y, z)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12  # 3 floats * 4 bytes each
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        
        # Convert points to bytes
        msg.data = points.astype(np.float32).tobytes()
        
        return msg
        

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