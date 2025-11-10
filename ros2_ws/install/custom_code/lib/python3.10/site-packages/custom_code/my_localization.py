#!/usr/bin/env python3

import numpy as np


# Patch for deprecated numpy aliases
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

    
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from custom_code.robot_functions import RobotFunctions
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
import math

def yaw_to_quaternion(yaw):
    """Convert a yaw angle (in radians) into a Quaternion message."""
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q

class Odom3Node(Node):
    def __init__(self):
        super().__init__("lidar_tf")


        # Declare parameter with default value (1000)
        self.declare_parameter("num_particles", 1000)
        num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value


        self.subscription_odom = self.create_subscription(
            Odometry, "/calc_odom", self.odom_callback, 10
        )

        self.subscription_real_odom = self.create_subscription(
            Odometry, "/odom", self.real_odom_callback, 10
        )

        self.subscription_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        map_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.likelyhood_map = self.create_subscription(
            OccupancyGrid, "/likelihood_map", self.map_callback, map_qos)

        self.last_odom = (0,0,0)  # Store previous odometry state
        self.read_odom = False

        self.map_received = False

        self.robot = RobotFunctions(num_particles)

        # Create a publisher for PointCloud2
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, "/particle_cloud", 10
        )

        self.real_path_pub = self.create_publisher(Path, "/real_robot_path", 10)
        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"

        self.calc_path_pub = self.create_publisher(Path, "/calc_robot_path", 10)
        self.calc_path_msg = Path()
        self.calc_path_msg.header.frame_id = "map"

        self.particle_path_pub = self.create_publisher(Path, "/particle_robot_path", 10)
        self.particle_path_msg = Path()
        self.particle_path_msg.header.frame_id = "map"

    def map_callback(self, msg):
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            self.grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
            self.get_logger().info('Map and inflated map processed.')


    def create_pointcloud2(self, points):
        """Create a PointCloud2 message from numpy array."""
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # or whatever your frame is
        
        if len(points.shape) == 3:
            points = points.reshape(-1, 3)
        
        intensities = np.array(self.robot.get_weights(), dtype=np.float32)
        if np.max(intensities) > 0:
            intensities /= np.max(intensities)
        intensities = intensities.reshape(-1, 1).astype(np.float32)
        points_with_intensity = np.hstack([points.astype(np.float32), intensities])

        msg.height = 1
        msg.width = points_with_intensity.shape[0]

        
        # Define the point fields (x, y, z)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16  # 3 floats * 4 bytes each
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        
        # Convert points to bytes
        msg.data = points_with_intensity.tobytes()
        
        return msg
    

    def plot_particles(self,):
        # Apply odometry motion model
        samples = self.robot.get_particle_states()
        weights = self.robot.get_weights()

        # Convert samples to PointCloud2 and publish
        # Add z=0 to make it 3D (required by PointCloud2)
        samples_3d = np.hstack([samples[:, :2], np.zeros((samples.shape[0], 1))])
        pointcloud_msg = self.create_pointcloud2(samples_3d)
        self.pointcloud_pub.publish(pointcloud_msg)

        # --- Compute weighted average pose ---
        selected_state = self.robot.get_selected_state()
        mean_x = selected_state[0]
        mean_y = selected_state[1]

        # Handle circular mean for angle
        cos_t = np.cos(selected_state[2])
        sin_t = np.sin(selected_state[2])
        mean_theta = np.arctan2(sin_t, cos_t)

        # Build PoseStamped for the mean particle
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(mean_x)
        pose_stamped.pose.position.y = float(mean_y)
        pose_stamped.pose.position.z = 0.0

        # Convert yaw to quaternion
        q = yaw_to_quaternion(mean_theta)
        pose_stamped.pose.orientation = q

        # Append to path and publish
        self.particle_path_msg.poses.append(pose_stamped)
        self.particle_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.particle_path_pub.publish(self.particle_path_msg)


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

        deltas = {'t': 0,
                  'r1': 0,
                  'r2': 0}

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

            deltas['t'] = delta_t
            deltas['r1'] = delta_rot1
            deltas['r2'] = delta_rot2

            self.robot.move_particles(deltas)

            self.plot_particles()

        self.last_odom = (x, y, theta)
        if self.read_odom == False:
            self.read_odom = True


    def scan_with_calc(self, data: LaserScan):
        if self.map_received:
            self.robot.update_particles(data, self.map_data, self.grid)


    def scan_callback(self, data):
        data.angle_min += np.pi
        data.angle_max += np.pi

        self.scan_with_calc(data)
        

def main(args=None):
    rclpy.init(args=args)
    node = Odom3Node()
    
    try:
        rclpy.spin(node)  # Keep ROS 2 running
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()