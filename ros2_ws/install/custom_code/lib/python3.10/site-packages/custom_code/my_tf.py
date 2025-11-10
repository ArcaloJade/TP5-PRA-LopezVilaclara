#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from custom_code.robot_functions import RobotFunctions


class Odom3Node(Node):
    def __init__(self):
        super().__init__("lidar_tf")

        self.subscription_odom = self.create_subscription(
            Odometry, "/calc_odom", self.odom_callback, 10
        )

        self.subscription_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        
        # Create a publisher for PointCloud2
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, "/particle_cloud", 10
        )

        # Create a publisher for PointCloud2
        self.pointcloud_pub_mod = self.create_publisher(
            PointCloud2, "/particle_cloud_modified", 10
        )

        self.last_odom = (0,0,0)  # Store previous odometry state
        
        self.robot = RobotFunctions()

    def create_pointcloud2(self, points):
        """Create a PointCloud2 message from numpy array."""
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"  # or whatever your frame is
        
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
    

    def create_pointcloud_mod(self, points, stamp):
        """Create a PointCloud2 message from numpy array."""
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = "calc_base_footprint"  # or whatever your frame is
        
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

    def odom_callback(self, data: Odometry):        
        
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        # Extract quaternion (w, x, y, z) format
        q_w = data.pose.pose.orientation.w
        q_x = data.pose.pose.orientation.x
        q_y = data.pose.pose.orientation.y
        q_z = data.pose.pose.orientation.z
        
        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2]  # Extract yaw

        self.last_odom = (x, y, theta)

    def scan_with_tf(self, data: LaserScan):
        ranges = np.array(data.ranges)
        valid_mask = (ranges >= data.range_min) & (ranges <= data.range_max)

        angles = data.angle_min + np.arange(len(ranges)) * data.angle_increment
        angles = angles[valid_mask]
        ranges = ranges[valid_mask]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        points_laser = np.vstack([xs, ys, np.ones_like(xs)])  # Shape: (3, N)

        samples_3d = np.vstack([
            points_laser[0],      # x
            points_laser[1],      # y
            np.zeros_like(points_laser[0])  # z = 0
        ]).T  # Shape: (N, 3)
        pointcloud_msg = self.create_pointcloud_mod(samples_3d, data.header.stamp)
        self.pointcloud_pub_mod.publish(pointcloud_msg)

    def scan_with_calc(self, data: LaserScan):
        points_map = self.robot.scan_refererence(data.ranges,
                                                 data.range_min,
                                                 data.range_max,
                                                 data.angle_min,
                                                 data.angle_max,
                                                 data.angle_increment,
                                                 self.last_odom)

        samples_3d = np.vstack([
            points_map[0],      # x
            points_map[1],      # y
            np.zeros_like(points_map[0])  # z = 0
        ]).T  # Shape: (N, 3)
        pointcloud_msg = self.create_pointcloud2(samples_3d)
        self.pointcloud_pub.publish(pointcloud_msg)


    def scan_callback(self, data):
        self.scan_with_tf(data)

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
