import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PointStamped, Quaternion
import math
import numpy as np
import tf2_ros
import cv2

def do_transform_point(point_stamped, transform_stamped):
    # Extract translation
    t = transform_stamped.transform.translation
    translation = np.array([t.x, t.y, t.z])

    # Extract yaw angle from quaternion (assuming only Z rotation matters)
    q = transform_stamped.transform.rotation
    # yaw = atan2(2*(wz), 1 - 2*z^2)
    yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))

    # Build 2D rotation matrix
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rot_matrix = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw,  cos_yaw, 0.0],
        [0.0,      0.0,     1.0]
    ])

    # Point to numpy
    p = np.array([point_stamped.point.x,
                  point_stamped.point.y,
                  point_stamped.point.z])

    # Apply transform
    p_transformed = rot_matrix.dot(p) + translation

    # Fill result
    result = PointStamped()
    result.header = transform_stamped.header
    result.point.x, result.point.y, result.point.z = p_transformed
    return result

class LandmarkObserver(Node):
    def __init__(self):
        super().__init__('landmark_observer')

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_map = self.create_subscription(PoseArray, '/landmarks', self.map_callback, 10)

        # Publisher
        self.pub_obs = self.create_publisher(PoseArray, '/observed_landmarks', 10)

        self.map_landmarks = None

    def map_callback(self, msg: PoseArray):
        self.map_landmarks = msg

    def scan_callback(self, msg: LaserScan):
        if self.map_landmarks is None:
            return

        try:
            # Get transform from map -> base_footprint
            transform = self.tf_buffer.lookup_transform(
                target_frame='base_footprint',
                source_frame=self.map_landmarks.header.frame_id,  # usually "map"
                time=rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"Transform unavailable: {str(e)}")
            return

        obs_array = PoseArray()
        obs_array.header = msg.header
        obs_array.header.frame_id = 'base_footprint'  # measurements are in robot frame

        # Laser parameters
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        r_min = msg.range_min
        r_max = msg.range_max

        for landmark in self.map_landmarks.poses:
            # Convert map landmark -> PointStamped
            pt = PointStamped()
            pt.header = self.map_landmarks.header
            pt.point = landmark.position

            # Transform into base_footprint
            pt_robot = do_transform_point(pt, transform)
            lx = pt_robot.point.x
            ly = pt_robot.point.y


            pose = Pose()

            # Polar coordinates in robot frame
            r_landmark = np.hypot(lx, ly)
            theta_landmark = np.arctan2(ly, lx) + 2*np.pi if np.arctan2(ly, lx) < 0 else np.arctan2(ly, lx)
            # print theta landmark and angle min and max to see if they match
            # self.get_logger().info(f"Landmark: r={r_landmark}, theta={theta_landmark}, angle_min={angle_min}, angle_max={angle_max}")

            # Check if landmark is within LiDAR FOV
            if angle_min <= theta_landmark <= angle_max and r_min <= r_landmark <= r_max:
                # Find corresponding LiDAR index
                idx = int((theta_landmark - angle_min) / msg.angle_increment)
                idx = np.clip(idx, 0, len(msg.ranges) - 1)

                # Look at nearby beams (±2) for robustness
                neighbor_idxs = range(max(0, idx-2), min(len(msg.ranges), idx+3))
                r_measured = np.array([msg.ranges[i] for i in neighbor_idxs])
                r_measured = r_measured[np.isfinite(r_measured)]  # filter out inf/NaN

                visible = False
                if len(r_measured) > 0:
                    # Check if LiDAR saw something close to landmark distance
                    if np.min(np.abs(r_measured - r_landmark)) < 0.2:  # tolerance = 20 cm
                        visible = True

                if visible:
                    pose.position.x = r_landmark + np.random.normal(0, 0.05)  # add small noise
                    pose.position.y = 0.0
                    pose.position.z = theta_landmark + np.random.normal(0, 0.05)  # add small noise
                else:
                    pose.position.x = 0.0
                    pose.position.y = 0.0
                    pose.position.z = 0.0
            else:
                pose.position.x = 0.0
                pose.position.y = 0.0
                pose.position.z = 0.0

            obs_array.poses.append(pose)

        self.pub_obs.publish(obs_array)

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkObserver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
