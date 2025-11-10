import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import cv2

class FeatureExtractor(Node):
    def __init__(self):
        super().__init__('feature_extractor')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        self.publisher = self.create_publisher(PoseArray, '/landmarks', 10)

    def map_callback(self, msg: OccupancyGrid):
        # Convert occupancy grid to numpy image
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))
        
        # Simple feature extraction: Harris corners
        img = np.uint8((data < 50) * 255)  # occupied cells -> white
        corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            pose_array = PoseArray()
            pose_array.header = msg.header
            for c in corners:
                x, y = c.ravel()
                # Convert from pixel to world coordinates
                wx = msg.info.origin.position.x + x * msg.info.resolution
                wy = msg.info.origin.position.y + y * msg.info.resolution
                p = Pose()
                p.position.x = wx
                p.position.y = wy
                pose_array.poses.append(p)
            
            self.publisher.publish(pose_array)
            self.get_logger().info(f"Published {len(pose_array.poses)} landmarks")

def main(args=None):
    rclpy.init(args=args)
    node = FeatureExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
