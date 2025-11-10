import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np


class LikelihoodMapPublisher(Node):
    def __init__(self):
        super().__init__('likelihood_map_publisher')
        qos = rclpy.qos.QoSProfile(depth=1)
        qos.durability = rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(OccupancyGrid, '/likelihood_map', qos)
        self.sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos
        )

    def map_callback(self, msg):
        prob_msg = OccupancyGrid()
        prob_msg.header = msg.header
        prob_msg.info = msg.info

        # Dimensiones del mapa
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        map_data = np.array(msg.data, dtype=int).reshape((height, width))

        # Inicializo la matriz del log-likelihood
        likelihood = np.full((height, width), -np.inf, dtype=np.float32)

        # Coordenadas de celdas ocupadas
        occupied_coords = np.argwhere(map_data == 100)

        if len(occupied_coords) == 0:
            # (Si pasa esto hay un problema con el mapa)
            likelihood.fill(-np.inf)
        else:
            sigma = 0.5  # desviación estándar para la gaussiana
            # Para cada celda libre, calculo la distancia mínima a una celda ocupada
            for i in range(height):
                for j in range(width):
                    if map_data[i, j] == 0:  # celdas libres
                        # Calculo distancias a todas las celdas ocupadas
                        distances = np.sqrt((occupied_coords[:, 0] - i)**2 + (occupied_coords[:, 1] - j)**2)
                        min_dist = np.min(distances) * resolution

                        # Probabilidad (Gaussiana)
                        prob = np.exp(-(min_dist**2) / (2 * sigma**2))

                        # Log-likelihood
                        likelihood[i, j] = np.log(prob + 1e-9)  # evitar log(0)
                    elif map_data[i, j] == 100:
                        likelihood[i, j] = np.log(1.0)  # máxima probabilidad
                    else:
                        likelihood[i, j] = np.log(1e-6)  # desconocidas → muy baja prob.

        # Escalo los valores de log-likelihood al rango [0, 100] para publicarlos
        finite_mask = np.isfinite(likelihood)
        if np.any(finite_mask):
            log_min = np.min(likelihood[finite_mask])
            log_max = np.max(likelihood[finite_mask])
            scaled = 100 * (likelihood - log_min) / (log_max - log_min + 1e-9)
            scaled = np.clip(scaled, 0, 100)
        else:
            scaled = np.zeros_like(likelihood)

        prob_msg.data = scaled.astype(np.uint8).flatten().tolist()

        self.pub.publish(prob_msg)
        self.get_logger().info("Published likelihood map")

def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()