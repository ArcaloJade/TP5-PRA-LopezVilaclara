#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose2D
from custom_msgs.msg import Belief
import numpy as np


class EKFCorrection(Node):
    def __init__(self):
        super().__init__('ekf_correction')

        # Suscripciones y publicación
        self.sub_belief = self.create_subscription(Belief, '/belief', self._belief_callback, 10)
        self.sub_landmarks = self.create_subscription(PoseArray, '/landmarks', self._landmarks_callback, 10)
        self.sub_observed = self.create_subscription(PoseArray, '/observed_landmarks', self._observed_callback, 10)
        self.pub_belief = self.create_publisher(Belief, '/belief', 10)

        # Variables internas
        self.mu = None
        self.Sigma = None
        self.landmarks = None

        # Matriz de covarianza del ruido de medición (que la dan en consigna)
        sigma_r = 0.05
        sigma_phi = 0.05
        self.Q_t = np.array([[sigma_r**2, 0],
                             [0, sigma_phi**2]])

    def _belief_callback(self, msg: Belief):
        """Guarda el belief actual"""
        self.mu = np.array([msg.mu.x, msg.mu.y, msg.mu.theta])
        self.Sigma = np.array(msg.covariance).reshape((3, 3))

    def _landmarks_callback(self, msg: PoseArray):
        """Guarda las posiciones reales de los landmarks"""
        self.landmarks = [(pose.position.x, pose.position.y) for pose in msg.poses]

    def _observed_callback(self, msg: PoseArray):
        """Ejecuta la corrección EKF usando las observaciones"""
        if self.mu is None or self.Sigma is None or self.landmarks is None:
            return  # Espero datos iniciales

        mu = self.mu
        Sigma = self.Sigma

        for i, obs in enumerate(msg.poses):
            r_i, phi_i = obs.position.x, obs.position.z

            # Si no hay detección (pongo valores en 0)
            if r_i == 0.0 and phi_i == 0.0:
                continue

            # Posición del landmark correspondiente
            m_x, m_y = self.landmarks[i]

            # --- Cálculo del EKF Correction Step ---
            dx = m_x - mu[0]
            dy = m_y - mu[1]
            q = dx**2 + dy**2

            # z_hat = [sqrt(q), atan2(dy, dx) - mu_theta]
            z_hat = np.array([np.sqrt(q),
                              np.arctan2(dy, dx) - mu[2]])

            # Normalizo el ángulo de z_hat
            z_hat[1] = np.arctan2(np.sin(z_hat[1]), np.cos(z_hat[1]))

            # Jacobiano H_i
            H_i = np.array([
                [-dx / np.sqrt(q), -dy / np.sqrt(q), 0],
                [dy / q, -dx / q, -1]
            ])

            # S_i, K_i
            S_i = H_i @ Sigma @ H_i.T + self.Q_t
            K_i = Sigma @ H_i.T @ np.linalg.inv(S_i)

            # Residual (z_i - z_hat_i)
            z_i = np.array([r_i, phi_i])
            y_i = z_i - z_hat
            y_i[1] = np.arctan2(np.sin(y_i[1]), np.cos(y_i[1]))  # normalizo el ángulo

            # Actualización EKF
            mu = mu + K_i @ y_i
            Sigma = (np.eye(3) - K_i @ H_i) @ Sigma

        # Guardar y publicar belief corregido
        self.mu = mu
        self.Sigma = Sigma

        corrected_belief = Belief()
        corrected_belief.mu.x = mu[0]
        corrected_belief.mu.y = mu[1]
        corrected_belief.mu.theta = mu[2]
        corrected_belief.covariance = list(Sigma.flatten())

        self.pub_belief.publish(corrected_belief)

        self.get_logger().info(f"Publicado el belief corregido: μ={mu}, Σ={Sigma.tolist()}")


def main(args=None):
    rclpy.init(args=args)
    node = EKFCorrection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
