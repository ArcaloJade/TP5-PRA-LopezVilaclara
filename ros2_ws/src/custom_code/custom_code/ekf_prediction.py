#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from custom_msgs.msg import Belief, DeltaOdom
from geometry_msgs.msg import Pose2D
import numpy as np


class EKFPrediction(Node):
    """Nodo que implementa el paso de PREDICCIÓN del EKF.

    - Se subscribe a /belief (Belief) para recibir el belief inicial y futuras correcciones.
    - Se subscribe a /delta (DeltaOdom) para recibir actualizaciones de odometría en formato deltas.
    - Publica en /belief (Belief) las predicciones del belief.
    """

    def __init__(self):
        super().__init__('ekf_prediction')

        # Publicador y suscriptores
        self.belief_pub = self.create_publisher(Belief, '/belief', 10)
        self.belief_sub = self.create_subscription(Belief, '/belief', self._belief_callback, 10)
        self.delta_sub = self.create_subscription(DeltaOdom, '/delta', self._delta_callback, 10)

        # Estado interno: mu (x,y,theta) y Sigma (3x3)
        self.have_belief = False
        self.mu = np.zeros(3)
        self.Sigma = np.eye(3) * 1e-6   # La inicializo con valores muy chicos

        self.get_logger().info('EKFPrediction inicializado')

    def _belief_callback(self, msg: Belief) -> None:
        """Recibe beliefs y actualiza el estado interno sin predecir."""
        self.mu[0] = msg.mu.x
        self.mu[1] = msg.mu.y
        self.mu[2] = msg.mu.theta

        try:
            cov_list = list(msg.covariance)
            if len(cov_list) != 9:
                raise ValueError('Belief.covariance length != 9')
            self.Sigma = np.array(cov_list).reshape((3, 3))
        except Exception as e:
            self.get_logger().warn('Error al leer covariance del belief: %s.' % str(e))
            self.Sigma = np.eye(3) * 1e-6

        self.have_belief = True
        self.get_logger().info('Belief recibido: x=%.3f y=%.3f theta=%.3f' % (self.mu[0], self.mu[1], self.mu[2]))

    def _delta_callback(self, msg: DeltaOdom) -> None:
        """Recibe deltas de odometría y realiza el paso de predicción (si ya se tiene belief inicial).

        Implementa las ecuaciones:
            δ_trans, δ_rot1, δ_rot2
            μ̄ = μ + [δ_trans cos(θ+δ_rot1); δ_trans sin(θ+δ_rot1); δ_rot1+δ_rot2]
            G_t = una matriz Jacobiana 3x3
            Q_t = matriz de ruido del movimiento, tambien 3x3, nos la dan en consigna
            Σ̄ = G Σ G^T + Q_t
        """
        if not self.have_belief:
            self.get_logger().warn('Delta recibido pero no hay belief inicial. Ignorando delta.') # Si veo esto hay algo mal, seguro corrí las terminales en orden equivocado
            return

        delta_rot1 = float(msg.dr1)
        delta_rot2 = float(msg.dr2)
        delta_trans = float(msg.dt)

        theta = self.mu[2]
        theta_mid = theta + delta_rot1

        # Predicción del estado (ecuación 6 del pseudocódigo de Nacho)
        mu_bar = self.mu.copy()
        mu_bar[0] += delta_trans * np.cos(theta_mid)
        mu_bar[1] += delta_trans * np.sin(theta_mid)
        mu_bar[2] = self._normalize_angle(self.mu[2] + (delta_rot1 + delta_rot2))   # Normalizo por si acaso

        # Jacobiano G_t (ecuación 3 del pseudocódigo de Nacho)
        G = np.eye(3)
        G[0, 2] = -delta_trans * np.sin(theta_mid)
        G[1, 2] = delta_trans * np.cos(theta_mid)

        # Matriz de ruido de movim. Q_t (consigna), en el pseudocódigo de Nacho sería el "equivalente" a las ec. 4 y 5 que calculan V, M
        Q_t = np.eye(3) * 0.02

        # Predicción de la covarianza: Sigma_gorrito = G Sigma G^T + Q_t (ec. 7 del pseudocódigo de Nacho)
        Sigma_bar = G.dot(self.Sigma).dot(G.T) + Q_t

        # Actualizamos estado interno
        self.mu = mu_bar
        self.Sigma = Sigma_bar

        # Publicamos belief predicho
        belief_msg = Belief()
        belief_msg.mu = Pose2D()
        belief_msg.mu.x = float(self.mu[0])
        belief_msg.mu.y = float(self.mu[1])
        belief_msg.mu.theta = float(self.mu[2])
        belief_msg.covariance = [float(x) for x in self.Sigma.reshape(9).tolist()]

        self.belief_pub.publish(belief_msg)

        self.get_logger().info('Predicción publicada: x=%.3f y=%.3f theta=%.3f' % (self.mu[0], self.mu[1], self.mu[2]))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normaliza el ángulo a [-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


def main(args=None):
    rclpy.init(args=args)
    node = EKFPrediction()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
