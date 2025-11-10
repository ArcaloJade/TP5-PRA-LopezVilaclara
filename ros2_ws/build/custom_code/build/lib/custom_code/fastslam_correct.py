#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Quaternion, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from custom_msgs.msg import DeltaOdom

# Parámetros propuestos
ALPHAS = [0.2, 0.2, 0.001, 0.001]
STD_RANGE = 0.05   # m
STD_ANGLE = 0.05   # rad
N_PARTICLES = 100

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def quaternion_from_yaw(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q

class Particle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, weight=1.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.weight = float(weight)
        # landmarks: dict id -> (mu: np.array([x,y]), sigma: 2x2 np.array)
        self.landmarks = {}

class FastSlamNode(Node):
    def __init__(self):
        super().__init__('fastslam_node')

        # Suscripciones
        self.create_subscription(DeltaOdom, '/delta', self.delta_callback, 10)
        self.create_subscription(PoseArray, '/observed_landmarks', self.observed_callback, 10)

        # Publicadores
        self.pose_pub = self.create_publisher(PoseStamped, '/fastslam/pose', 10)
        self.landmark_pub = self.create_publisher(MarkerArray, '/fastslam/landmarks', 10)

        # Partículas
        self.particles = [Particle() for _ in range(N_PARTICLES)]
        self.get_logger().info(f'fastslam_node: initialized {N_PARTICLES} particles')

        # Matriz Q
        self.Q = np.diag([STD_RANGE**2, STD_ANGLE**2])

        self.last_delta = None

        # Publico la mejor pose si no llegan obs
        self.create_timer(0.2, self.publish_best_particle)


    # Actualización de movimiento

    def delta_callback(self, msg: DeltaOdom):
        dr1 = float(msg.dr1)
        dt = float(msg.dt)
        dr2 = float(msg.dr2)
        self.motion_update(dr1, dt, dr2)

    def motion_update(self, delta_rot1, delta_trans, delta_rot2):
        a1, a2, a3, a4 = ALPHAS
        for p in self.particles:
            sd_rot1 = math.sqrt(a1 * (delta_rot1**2) + a2 * (delta_trans**2))
            sd_trans = math.sqrt(a3 * (delta_trans**2) + a4 * (delta_rot1**2 + delta_rot2**2))
            sd_rot2 = math.sqrt(a1 * (delta_rot2**2) + a2 * (delta_trans**2))

            noisy_rot1 = delta_rot1 + np.random.normal(0, sd_rot1)
            noisy_trans = delta_trans + np.random.normal(0, sd_trans)
            noisy_rot2 = delta_rot2 + np.random.normal(0, sd_rot2)

            p.x += noisy_trans * math.cos(p.theta + noisy_rot1)
            p.y += noisy_trans * math.sin(p.theta + noisy_rot1)
            p.theta = wrap_to_pi(p.theta + noisy_rot1 + noisy_rot2)

    # Observaciones
    def observed_callback(self, msg: PoseArray):
        """
        Assumes PoseArray where pose.position.x = range, pose.position.z = bearing (relative),
        and landmark identity is the index in msg.poses (replace if your data includes explicit IDs).
        """

        observations = []
        for i, pose in enumerate(msg.poses):
            r = float(pose.position.x)
            b = float(pose.position.z)
            observations.append((i, r, b))

        # Actualizo peso y landmarks de cada partícula
        for p in self.particles:
            new_weight = p.weight if p.weight > 0 else 1.0
            for (lm_id, r, b) in observations:
                b_rel = wrap_to_pi(b)
                if lm_id not in p.landmarks:
                    mx = p.x + r * math.cos(p.theta + b_rel)
                    my = p.y + r * math.sin(p.theta + b_rel)
                    J = np.array([
                        [math.cos(p.theta + b_rel), -r * math.sin(p.theta + b_rel)],
                        [math.sin(p.theta + b_rel),  r * math.cos(p.theta + b_rel)]
                    ])
                    sigma = J @ self.Q @ J.T
                    p.landmarks[lm_id] = (np.array([mx, my]), sigma)
                    # No hago actualizacion de peso pq es nuevo landmark
                else:
                    mu, sigma = p.landmarks[lm_id]
                    dx = mu[0] - p.x
                    dy = mu[1] - p.y
                    q = dx*dx + dy*dy
                    expected_r = math.sqrt(q)
                    expected_b = wrap_to_pi(math.atan2(dy, dx) - p.theta)
                    z_hat = np.array([expected_r, expected_b])

                    if expected_r == 0:
                        # Pongo esto para no dividir por cero cuando calculo H
                        continue
                    H = np.array([
                        [dx / expected_r, dy / expected_r],
                        [-dy / q,          dx / q]
                    ])

                    S = H @ sigma @ H.T + self.Q
                    # ganancia de Kalman
                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        S_inv = np.linalg.pinv(S)
                    K = sigma @ H.T @ S_inv

                    z = np.array([r, b_rel])
                    y_innov = z - z_hat
                    y_innov[1] = wrap_to_pi(y_innov[1])

                    mu_new = mu + K @ y_innov
                    sigma_new = (np.eye(2) - K @ H) @ sigma
                    p.landmarks[lm_id] = (mu_new, sigma_new)

                    detS = np.linalg.det(S)
                    if detS <= 0:
                        likelihood = 1e-12
                    else:
                        denom = 2 * math.pi * math.sqrt(detS)
                        exponent = -0.5 * (y_innov.T @ S_inv @ y_innov)
                        likelihood = math.exp(exponent) / denom
                        likelihood = max(likelihood, 1e-12)

                    new_weight *= likelihood

            p.weight = new_weight

        self.normalize_weights()
        self.resample_particles()

        self.publish_landmarks_of_best()

    # Funciones auxiliares para pesos y resampling
    def normalize_weights(self):
        ws = np.array([p.weight for p in self.particles], dtype=float)
        s = ws.sum()
        if s <= 0:
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)
        else:
            for p in self.particles:
                p.weight = p.weight / s

    def systematic_resample(self, weights):
        N = len(weights)
        positions = (np.arange(N) + np.random.random()) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def resample_particles(self):
        weights = np.array([p.weight for p in self.particles], dtype=float)
        if np.allclose(weights, 1.0 / len(weights)):
            return  # quiere decir que ya son uniformes!
        indexes = self.systematic_resample(weights)
        new_particles = []
        for idx in indexes:
            p = self.particles[idx]
            new_p = Particle(p.x, p.y, p.theta, weight=1.0/len(self.particles))
            new_p.landmarks = {lm_id: (mu.copy(), sigma.copy()) for lm_id, (mu, sigma) in p.landmarks.items()}
            new_particles.append(new_p)
        self.particles = new_particles

    # Cosas de mejor partícula y publicación
    def get_best_particle(self):
        return max(self.particles, key=lambda p: p.weight)

    def publish_best_particle(self):
        best = self.get_best_particle()
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = best.x
        ps.pose.position.y = best.y
        ps.pose.position.z = 0.0
        ps.pose.orientation = quaternion_from_yaw(best.theta)
        self.pose_pub.publish(ps)

    def make_landmark_marker(self, idx, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = int(idx)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        return m

    def make_covariance_marker(self, idx, x, y, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = math.atan2(vecs[1, 0], vecs[0, 0])
        scale_x = 30 * 2 * math.sqrt(max(vals[0], 1e-12))
        scale_y = 30 * 2 * math.sqrt(max(vals[1], 1e-12))
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = int(idx)
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.pose.orientation = quaternion_from_yaw(angle)
        m.scale.x = float(scale_x)
        m.scale.y = float(scale_y)
        m.scale.z = 0.01
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 0.3
        return m

    def publish_landmarks_of_best(self):
        best = self.get_best_particle()
        ma = MarkerArray()
        for lm_id, (lm_mu, lm_sigma) in best.landmarks.items():
            try:
                li = int(lm_id)
            except Exception:
                li = hash(lm_id) % 100000
            ma.markers.append(self.make_landmark_marker(li*2, lm_mu[0], lm_mu[1]))
            ma.markers.append(self.make_covariance_marker(li*2+1, lm_mu[0], lm_mu[1], lm_sigma))
        self.landmark_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = FastSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
