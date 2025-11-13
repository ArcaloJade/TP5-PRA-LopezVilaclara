#!/usr/bin/env python3
# fastslam_correct.py
"""
FastSlamNode - implementación del paso de corrección de FastSLAM
- Nodo: 'fastslam_node'
- Suscribe a:
    /delta (DeltaOdom) -- mensaje custom con deltas de odometría (supongo campos dx, dy, dtheta)
    /observed_landmarks (PoseArray) -- cada pose: position.x = range, position.z = bearing
- Publica:
    /fastslam/landmarks_markers (MarkerArray) -- landmarks + covariances de la mejor partícula
    /fastslam/pose (PoseStamped) -- pose estimada del robot (mejor partícula)
- Parámetros / supuestos:
    alphas = [0.2, 0.2, 0.001, 0.001]
    measurement std dev = 0.05 (m, rad)
    p0 (new-feature prior) = small constant (0.001)
    perceptual_range = 5.0 (m)  -- si querés otro valor, cambiar la constante
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
import random

# ROS messages
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

# Import custom odom delta message - adjust import path if different
# Asumiendo mensaje custom llamado DeltaOdom con campos dx, dy, dtheta
try:
    from custom_msgs.msg import DeltaOdom
except Exception:
    # Si el import falla en pruebas locales, crea un fallback dummy tipo
    DeltaOdom = None

# Utility: normalize angle to [-pi, pi]
def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class FastSlamNode(Node):
    def __init__(self):
        super().__init__('fastslam_node')

        # Parameters / constants from consigna
        self.alphas = [0.2, 0.2, 0.001, 0.001]
        self.meas_std_range = 0.05
        self.meas_std_bearing = 0.05
        self.Q_t = np.diag([self.meas_std_range**2, self.meas_std_bearing**2])
        self.p0 = 1e-3  # prior weight for new feature
        self.M = 30     # number of particles (puedes ajustar)
        self.perceptual_range = 5.0

        # Particle set Y: list of particle dicts
        # particle = {
        #   'pose': np.array([x,y,theta]),
        #   'landmarks': { id: {'mu': np.array([mx,my]), 'sigma': 2x2, 'count': int} },
        #   'weight': float
        # }
        self.particles = [self._make_empty_particle() for _ in range(self.M)]

        # Storage for last messages
        self.last_delta = None
        self.last_obs = None

        # Subscribers
        self.create_subscription(PoseArray, '/observed_landmarks', self.observed_cb, 10)
        if DeltaOdom is not None:
            self.create_subscription(DeltaOdom, '/delta', self.delta_cb, 10)
        else:
            # If custom message not present, subscribe to a fallback topic name
            self.get_logger().warn("DeltaOdom msg import failed; ensure custom_msgs.msg.DeltaOdom is available.")
            # Still create a dummy subscription to avoid failing (it won't receive)
            # You should replace when running in actual environment.

        # Publishers
        self.landmark_pub = self.create_publisher(MarkerArray, '/fastslam/landmarks_markers', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/fastslam/pose', 10)

        self.get_logger().info('FastSlamNode initialized.')

    def _make_empty_particle(self):
        return {'pose': np.array([0.0, 0.0, 0.0]), 'landmarks': dict(), 'weight': 1.0 / max(1, self.M)}

    #
    # Callbacks: store last messages and trigger a correction step when we have both
    #
    def delta_cb(self, msg):
        # Assumo campos dx, dy, dtheta. Si tu DeltaOdom usa otros nombres, adaptá acá.
        try:
            self.last_delta = (float(msg.dx), float(msg.dy), float(msg.dtheta))
        except Exception:
            # Try alternative fields (x,y,theta)
            try:
                self.last_delta = (float(msg.delta_x), float(msg.delta_y), float(msg.delta_theta))
            except Exception:
                self.get_logger().warn("DeltaOdom fields not recognized. Please ensure dx,dy,dtheta or delta_x,...")
                self.last_delta = None
        # If we already have observations, run correction
        if self.last_obs is not None and self.last_delta is not None:
            self.run_fastslam_step(self.last_delta, self.last_obs)
            # reset last_obs so we don't re-process same observation (obs arrive at sensor rate)
            self.last_obs = None

    def observed_cb(self, msg: PoseArray):
        # Store observations: convert PoseArray to list of (range, bearing)
        obs = []
        for p in msg.poses:
            r = float(p.position.x)
            # consigna: angle stored in z
            b = float(p.position.z)
            obs.append(np.array([r, b]))
        self.last_obs = obs
        # If we already have delta, run correction
        if self.last_delta is not None and self.last_obs is not None:
            self.run_fastslam_step(self.last_delta, self.last_obs)
            self.last_obs = None

    #
    # FastSLAM main correction step following your pseudocode
    #
    def run_fastslam_step(self, u_t, observations):
        """
        u_t: tuple (dx, dy, dtheta) -- odometry delta in robot frame (assumption)
        observations: list of np.array([range, bearing])
        """
        M = self.M
        Y_aux = []
        weights = np.zeros(M)

        # For each particle, propagate pose (sample from motion model) and process observations
        for k in range(M):
            p = self.particles[k]
            x_prev = p['pose'].copy()
            # 3: sample new pose
            x_new = self.sample_motion_model(x_prev, u_t)
            p_new = {'pose': x_new.copy(), 'landmarks': dict(), 'weight': 1.0}

            # copy existing landmarks initially
            for lm_id, lm in p['landmarks'].items():
                p_new['landmarks'][lm_id] = {'mu': lm['mu'].copy(), 'sigma': lm['sigma'].copy(), 'count': int(lm['count'])}

            # Importance weight for this particle (product of observation weights)
            particle_weight = 1.0

            # For each observation z_t (assuming independent measurements)
            for z_t in observations:
                # For current particle, compute w_j for each existing landmark
                landmark_ids = sorted(p_new['landmarks'].keys())
                w_js = []
                H_list = []
                zhat_list = []
                sig_list = []

                for lm_id in landmark_ids:
                    lm = p_new['landmarks'][lm_id]
                    mu = lm['mu']
                    sigma = lm['sigma']
                    zhat = self.measurement_prediction(mu, x_new)
                    H = self.jacobian_wrt_landmark(mu, x_new)
                    Q_j = H @ sigma @ H.T + self.Q_t
                    # innovation:
                    nu = z_t - zhat
                    nu[1] = normalize_angle(nu[1])
                    # gaussian likelihood
                    try:
                        denom = 2 * np.pi * np.sqrt(np.linalg.det(Q_j))
                        exponent = -0.5 * (nu.T @ np.linalg.inv(Q_j) @ nu)
                        wj = (1.0 / denom) * np.exp(exponent)
                    except Exception:
                        wj = 1e-12
                    w_js.append(wj)
                    H_list.append(H)
                    zhat_list.append(zhat)
                    sig_list.append(Q_j)

                # weight for new feature
                w_new = self.p0
                w_js.append(w_new)

                # find best correspondence
                j_max = int(np.argmax(w_js))
                w_c = float(w_js[j_max])
                # multiply into particle weight
                particle_weight *= w_c

                # Decide if new feature or existing (note: j_max index corresponds to either an lm index or new)
                if j_max == len(landmark_ids):
                    # new feature: initialize with h^{-1}
                    new_id = 1
                    if len(landmark_ids) > 0:
                        new_id = max(landmark_ids) + 1
                    mu_init = self.inverse_measurement(z_t, x_new)
                    Hj = self.jacobian_wrt_landmark(mu_init, x_new)
                    # initialize covariance: (H^{-1})^T Q H^{-1}
                    try:
                        Hj_inv = np.linalg.inv(Hj)
                        sigma_init = Hj_inv.T @ self.Q_t @ Hj_inv
                    except Exception:
                        # fallback: large uncertainty
                        sigma_init = np.eye(2) * 1.0
                    p_new['landmarks'][new_id] = {'mu': mu_init.copy(), 'sigma': sigma_init.copy(), 'count': 1}
                else:
                    # observed existing feature: perform EKF update
                    lm_id = landmark_ids[j_max]
                    lm = p_new['landmarks'][lm_id]
                    mu_prev = lm['mu']
                    sigma_prev = lm['sigma']
                    H = H_list[j_max]
                    Qj = sig_list[j_max]
                    zhat = zhat_list[j_max]
                    # Kalman gain
                    try:
                        K = sigma_prev @ H.T @ np.linalg.inv(Qj)
                    except Exception:
                        K = np.zeros((2,2))
                    nu = z_t - zhat
                    nu[1] = normalize_angle(nu[1])
                    mu_upd = mu_prev + K @ nu
                    sigma_upd = (np.eye(2) - K @ H) @ sigma_prev
                    p_new['landmarks'][lm_id]['mu'] = mu_upd
                    p_new['landmarks'][lm_id]['sigma'] = sigma_upd
                    p_new['landmarks'][lm_id]['count'] = p_new['landmarks'][lm_id]['count'] + 1

                # For all other features not observed, apply counter decrement logic:
                for lm_id in list(p_new['landmarks'].keys()):
                    if (len(landmark_ids) > 0 and lm_id in landmark_ids and
                        lm_id != (landmark_ids[j_max] if j_max < len(landmark_ids) else None)):
                        # not the observed feature this iteration -> check perceptual range
                        mu = p_new['landmarks'][lm_id]['mu']
                        dist = np.linalg.norm(mu - x_new[0:2])
                        if dist <= self.perceptual_range:
                            # should have been seen but wasn't -> decrement
                            p_new['landmarks'][lm_id]['count'] -= 1
                        else:
                            # outside range -> do not change count
                            pass
                        # discard if below zero
                        if p_new['landmarks'][lm_id]['count'] < 0:
                            del p_new['landmarks'][lm_id]

            # After processing all observations for this particle, set weight and add to Y_aux
            p_new['weight'] = particle_weight
            Y_aux.append(p_new)
            weights[k] = particle_weight

        # Normalize weights (avoid division by zero)
        if np.sum(weights) <= 0:
            # Assign uniform weights to avoid degeneracy
            weights = np.ones(M) / M
            for k in range(M):
                Y_aux[k]['weight'] = weights[k]
        else:
            weights = weights / np.sum(weights)
            for k in range(M):
                Y_aux[k]['weight'] = weights[k]

        # Resampling: draw M times according to weights
        indices = self.multinomial_resample(weights, M)
        Y_t = []
        for idx in indices:
            # Deep copy selected particle into new set
            sel = Y_aux[idx]
            new_particle = {'pose': sel['pose'].copy(),
                            'landmarks': {},
                            'weight': 1.0 / M}
            for lm_id, lm in sel['landmarks'].items():
                new_particle['landmarks'][lm_id] = {'mu': lm['mu'].copy(), 'sigma': lm['sigma'].copy(), 'count': int(lm['count'])}
            Y_t.append(new_particle)

        # Replace particle set
        self.particles = Y_t

        # Publish best particle's landmarks and pose
        best_idx = int(np.argmax([p['weight'] for p in self.particles]))
        best_particle = self.particles[best_idx]
        self.publish_best_particle(best_particle)

    #
    # Motion sampling: p(x_t | x_{t-1}, u_t)
    # u_t assumed (dx, dy, dtheta) in robot frame. We add noise according to alphas.
    #
    def sample_motion_model(self, x_prev, u_t):
        dx, dy, dtheta = u_t
        # transform odometry delta into world frame using previous theta
        theta = x_prev[2]
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        delta_world = R @ np.array([dx, dy])
        # Add noise: approximation — add gaussian noise proportional to alphas and magnitudes
        transl_mag = np.linalg.norm([dx, dy])
        std_trans = self.alphas[0] * abs(transl_mag) + self.alphas[1] * abs(dtheta)
        std_rot = self.alphas[2] * abs(dtheta) + self.alphas[3] * abs(transl_mag)
        noisy_tx = delta_world[0] + np.random.normal(0, std_trans)
        noisy_ty = delta_world[1] + np.random.normal(0, std_trans)
        noisy_dtheta = dtheta + np.random.normal(0, std_rot)
        x_new = np.array([x_prev[0] + noisy_tx, x_prev[1] + noisy_ty, normalize_angle(theta + noisy_dtheta)])
        return x_new

    #
    # Measurement model h(mu, x): given landmark mu = [mx,my] and robot pose x=[x,y,theta],
    # return predicted measurement [range, bearing]
    #
    def measurement_prediction(self, mu, x):
        dx = mu[0] - x[0]
        dy = mu[1] - x[1]
        r = math.hypot(dx, dy)
        b = normalize_angle(math.atan2(dy, dx) - x[2])
        return np.array([r, b])

    #
    # Jacobian of h with respect to landmark mu (2x2)
    #
    def jacobian_wrt_landmark(self, mu, x):
        dx = mu[0] - x[0]
        dy = mu[1] - x[1]
        q = dx*dx + dy*dy
        r = math.sqrt(q) if q > 1e-12 else 1e-6
        # H = [ [dx/r, dy/r],
        #       [-dy/q, dx/q] ]
        H = np.array([[dx / r, dy / r],
                      [-dy / q, dx / q]])
        return H

    #
    # Inverse measurement h^{-1}(z, x): given range,bearing and robot pose returns landmark position mu
    #
    def inverse_measurement(self, z, x):
        r = z[0]
        b = z[1]
        angle = x[2] + b
        mx = x[0] + r * math.cos(angle)
        my = x[1] + r * math.sin(angle)
        return np.array([mx, my])

    #
    # Resampling utility: multinomial resampling
    #
    def multinomial_resample(self, weights, M):
        # weights assumed normalized
        cum = np.cumsum(weights)
        indices = []
        for _ in range(M):
            u = random.random()
            idx = int(np.searchsorted(cum, u))
            if idx >= len(weights):
                idx = len(weights) - 1
            indices.append(idx)
        return indices

    #
    # Publishers & marker helpers (from consigna)
    #
    def quaternion_from_yaw(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def make_landmark_marker(self, idx, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = idx
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
        # Compute ellipse parameters
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = math.atan2(vecs[1, 0], vecs[0, 0])
        scale_x = 30 * 2 * math.sqrt(max(vals[0], 1e-12))
        scale_y = 30 * 2 * math.sqrt(max(vals[1], 1e-12))
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = idx
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        q = self.quaternion_from_yaw(angle)
        m.pose.orientation = q
        m.scale.x = float(scale_x)
        m.scale.y = float(scale_y)
        m.scale.z = 0.01
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 0.3
        return m

    def publish_best_particle(self, particle):
        # Publish PoseStamped for robot pose
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(particle['pose'][0])
        ps.pose.position.y = float(particle['pose'][1])
        quat = self.quaternion_from_yaw(float(particle['pose'][2]))
        ps.pose.orientation = quat
        self.pose_pub.publish(ps)

        # Prepare MarkerArray for landmarks
        ma = MarkerArray()
        idx = 0
        for lm_id, lm in sorted(particle['landmarks'].items()):
            mu = lm['mu']
            sigma = lm['sigma']
            ma.markers.append(self.make_landmark_marker(idx*2, mu[0], mu[1]))
            ma.markers.append(self.make_covariance_marker(idx*2+1, mu[0], mu[1], sigma))
            idx += 1
        self.landmark_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = FastSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
