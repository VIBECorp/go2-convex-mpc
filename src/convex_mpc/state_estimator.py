import numpy as np
from .go2_robot_data import PinGo2Model

# --------------------------------------------------------------------------------
# Leg-Odometry Kalman Filter (MIT Cheetah 3 style)
# --------------------------------------------------------------------------------
#
# Estimates base position/velocity in the world frame using only proprioception:
#   - IMU: attitude (R_wb), angular velocity (body), linear acceleration (body,
#     specific force including gravity reaction)
#   - Joint encoders: q, dq of the 12 leg joints
#   - Contact schedule: which feet are in stance
#
# The attitude is taken from the IMU (or an upstream attitude filter) and treated
# as known, which makes the remaining filter LINEAR.
#
# State (18):  x = [ p (3), v (3), p_f1..p_f4 (12) ]   all in world frame
#
# Prediction (IMU):    a_w = R_wb a_imu + g
#                      p <- p + v dt + 1/2 a_w dt^2,   v <- v + a_w dt
#                      p_fi <- p_fi   (feet quasi-static; process noise inflated
#                                      for swing feet so they track kinematics)
#
# Measurements (encoders + stance assumption), all linear in the state:
#   1. relative foot position:  p_fi - p   =  R_wb fk_i(q)           (all feet)
#   2. base velocity:           v          = -R_wb (w x fk_i + J_i dq) (stance)
#   3. foot height:             p_fi,z     =  foot_radius              (stance)
#
# Swing-leg rows stay in the filter with inflated noise instead of being removed,
# so the matrix sizes are constant.
#
# Absolute x, y (and yaw) are unobservable and drift slowly; this is harmless for
# the convex MPC because it only uses position relative to the recent past
# (see docs/state_estimation.md).
# --------------------------------------------------------------------------------

GRAVITY = np.array([0.0, 0.0, -9.81])

LEGS = ("FL", "FR", "RL", "RR")


class LegOdometryKF:

    def __init__(self, robot: PinGo2Model, dt: float,
                 sigma_acc: float = 0.5,          # accel process noise [m/s^2]
                 sigma_foot_stance: float = 0.02, # stance foot random walk [m/s]
                 sigma_foot_swing: float = 1e3,   # swing foot: effectively forget
                 sigma_rel_pos: float = 0.005,    # kinematic rel-pos meas noise [m]
                 sigma_vel: float = 0.05,         # stance velocity meas noise [m/s]
                 sigma_height: float = 0.005,     # stance foot height meas noise [m]
                 sigma_swing_meas: float = 1e4):  # inflated noise for swing rows
        self.dt = dt
        self.foot_radius = robot.FOOT_RADIUS

        # Dedicated kinematics model: base fixed at identity so that foot
        # placements/Jacobians come out directly in the base frame
        self.kin = type(robot)()
        self._q_kin = self.kin.q_init.copy()
        self._dq_kin = self.kin.dq_init.copy()

        self.sigma_acc = sigma_acc
        self.sigma_foot_stance = sigma_foot_stance
        self.sigma_foot_swing = sigma_foot_swing
        self.sigma_rel_pos = sigma_rel_pos
        self.sigma_vel = sigma_vel
        self.sigma_height = sigma_height
        self.sigma_swing_meas = sigma_swing_meas

        # Constant matrices
        self.F = np.eye(18)
        self.F[0:3, 3:6] = dt * np.eye(3)

        # Measurement model H (28 x 18): [12 rel-pos, 12 velocity, 4 height]
        H = np.zeros((28, 18))
        for i in range(4):
            H[3*i:3*i+3, 0:3] = -np.eye(3)              # -p
            H[3*i:3*i+3, 6+3*i:9+3*i] = np.eye(3)       # +p_fi
            H[12+3*i:15+3*i, 3:6] = np.eye(3)           # v
            H[24+i, 6+3*i+2] = 1.0                      # p_fi,z
        self.H = H

        self.x = np.zeros(18)
        self.P = np.eye(18)

    # ----------------------------------------------------------------------------
    def _leg_kinematics(self, q_joints, dq_joints):
        """Foot positions and velocities in the BASE frame from encoders only."""
        self._q_kin[0:3] = 0.0
        self._q_kin[3:7] = [0.0, 0.0, 0.0, 1.0]
        self._q_kin[7:19] = q_joints
        self._dq_kin[0:6] = 0.0
        self._dq_kin[6:18] = dq_joints
        self.kin.update_model(self._q_kin, self._dq_kin)

        p_rel, v_rel = [], []
        for leg in LEGS:
            p, v = self.kin.get_single_foot_state_in_world(leg)  # base == world here
            p_rel.append(p)
            v_rel.append(v)
        return p_rel, v_rel

    # ----------------------------------------------------------------------------
    def reset(self, p0, R_wb, q_joints):
        """Initialize at known base position (e.g. the robot's own start pose)."""
        p_rel, _ = self._leg_kinematics(q_joints, np.zeros(12))

        self.x[:] = 0.0
        self.x[0:3] = p0
        for i in range(4):
            self.x[6+3*i:9+3*i] = p0 + R_wb @ p_rel[i]

        self.P = np.diag([1e-4]*3 + [1e-2]*3 + [1e-3]*12)

    # ----------------------------------------------------------------------------
    def update(self, R_wb, gyro_body, accel_body, q_joints, dq_joints, contact_mask):
        """
        One predict + correct cycle.

        R_wb        : (3,3) base-to-world rotation from the IMU / attitude filter
        gyro_body   : (3,)  angular velocity in base frame
        accel_body  : (3,)  accelerometer specific force in base frame
                      (at rest reads +9.81 on z)
        q_joints    : (12,) joint positions in pinocchio order (FL,FR,RL,RR)
        dq_joints   : (12,) joint velocities
        contact_mask: (4,)  1 = stance, 0 = swing (from the gait schedule or
                      a contact detector)
        """
        dt = self.dt

        # ---- Predict: strapdown IMU integration ----
        a_w = R_wb @ np.asarray(accel_body) + GRAVITY
        self.x[0:3] += self.x[3:6] * dt + 0.5 * a_w * dt * dt
        self.x[3:6] += a_w * dt

        Q = np.zeros((18, 18))
        Q[0:3, 0:3] = (0.5 * self.sigma_acc * dt * dt)**2 * np.eye(3)
        Q[3:6, 3:6] = (self.sigma_acc * dt)**2 * np.eye(3)
        for i in range(4):
            sig = self.sigma_foot_stance if contact_mask[i] else self.sigma_foot_swing
            Q[6+3*i:9+3*i, 6+3*i:9+3*i] = (sig * dt)**2 * np.eye(3)

        self.P = self.F @ self.P @ self.F.T + Q

        # ---- Measurements from leg kinematics ----
        p_rel, v_rel = self._leg_kinematics(q_joints, dq_joints)
        w = np.asarray(gyro_body)

        z = np.zeros(28)
        r_diag = np.zeros(28)
        for i in range(4):
            stance = bool(contact_mask[i])

            # 1) relative foot position (valid for every foot)
            z[3*i:3*i+3] = R_wb @ p_rel[i]
            r_diag[3*i:3*i+3] = self.sigma_rel_pos**2

            # 2) base velocity from the stationary-foot assumption
            z[12+3*i:15+3*i] = -R_wb @ (np.cross(w, p_rel[i]) + v_rel[i])
            r_diag[12+3*i:15+3*i] = (self.sigma_vel if stance else self.sigma_swing_meas)**2

            # 3) stance foot height on the ground plane
            z[24+i] = self.foot_radius
            r_diag[24+i] = (self.sigma_height if stance else self.sigma_swing_meas)**2

        # ---- Correct ----
        H = self.H
        S = H @ self.P @ H.T + np.diag(r_diag)
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(28))
        self.x += K @ (z - H @ self.x)
        I_KH = np.eye(18) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ np.diag(r_diag) @ K.T  # Joseph form

    # ----------------------------------------------------------------------------
    @property
    def base_pos(self):
        return self.x[0:3].copy()

    @property
    def base_vel(self):
        return self.x[3:6].copy()

    @property
    def foot_pos(self):
        return self.x[6:18].reshape(4, 3).copy()
