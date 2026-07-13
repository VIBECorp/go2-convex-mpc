"""
Demo 07: Proprioceptive state estimation (leg-odometry Kalman filter)

The Go2 trots while a LegOdometryKF estimates base position/velocity using ONLY
noisy IMU + joint encoder signals (no ground truth). At the end, the estimate
is compared against MuJoCo ground truth.

USE_ESTIMATOR selects what drives the controller:
    True  (default) : CLOSED LOOP — the MPC/leg controller runs on the KF
                      estimate + noisy encoders, like on the real robot
    False           : open loop  — the controller runs on ground truth and the
                      KF is only evaluated in parallel

Command schedule: forward trot -> trot in place -> forward + yaw turn.
"""
import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import numpy as np
import pinocchio as pin
from dataclasses import dataclass

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait import Gait
from convex_mpc.state_estimator import LegOdometryKF
from convex_mpc.plot_helper import hold_until_all_fig_closed

# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------

INITIAL_X_POS = -5
INITIAL_Y_POS = 0
RUN_SIM_LENGTH_S = 6.0

# True: controller runs on the KF estimate (real-robot configuration)
# False: controller runs on ground truth, KF evaluated in parallel only
USE_ESTIMATOR = True

# Locomotion command schedule
@dataclass
class BodyCmdPhase:
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float

CMD_SCHEDULE = [
    BodyCmdPhase(0.0, 2.0, 0.5, 0.0, 0.27, 0.0),   # forward
    BodyCmdPhase(2.0, 3.5, 0.0, 0.0, 0.27, 0.0),   # in place
    BodyCmdPhase(3.5, 6.0, 0.5, 0.0, 0.27, 0.5),   # forward + turn
]

# Gait Setting
GAIT_HZ = 3
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

# Sensor noise injected on top of the MuJoCo signals (the estimator sees ONLY these)
RNG_SEED = 7
SIGMA_GYRO = 0.01       # rad/s
SIGMA_ACC = 0.10        # m/s^2
ACC_BIAS = np.array([0.05, -0.03, 0.08])  # constant accelerometer bias [m/s^2]
SIGMA_ATT = 0.002       # rad, attitude error of the IMU orientation output
SIGMA_ENC_Q = 0.001     # rad
SIGMA_ENC_DQ = 0.01     # rad/s

# Rates
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ

SIM_STEPS = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

# Go2 joint torque limit
TAU_LIM = 0.9 * np.array([23.7, 23.7, 45.43] * 4)

LEG_SLICE = {"FL": slice(0, 3), "FR": slice(3, 6), "RL": slice(6, 9), "RR": slice(9, 12)}

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------
def get_body_cmd(t: float):
    for phase in CMD_SCHEDULE:
        if phase.t_start <= t < phase.t_end:
            return phase.x_vel, phase.y_vel, phase.z_pos, phase.yaw_rate
    return 0.0, 0.0, 0.27, 0.0


def read_noisy_sensors(m_go2: MuJoCo_GO2_Model, rng: np.random.Generator):
    """IMU + encoder signals as the estimator would see them on the real robot."""
    model, data = m_go2.model, m_go2.data

    def sensor(name):
        s = model.sensor(name)
        return data.sensordata[s.adr[0]: s.adr[0] + s.dim[0]].copy()

    # IMU attitude (framequat gives w,x,y,z) with a small random tilt error
    qw, qx, qy, qz = sensor("imu_quat")
    R_wb = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    R_wb = R_wb @ pin.exp3(rng.normal(0.0, SIGMA_ATT, 3))

    gyro = sensor("imu_gyro") + rng.normal(0.0, SIGMA_GYRO, 3)
    acc = sensor("imu_acc") + ACC_BIAS + rng.normal(0.0, SIGMA_ACC, 3)

    # Joint encoders, mapped to pinocchio order (FL,FR,RL,RR)
    q_j = data.qpos[m_go2.joint_qpos_adr] + rng.normal(0.0, SIGMA_ENC_Q, 12)
    dq_j = data.qvel[m_go2.joint_dof_adr] + rng.normal(0.0, SIGMA_ENC_DQ, 12)

    return R_wb, gyro, acc, q_j, dq_j

# --------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------
time_log = np.zeros(CTRL_STEPS)
pos_true = np.zeros((3, CTRL_STEPS))
vel_true = np.zeros((3, CTRL_STEPS))
pos_est = np.zeros((3, CTRL_STEPS))
vel_est = np.zeros((3, CTRL_STEPS))

# --------------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------------
go2 = PinGo2Model()
mujoco_go2 = MuJoCo_GO2_Model()
leg_controller = LegController()
traj = ComTraj(go2)
gait = Gait(GAIT_HZ, GAIT_DUTY)
rng = np.random.default_rng(RNG_SEED)

q_init = go2.current_config.get_q()
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
mujoco_go2.update_with_q_pin(q_init)
mujoco_go2.model.opt.timestep = SIM_DT

traj.generate_traj(go2, gait, 0.0, 0.0, 0.0, 0.27, 0.0, time_step=MPC_DT)
mpc = CentroidalMPC(go2, traj)
U_opt = np.zeros((12, traj.N), dtype=float)

# Estimator: initialized at the robot's own start pose (a real robot defines
# its odometry origin the same way); everything after this uses sensors only
estimator = LegOdometryKF(go2, CTRL_DT)
R0, _, _, qj0, _ = read_noisy_sensors(mujoco_go2, rng)
estimator.reset(mujoco_go2.data.qpos[0:3].copy(), R0, qj0)

# --------------------------------------------------------------------------------
# Simulation Loop
# --------------------------------------------------------------------------------
print(f"Running simulation for {RUN_SIM_LENGTH_S}s")
sim_start_time = time.perf_counter()

ctrl_i = 0
tau_hold = np.zeros(12, dtype=float)

for k in range(SIM_STEPS):
    time_now_s = float(mujoco_go2.data.time)

    if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
        x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body = get_body_cmd(time_now_s)

        # ---- Estimator (noisy IMU + encoders ONLY) ----
        R_wb, gyro, acc, q_j, dq_j = read_noisy_sensors(mujoco_go2, rng)
        # Trust a foot only in the middle of its scheduled stance: right after
        # touchdown / before liftoff the real contact state lags the schedule
        # and the foot still moves, which would bias the velocity measurement
        GUARD_S = 0.03
        contact_mask = (gait.compute_current_mask(time_now_s).reshape(-1)
                        & gait.compute_current_mask(time_now_s - GUARD_S).reshape(-1)
                        & gait.compute_current_mask(time_now_s + GUARD_S).reshape(-1))
        estimator.update(R_wb, gyro, acc, q_j, dq_j, contact_mask)

        # ---- Controller state source ----
        if USE_ESTIMATOR:
            # Real-robot configuration: base state from the KF + IMU, joints
            # from the (noisy) encoders — no ground truth anywhere
            quat_xyzw = pin.Quaternion(R_wb).coeffs()   # (x, y, z, w)
            q_pin = np.concatenate([estimator.base_pos, quat_xyzw, q_j])
            v_body = R_wb.T @ estimator.base_vel        # pinocchio wants body frame
            dq_pin = np.concatenate([v_body, gyro, dq_j])
            go2.update_model(q_pin, dq_pin)
        else:
            mujoco_go2.update_pin_with_mujoco(go2)

        if (ctrl_i % STEPS_PER_MPC) == 0:
            print(f"\rSimulation Time: {time_now_s:.3f} s", end="", flush=True)
            traj.generate_traj(go2, gait, time_now_s,
                               x_vel_des_body, y_vel_des_body,
                               z_pos_des_body, yaw_rate_des_body,
                               time_step=MPC_DT)
            sol = mpc.solve_QP(go2, traj, False)
            N = traj.N
            w_opt = sol["x"].full().flatten()
            U_opt = w_opt[12 * N:].reshape((12, N), order="F")

        tau = np.zeros(12)
        for leg in ("FL", "FR", "RL", "RR"):
            out = leg_controller.compute_leg_torque(
                leg, go2, gait, U_opt[LEG_SLICE[leg], 0], time_now_s)
            tau[LEG_SLICE[leg]] = out.tau
        tau_hold = np.clip(tau, -TAU_LIM, TAU_LIM)

        # ---- Logs ----
        time_log[ctrl_i] = time_now_s
        pos_true[:, ctrl_i] = mujoco_go2.data.qpos[0:3]
        vel_true[:, ctrl_i] = mujoco_go2.data.qvel[0:3]
        pos_est[:, ctrl_i] = estimator.base_pos
        vel_est[:, ctrl_i] = estimator.base_vel
        ctrl_i += 1

    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
    mujoco_go2.set_joint_torque(tau_hold)
    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

sim_end_time = time.perf_counter()
print(f"\nSimulation ended. Elapsed: {sim_end_time - sim_start_time:.3f}s")

# --------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------
n = ctrl_i
t = time_log[:n]
pos_err = pos_est[:, :n] - pos_true[:, :n]
vel_err = vel_est[:, :n] - vel_true[:, :n]

def rms(e):
    return np.sqrt(np.mean(e**2))

print(f"\n--- Estimation errors vs ground truth "
      f"({'CLOSED loop: controller on KF estimate' if USE_ESTIMATOR else 'open loop: controller on ground truth'}) ---")
print(f"velocity RMS [m/s]:  vx={rms(vel_err[0]):.4f}  vy={rms(vel_err[1]):.4f}  vz={rms(vel_err[2]):.4f}")
print(f"height   RMS [m]  :  z ={rms(pos_err[2]):.4f}")
print(f"xy drift over {t[-1]:.1f} s [m]:  x={pos_err[0, -1]:+.4f}  y={pos_err[1, -1]:+.4f}")
dist = np.sum(np.linalg.norm(np.diff(pos_true[0:2, :n], axis=1), axis=0))
drift = np.linalg.norm(pos_err[0:2, -1])
print(f"traveled {dist:.2f} m, final xy drift = {drift:.4f} m ({100*drift/max(dist,1e-9):.1f} % of distance)")

# --------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
for i, lbl in enumerate("xyz"):
    axes[i, 0].plot(t, pos_true[i, :n], label="ground truth")
    axes[i, 0].plot(t, pos_est[i, :n], "--", label="KF estimate")
    axes[i, 0].set_ylabel(f"p{lbl} [m]")
    axes[i, 0].grid(True)
    axes[i, 1].plot(t, vel_true[i, :n], label="ground truth")
    axes[i, 1].plot(t, vel_est[i, :n], "--", label="KF estimate")
    axes[i, 1].set_ylabel(f"v{lbl} [m/s]")
    axes[i, 1].grid(True)
axes[0, 0].legend(loc="upper right")
axes[0, 0].set_title("Base position (world)")
axes[0, 1].set_title("Base velocity (world)")
axes[-1, 0].set_xlabel("Time [s]")
axes[-1, 1].set_xlabel("Time [s]")
fig.suptitle("Leg-Odometry KF vs Ground Truth (IMU + encoders only)"
             + (" — controller running on the estimate" if USE_ESTIMATOR else ""))
fig.tight_layout()
plt.show(block=False)

fig2, axes2 = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
for i, lbl in enumerate("xyz"):
    axes2[0].plot(t, pos_err[i], label=f"p{lbl}")
    axes2[1].plot(t, vel_err[i], label=f"v{lbl}")
axes2[0].set_ylabel("position error [m]")
axes2[1].set_ylabel("velocity error [m/s]")
axes2[1].set_xlabel("Time [s]")
for ax in axes2:
    ax.grid(True)
    ax.legend(loc="upper right")
fig2.suptitle("Estimation errors")
fig2.tight_layout()
plt.show(block=True)

hold_until_all_fig_closed()
