"""
Demo 09: Unitree A2 standing posture control from a randomized initial stance

Each foot of the A2 (~40 kg) starts displaced from its nominal (default-stand)
xy position by a random offset drawn uniformly from +/- 30 mm, computed via
per-leg inverse kinematics. The MPC uses the actual foot positions (lever
arms) so no controller change is needed; this demo shows posture control
remains stable on the asymmetric support polygon:
    0-3 s   : hold neutral posture
    3-6 s   : pitch nod        (+/- 0.15 rad)
    6-9 s   : roll sway        (+/- 0.15 rad)
    9-12 s  : height squat     (+/- 50 mm)
    12-16 s : y body sway      (+/- 30 mm, 0.25 Hz)

Set the environment variable EX09_FOOT_SEED to an integer for reproducible
foot offsets (default: random every run).
"""
import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import numpy as np
from dataclasses import dataclass

from convex_mpc.go2_robot_data import PinA2Model
from convex_mpc.mujoco_model import MuJoCo_A2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait import StandGait
from convex_mpc.plot_helper import plot_mpc_result, plot_solve_time, hold_until_all_fig_closed

# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------

# Simulation Setting
INITIAL_X_POS = -5
INITIAL_Y_POS = 0
RUN_SIM_LENGTH_S = 16.0

RENDER_HZ = 120.0
RENDER_DT = 1.0 / RENDER_HZ
REALTIME_FACTOR = 1

# Random foot-placement offsets
FOOT_OFFSET_MAX = 0.03      # m, uniform +/- 30 mm on each foot's x and y
SEED = os.environ.get("EX09_FOOT_SEED")
rng = np.random.default_rng(None if SEED is None else int(SEED))

# Nominal standing height
Z_NOMINAL = 0.40

# Posture command amplitudes
PITCH_AMP = 0.15    # rad
ROLL_AMP = 0.15     # rad
Z_AMP = 0.05        # m
Y_AMP = 0.03        # m
CMD_FREQ_HZ = 0.5   # frequency of the attitude/height oscillations
XY_FREQ_HZ = 0.25   # slower y sway (the MPC horizon extrapolates position linearly)

# Gait Setting (frequency only sets the MPC horizon; no leg ever swings)
GAIT_HZ = 3
GAIT_T = 1.0 / GAIT_HZ

# MPC cost: same posture-control weights as ex06 (the A2 is ~2.7x heavier than
# the Go2, so velocity damping weights are higher)
POSTURE_COST_Q = np.diag([200, 200, 200,  250, 250, 100,  80, 80, 5,  1, 1, 1])

#MuJoCo Sim Update Rate
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ

#Leg Coontroller Update Rate
CTRL_HZ = 200       # 200 Hz
CTRL_DT = 1.0 / CTRL_HZ

# Must be an integer ratio for clean decimation
if SIM_HZ % CTRL_HZ != 0:
    raise ValueError(
        f"SIM_HZ ({SIM_HZ}) must be divisible by CTRL_HZ ({CTRL_HZ}) for this decimation method."
    )
CTRL_DECIM = SIM_HZ // CTRL_HZ

SIM_STEPS = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

# Relation between MPC loop and control loop
MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))  # MPC update every N control ticks

# A2 Joint Torque Limit
HIP_LIM = 120.0
ABD_LIM = 120.0
KNEE_LIM = 180.0
SAFETY = 0.9

TAU_LIM = SAFETY * np.array([
    HIP_LIM, ABD_LIM, KNEE_LIM,   # FL: hip, thigh, calf
    HIP_LIM, ABD_LIM, KNEE_LIM,   # FR
    HIP_LIM, ABD_LIM, KNEE_LIM,   # RL
    HIP_LIM, ABD_LIM, KNEE_LIM,   # RR
])

LEGS = ("FL", "FR", "RL", "RR")
LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

# --------------------------------------------------------------------------------
# Randomized initial stance via per-leg inverse kinematics
# --------------------------------------------------------------------------------

def solve_leg_ik(kin: PinA2Model, leg: str, p_target_base, q_leg_init,
                 max_iter=100, tol=1e-8, damping=1e-6):
    """
    3-DOF damped-least-squares IK for one leg, base fixed at the identity so
    the kinematics model's world frame coincides with the base frame.
    Returns (joint angles, residual norm).
    """
    q = kin.q_init.copy()
    q[0:3] = 0.0
    q[3:7] = [0.0, 0.0, 0.0, 1.0]
    dq = np.zeros(18)
    q_leg = np.array(q_leg_init, dtype=float)
    err = np.zeros(3)

    for _ in range(max_iter):
        q[7:19][LEG_SLICE[leg]] = q_leg
        kin.update_model(q, dq)
        p, _ = kin.get_single_foot_state_in_world(leg)   # base frame here
        err = np.asarray(p_target_base) - p
        if np.linalg.norm(err) < tol:
            break
        J = kin.compute_3x3_foot_Jacobian_world(leg)     # base frame here
        step = np.linalg.solve(J.T @ J + damping * np.eye(3), J.T @ err)
        q_leg += step

    return q_leg, float(np.linalg.norm(err))


def sample_random_stance():
    """
    Sample per-leg xy offsets in +/- FOOT_OFFSET_MAX around the nominal stance
    and solve IK. Returns (12 joint angles, nominal and target foot positions
    in the base frame, (4,2) offsets).
    """
    kin = PinA2Model()                        # dedicated model, base at identity
    kin.update_model(np.concatenate([[0, 0, 0, 0, 0, 0, 1], kin.q_init[7:19]]),
                     np.zeros(18))
    p_nominal = {leg: kin.get_single_foot_state_in_world(leg)[0] for leg in LEGS}

    offsets = rng.uniform(-FOOT_OFFSET_MAX, FOOT_OFFSET_MAX, size=(4, 2))

    q_joints = np.zeros(12)
    p_target = {}
    for i, leg in enumerate(LEGS):
        p_target[leg] = p_nominal[leg] + np.array([offsets[i, 0], offsets[i, 1], 0.0])
        q_leg, residual = solve_leg_ik(kin, leg, p_target[leg],
                                       kin.DEFAULT_JOINT_ANGLES)
        if residual > 1e-6:
            raise RuntimeError(f"IK for {leg} did not converge (residual {residual:.2e} m)")
        q_joints[LEG_SLICE[leg]] = q_leg

    return q_joints, p_nominal, p_target, offsets


q_joints_init, p_foot_nominal, p_foot_target, foot_offsets = sample_random_stance()

print("Random foot xy offsets [mm] (uniform +/- 30):")
for i, leg in enumerate(LEGS):
    print(f"  {leg}: dx = {1e3*foot_offsets[i,0]:+6.1f}, dy = {1e3*foot_offsets[i,1]:+6.1f}")

# --------------------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------------------
@dataclass
class PostureCmd:
    roll: float = 0.0        # rad
    pitch: float = 0.0       # rad
    z_pos: float = Z_NOMINAL # m
    roll_rate: float = 0.0   # rad/s (feedforward)
    pitch_rate: float = 0.0  # rad/s (feedforward)
    y_off: float = 0.0       # m, body y offset from initial stance position
    y_vel: float = 0.0       # m/s (feedforward)
    z_vel: float = 0.0       # m/s (feedforward)


def get_posture_cmd(t: float) -> PostureCmd:
    """Posture command at time t; rates are analytic derivatives (feedforward)."""
    w = 2.0 * np.pi * CMD_FREQ_HZ
    cmd = PostureCmd()

    if t < 3.0:
        pass                                   # hold neutral on the random stance
    elif t < 6.0:
        cmd.pitch = PITCH_AMP * np.sin(w * (t - 3.0))
        cmd.pitch_rate = PITCH_AMP * w * np.cos(w * (t - 3.0))
    elif t < 9.0:
        cmd.roll = ROLL_AMP * np.sin(w * (t - 6.0))
        cmd.roll_rate = ROLL_AMP * w * np.cos(w * (t - 6.0))
    elif t < 12.0:
        cmd.z_pos = Z_NOMINAL + Z_AMP * np.sin(w * (t - 9.0))
        cmd.z_vel = Z_AMP * w * np.cos(w * (t - 9.0))
    else:
        w_xy = 2.0 * np.pi * XY_FREQ_HZ
        cmd.y_off = Y_AMP * np.sin(w_xy * (t - 12.0))
        cmd.y_vel = Y_AMP * w_xy * np.cos(w_xy * (t - 12.0))

    return cmd

# --------------------------------------------------------------------------------
# Storage Variables (CONTROL-rate logs for plots)
# --------------------------------------------------------------------------------

# Centroidal state x = [px, py, pz, r, p, y, vx, vy, vz, wx, wy, wz]
x_vec = np.zeros((12, CTRL_STEPS))

# Desired posture log [roll, pitch, z, y]
posture_des_log = np.zeros((4, CTRL_STEPS))

# Foot positions in world (slip check): [FLx,FLy,FLz, ...]
foot_pos_log = np.zeros((12, CTRL_STEPS))

# MPC contact force log (world)
mpc_force_world = np.zeros((12, CTRL_STEPS))

# Torques
tau_raw = np.zeros((12, CTRL_STEPS))
tau_cmd = np.zeros((12, CTRL_STEPS))

# Control-rate log
time_log_ctrl_s = np.zeros(CTRL_STEPS)
q_log_ctrl = np.zeros((CTRL_STEPS, 19))
tau_log_ctrl_Nm = np.zeros((CTRL_STEPS, 12))

mpc_update_time_ms = []
mpc_solve_time_ms = []
X_opt = None
U_opt = None

# --------------------------------------------------------------------------------
# Simulation Initialization
# --------------------------------------------------------------------------------

go2 = PinA2Model()
mujoco_go2 = MuJoCo_A2_Model()
leg_controller = LegController()
traj = ComTraj(go2)
gait = StandGait(GAIT_HZ)

# Initialize robot configuration with the randomized stance
q_init = go2.current_config.get_q()
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
q_init[7:19] = q_joints_init
go2.update_model(q_init, go2.dq_init)
mujoco_go2.update_with_q_pin(q_init)

# Set physics dt (keep it fast!)
mujoco_go2.model.opt.timestep = SIM_DT

# Initialize MPC
traj.generate_traj(
    go2,
    gait,
    0.0,
    0.0,
    0.0,
    Z_NOMINAL,
    0.0,
    time_step=MPC_DT,
)
mpc = CentroidalMPC(go2, traj, Q=POSTURE_COST_Q)

# Safe defaults until first solve
U_opt = np.zeros((12, traj.N), dtype=float)

# --------------------------------------------------------------------------------
# Replay logs sampled at RENDER_HZ
# --------------------------------------------------------------------------------
time_log_render = []
q_log_render = []
tau_log_render = []

next_render_t = 0.0

# --------------------------------------------------------------------------------
# Simulation Loop
# --------------------------------------------------------------------------------
print(f"Running simulation for {RUN_SIM_LENGTH_S}s")
sim_start_time = time.perf_counter()

ctrl_i = 0
tau_hold = np.zeros(12, dtype=float)

# Stance-center reference for y sway (captured at the first control tick)
x_ref0, y_ref0 = None, None

for k in range(SIM_STEPS):
    time_now_s = float(mujoco_go2.data.time)

    # Control update at CTRL_HZ
    if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
        # Posture commands (updated at control rate)
        cmd = get_posture_cmd(time_now_s)

        # Update Pinocchio from current MuJoCo state
        mujoco_go2.update_pin_with_mujoco(go2)

        # Capture the neutral stance COM position once, as the sway reference
        if x_ref0 is None:
            x_ref0 = float(go2.pos_com_world[0])
            y_ref0 = float(go2.pos_com_world[1])

        x_pos_des_world = x_ref0
        y_pos_des_world = y_ref0 + cmd.y_off

        x_vec[:, ctrl_i] = go2.compute_com_x_vec().reshape(-1)
        posture_des_log[:, ctrl_i] = [cmd.roll, cmd.pitch, cmd.z_pos, y_pos_des_world]
        foot_pos_log[:, ctrl_i] = np.concatenate(go2.get_foot_placement_in_world())

        # Control-rate logs
        time_log_ctrl_s[ctrl_i] = time_now_s
        q_log_ctrl[ctrl_i, :] = mujoco_go2.data.qpos

        # Update MPC if needed
        if (ctrl_i % STEPS_PER_MPC) == 0:
            print(f"\rSimulation Time: {time_now_s:.3f} s", end="", flush=True)

            traj.generate_traj(
                go2,
                gait,
                time_now_s,
                0.0,
                cmd.y_vel,
                cmd.z_pos,
                0.0,
                time_step=MPC_DT,
                roll_des_body=cmd.roll,
                pitch_des_body=cmd.pitch,
                roll_rate_des_body=cmd.roll_rate,
                pitch_rate_des_body=cmd.pitch_rate,
                x_pos_des_world=x_pos_des_world,
                y_pos_des_world=y_pos_des_world,
                z_vel_des_body=cmd.z_vel,
            )

            sol = mpc.solve_QP(go2, traj, False)
            mpc_solve_time_ms.append(mpc.solve_time)
            mpc_update_time_ms.append(mpc.update_time)

            N = traj.N
            w_opt = sol["x"].full().flatten()
            X_opt = w_opt[: 12 * (N)].reshape((12, N), order="F")
            U_opt = w_opt[12 * (N) :].reshape((12, N), order="F")

        # Extract first GRF from MPC
        mpc_force_world[:, ctrl_i] = U_opt[:, 0]

        # Compute joint torques (all legs are in stance: tau = J^T * -f)
        for leg in LEGS:
            out = leg_controller.compute_leg_torque(
                leg, go2, gait, mpc_force_world[LEG_SLICE[leg], ctrl_i], time_now_s
            )
            tau_raw[LEG_SLICE[leg], ctrl_i] = out.tau

        # Saturate + hold
        tau_cmd[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -TAU_LIM, TAU_LIM)
        tau_hold = tau_cmd[:, ctrl_i].copy()

        tau_log_ctrl_Nm[ctrl_i, :] = tau_hold

        ctrl_i += 1

    #Apply held torques at every SIM step
    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
    mujoco_go2.set_joint_torque(tau_hold)
    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    #Render-rate logging for smooth replay
    t_after = float(mujoco_go2.data.time)
    if t_after + 1e-12 >= next_render_t:
        time_log_render.append(t_after)
        q_log_render.append(mujoco_go2.data.qpos.copy())
        tau_log_render.append(tau_hold.copy())
        next_render_t += RENDER_DT

sim_end_time = time.perf_counter()
print(
    f"\nSimulation ended."
    f"\nElapsed time: {sim_end_time - sim_start_time:.3f}s"
    f"\nControl ticks: {ctrl_i}/{CTRL_STEPS}"
)

# Foot slip over the whole run (feet should stay at their offset positions)
foot_slip_mm = np.zeros(4)
for i, leg in enumerate(LEGS):
    xy = foot_pos_log[3*i:3*i+2, :ctrl_i]
    foot_slip_mm[i] = 1e3 * np.linalg.norm(xy[:, -1] - xy[:, 0])
    print(f"{leg} foot xy slip over the run: {foot_slip_mm[i]:.1f} mm")

# --------------------------------------------------------------------------------
# Simulation Results
# --------------------------------------------------------------------------------

t_vec = np.arange(ctrl_i) * CTRL_DT

import matplotlib.pyplot as plt

# Top view of the stance: nominal vs randomized target vs simulated feet
fig, ax = plt.subplots(figsize=(6, 6))
base_xy = np.array([INITIAL_X_POS, INITIAL_Y_POS])
for i, leg in enumerate(LEGS):
    nom = base_xy + p_foot_nominal[leg][0:2]
    tgt = base_xy + p_foot_target[leg][0:2]
    meas = foot_pos_log[3*i:3*i+2, :ctrl_i]
    ax.plot(*nom, "ko", fillstyle="none", markersize=10,
            label="nominal" if i == 0 else None)
    ax.plot(*tgt, "rx", markersize=10, label="target (+offset)" if i == 0 else None)
    ax.plot(meas[0], meas[1], "b.", markersize=2,
            label="simulated" if i == 0 else None)
    ax.annotate(leg, nom, textcoords="offset points", xytext=(8, 8))
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("A2 foot placements (top view), offsets uniform +/- 30 mm")
ax.axis("equal")
ax.grid(True)
ax.legend(loc="upper right")
fig.tight_layout()
plt.show(block=False)

# Posture tracking summary (desired vs measured)
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
labels = [("Roll [rad]", 3, 0), ("Pitch [rad]", 4, 1), ("Height z [m]", 2, 2),
          ("Y pos [m]", 1, 3)]
for ax, (ylabel, x_idx, des_idx) in zip(axes, labels):
    ax.plot(t_vec, x_vec[x_idx, :ctrl_i], label="measured")
    ax.plot(t_vec, posture_des_log[des_idx, :ctrl_i], "--", label="desired")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Time [s]")
fig.suptitle("A2 Posture Tracking on a Randomized Stance")
fig.tight_layout()
plt.show(block=False)

plot_mpc_result(t_vec, mpc_force_world, tau_cmd, x_vec, block=False)
plot_solve_time(mpc_solve_time_ms, mpc_update_time_ms, MPC_DT, MPC_HZ, block=True)

# Replay simulation
time_log_render = np.asarray(time_log_render, dtype=float)
q_log_render = np.asarray(q_log_render, dtype=float)
tau_log_render = np.asarray(tau_log_render, dtype=float)

mujoco_go2.replay_simulation(time_log_render, q_log_render, tau_log_render, RENDER_DT, REALTIME_FACTOR)
hold_until_all_fig_closed()
