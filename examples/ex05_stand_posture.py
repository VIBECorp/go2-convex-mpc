"""
Demo 05: Standing body posture control (all four feet in stance)

The robot keeps all four feet on the ground (StandGait) while the MPC tracks
time-varying body posture commands:
    0-3 s   : pitch nod        (+/- 0.2 rad)
    3-6 s   : roll sway        (+/- 0.2 rad)
    6-9 s   : yaw oscillation  (+/- 0.3 rad, commanded via yaw rate)
    9-12 s  : height squat     (0.22 m - 0.32 m)
    12-16 s : x body sway      (+/- 50 mm, 0.25 Hz)
    16-20 s : y body sway      (+/- 50 mm, 0.25 Hz)
    20-23 s : combined roll + pitch circling
"""
import os
os.environ["MPLBACKEND"] = "TkAgg"
import time
import mujoco as mj
import numpy as np
from dataclasses import dataclass, field

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
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
RUN_SIM_LENGTH_S = 23.0

RENDER_HZ = 120.0
RENDER_DT = 1.0 / RENDER_HZ
REALTIME_FACTOR = 1

# Nominal standing height
Z_NOMINAL = 0.27

# Posture command amplitudes
PITCH_AMP = 0.2     # rad
ROLL_AMP = 0.2      # rad
YAW_AMP = 0.3       # rad
Z_AMP = 0.05        # m
XY_AMP = 0.05       # m (+/- 50 mm body sway)
CIRCLE_AMP = 0.15   # rad
CMD_FREQ_HZ = 0.5   # frequency of the attitude/height oscillations
XY_FREQ_HZ = 0.25   # slower x/y sway: the MPC horizon (~0.33 s) extrapolates the
                    # position reference linearly, which clips fast sway peaks

# Gait Setting (frequency only sets the MPC horizon; no leg ever swings)
GAIT_HZ = 3
GAIT_T = 1.0 / GAIT_HZ

# MPC cost: emphasize position and orientation tracking for posture control.
# x/y velocity weights damp the lateral sway response (too little -> oscillation)
POSTURE_COST_Q = np.diag([200, 200, 200,  250, 250, 100,  40, 40, 1,  1, 1, 1])

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

# Go2 Joint Torque Limit
HIP_LIM = 23.7
ABD_LIM = 23.7
KNEE_LIM = 45.43
SAFETY = 0.9

TAU_LIM = SAFETY * np.array([
    HIP_LIM, ABD_LIM, KNEE_LIM,   # FL: hip, thigh, calf
    HIP_LIM, ABD_LIM, KNEE_LIM,   # FR
    HIP_LIM, ABD_LIM, KNEE_LIM,   # RL
    HIP_LIM, ABD_LIM, KNEE_LIM,   # RR
])

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

# --------------------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------------------
@dataclass
class PostureCmd:
    roll: float = 0.0        # rad
    pitch: float = 0.0       # rad
    z_pos: float = Z_NOMINAL # m
    yaw_rate: float = 0.0    # rad/s
    roll_rate: float = 0.0   # rad/s (feedforward)
    pitch_rate: float = 0.0  # rad/s (feedforward)
    x_off: float = 0.0       # m, body x offset from initial stance position
    y_off: float = 0.0       # m, body y offset from initial stance position
    x_vel: float = 0.0       # m/s (feedforward)
    y_vel: float = 0.0       # m/s (feedforward)
    z_vel: float = 0.0       # m/s (feedforward)


def get_posture_cmd(t: float) -> PostureCmd:
    """
    Posture command at time t. Rates/velocities are the analytic derivatives,
    used as feedforward. Each segment is a whole number of half-periods so it
    ends back at neutral.
    """
    w = 2.0 * np.pi * CMD_FREQ_HZ
    cmd = PostureCmd()

    if t < 3.0:
        cmd.pitch = PITCH_AMP * np.sin(w * t)
        cmd.pitch_rate = PITCH_AMP * w * np.cos(w * t)
    elif t < 6.0:
        cmd.roll = ROLL_AMP * np.sin(w * (t - 3.0))
        cmd.roll_rate = ROLL_AMP * w * np.cos(w * (t - 3.0))
    elif t < 9.0:
        # yaw(t) = YAW_AMP * sin(w t)  ->  command its rate
        cmd.yaw_rate = YAW_AMP * w * np.cos(w * (t - 6.0))
    elif t < 12.0:
        cmd.z_pos = Z_NOMINAL + Z_AMP * np.sin(w * (t - 9.0))
        cmd.z_vel = Z_AMP * w * np.cos(w * (t - 9.0))
    elif t < 16.0:
        w_xy = 2.0 * np.pi * XY_FREQ_HZ
        cmd.x_off = XY_AMP * np.sin(w_xy * (t - 12.0))
        cmd.x_vel = XY_AMP * w_xy * np.cos(w_xy * (t - 12.0))
    elif t < 20.0:
        w_xy = 2.0 * np.pi * XY_FREQ_HZ
        cmd.y_off = XY_AMP * np.sin(w_xy * (t - 16.0))
        cmd.y_vel = XY_AMP * w_xy * np.cos(w_xy * (t - 16.0))
    else:
        # ramp the amplitude in over 1 s to avoid a step in pitch
        s = t - 20.0
        amp = CIRCLE_AMP * min(1.0, s)
        damp = CIRCLE_AMP if s < 1.0 else 0.0
        cmd.roll = amp * np.sin(w * s)
        cmd.pitch = amp * np.cos(w * s)
        cmd.roll_rate = damp * np.sin(w * s) + amp * w * np.cos(w * s)
        cmd.pitch_rate = damp * np.cos(w * s) - amp * w * np.sin(w * s)

    return cmd

# --------------------------------------------------------------------------------
# Storage Variables (CONTROL-rate logs for plots)
# --------------------------------------------------------------------------------

# Centroidal state x = [px, py, pz, r, p, y, vx, vy, vz, wx, wy, wz]
x_vec = np.zeros((12, CTRL_STEPS))

# Desired posture log [roll, pitch, z, x, y]
posture_des_log = np.zeros((5, CTRL_STEPS))

# MPC contact force log (world): [FLx,FLy,FLz, FRx,FRy,FRz, RLx,RLy,RLz, RRx,RRy,RRz]
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

go2 = PinGo2Model()
mujoco_go2 = MuJoCo_GO2_Model()
leg_controller = LegController()
traj = ComTraj(go2)
gait = StandGait(GAIT_HZ)

# Initialize robot configuration
q_init = go2.current_config.get_q()
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
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

# Stance-center reference for x/y sway (captured at the first control tick)
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

        x_pos_des_world = x_ref0 + cmd.x_off
        y_pos_des_world = y_ref0 + cmd.y_off

        x_vec[:, ctrl_i] = go2.compute_com_x_vec().reshape(-1)
        posture_des_log[:, ctrl_i] = [cmd.roll, cmd.pitch, cmd.z_pos,
                                      x_pos_des_world, y_pos_des_world]

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
                cmd.x_vel,
                cmd.y_vel,
                cmd.z_pos,
                cmd.yaw_rate,
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
        for leg in ("FL", "FR", "RL", "RR"):
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

# --------------------------------------------------------------------------------
# Simulation Results
# --------------------------------------------------------------------------------

# Posture tracking summary (desired vs measured)
t_vec = np.arange(ctrl_i) * CTRL_DT

import matplotlib.pyplot as plt
fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 11))
labels = [("Roll [rad]", 3, 0), ("Pitch [rad]", 4, 1), ("Height z [m]", 2, 2),
          ("X pos [m]", 0, 3), ("Y pos [m]", 1, 4)]
for ax, (ylabel, x_idx, des_idx) in zip(axes, labels):
    ax.plot(t_vec, x_vec[x_idx, :ctrl_i], label="measured")
    ax.plot(t_vec, posture_des_log[des_idx, :ctrl_i], "--", label="desired")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Time [s]")
fig.suptitle("Standing Body Posture Tracking")
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
