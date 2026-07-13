import numpy as np
from .go2_robot_data import PinGo2Model
from .gait import Gait
from dataclasses import dataclass

# --------------------------------------------------------------------------------
# Leg Controller Setting
# --------------------------------------------------------------------------------

KP_SWING = np.diag([400, 400, 400])
KD_SWING = np.diag([75, 75, 75])

# Joint dry-friction feedforward (per-joint values come from the robot model,
# matching frictionloss in the MJCF). Without it, the friction deadband absorbs
# small stance-force corrections and the body stalls short of slow position
# targets (e.g. standing sway).
FRICTION_VEL_EPS = 0.02     # rad/s, tanh smoothing to avoid chatter at rest

# Mapping from leg name to index in the mask
LEG_INDEX = {
    "FL": 0,
    "FR": 1,
    "RL": 2,
    "RR": 3,
}

@dataclass
class LegOutput:
    tau: np.ndarray       # shape (3,)
    pos_des: np.ndarray   # shape (3,)
    pos_now: np.ndarray   # shape (3,)
    vel_des: np.ndarray   # shape (3,)
    vel_now: np.ndarray   # shape (3,)


class LegController():
        
    def __init__(self):
            self.last_mask = np.array([2, 2, 2, 2])

    def compute_leg_torque(
        self,
        leg: str,
        go2: PinGo2Model,
        gait: Gait,
        contact_force: np.ndarray,
        current_time: float,
    ):
        # Extract Parameters
        leg_idx = LEG_INDEX[leg]
        joint_slice = go2.get_leg_v_indices(leg)

        J_foot_world = go2.compute_3x3_foot_Jacobian_world(leg)      # (3x3)
        J_full_foot_world = go2.compute_full_foot_Jacobian_world(leg)  # (3x18)
        g, C, M = go2.compute_dynamcis_terms()

        current_mask = gait.compute_current_mask(current_time)
        tau_cmd = np.zeros((3, 1))

        # Initialize desired to current
        foot_pos_des, foot_vel_des = go2.get_single_foot_state_in_world(leg)
        foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

        # Detect takeoff transition
        if self.last_mask[leg_idx] != current_mask[leg_idx] and current_mask[leg_idx] == 0:
            # This leg just took off
            setattr(self, f"{leg}_takeoff_time", current_time)
            traj, td_pos = gait.compute_swing_traj_and_touchdown(go2, leg)
            setattr(self, f"{leg}_traj", traj)
            setattr(self, f"{leg}_td_pos", td_pos)

        # Swing vs stance
        if current_mask[leg_idx] == 0:  # Swing phase
            takeoff_time = getattr(self, f"{leg}_takeoff_time")
            traj = getattr(self, f"{leg}_traj")

            time_since_takeoff = current_time - takeoff_time
            foot_pos_des, foot_vel_des, foot_acl_des = traj(time_since_takeoff)
            foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

            pos_error = foot_pos_des - foot_pos_now
            vel_error = foot_vel_des - foot_vel_now

            Lambda = np.linalg.inv(
                J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T
            )  # (3x3)
            Jdot_dq = go2.compute_Jdot_dq_world(leg)

            # Feedforward term (3x1)
            f_ff = Lambda @ (foot_acl_des - Jdot_dq)

            # PD + feedforward in Cartesian space
            force = KP_SWING @ pos_error + KD_SWING @ vel_error + f_ff  # (3x1)

            # Map to joint torques + add (C*dq + g) leg segment
            tau_cmd = J_foot_world.T @ force + (C @ go2.current_config.get_dq() + g)[joint_slice]

        else:  # Stance phase
            # Compensate the leg's own gravity/Coriolis torques so the commanded
            # contact force is actually realized at the foot (without this, the
            # realized force is biased by the leg-link weight)
            tau_cmd = J_foot_world.T @ -contact_force + (C @ go2.current_config.get_dq() + g)[joint_slice]

        # Dry-friction feedforward along the current joint motion direction
        joint_vel = np.asarray(getattr(go2.current_config, f"{leg}_joint_vel")).reshape(3,)
        tau_friction = np.asarray(go2.TAU_FRICTION_COMP, dtype=float)
        tau_cmd = tau_cmd.reshape(3,) + tau_friction * np.tanh(joint_vel / FRICTION_VEL_EPS)

        # Update mask memory
        self.last_mask[leg_idx] = current_mask[leg_idx]

        return LegOutput(
            tau=tau_cmd.reshape(3,),
            pos_des=foot_pos_des,
            pos_now=foot_pos_now,
            vel_des=foot_vel_des,
            vel_now=foot_vel_now,
        )