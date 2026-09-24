"""Shared standing controller and simulation."""

from functools import partial
from pathlib import Path

import numpy as np

from assignment_2.assignment_2_tables import in_standing_roa, load_standing_roa
from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def control(t, state, params):
    m = params["mass"]
    g = params["gravity"]
    l = params["spoke_length"]

    angle, angular_velocity = state

    T = -2.5 * angle - 4.0 * angular_velocity - 2 * m * g * l * np.sin(angle)

    return np.clip(T, params["ankle_torque_min"], params["ankle_torque_max"])


def simulate(
    initial_state,
    params=None,
    *,
    timestep=1e-4,
    sim_time=5.0,
    max_collisions=1,
    controller=None,
    roa_file=OUTPUT_DIR / "roa" / "roa_results.npz",
    roa=None,
    section_controller=None,
    stop_at_balance=False,
    poincare_return=False,
):
    """Run the shared physics and RoA-triggered ankle controller.

    Normally return trajectory histories, stopping at the collision/time limit
    or, with stop_at_balance, when balancing starts. section_controller chooses
    alpha at forward upright crossings while walking.

    poincare_return instead returns only the upright velocity after one forward
    impact, or NaN on backward impact, another forward impact, or timeout. This
    mode skips trajectory storage. The initial upright state is not a return.

    An explicit controller overrides RoA gating. RoA construction uses control
    directly to measure the basin; validation uses it to continue balancing.
    """
    params = dict(model.generate_params() if params is None else params)
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain two finite values")
    if (
        not np.isfinite(timestep)
        or timestep <= 0
        or not np.isfinite(sim_time)
        or sim_time < 0
    ):
        raise ValueError("timestep must be positive and sim_time nonnegative")
    if max_collisions is not None and max_collisions < 1:
        raise ValueError("max_collisions must be positive or None")
    lower, upper = model.angle_collision_bounds(params)
    if not lower <= state[0] <= upper:
        raise ValueError("Initial angle must lie within the collision bounds")

    if controller is None:
        roa = load_standing_roa(roa_file) if roa is None else roa
        params["ankle_torque"] = 0.0
    balance_start_time = None

    def check_roa(t, current_state):
        nonlocal balance_start_time
        if (
            controller is None
            and balance_start_time is None
            and in_standing_roa(current_state, roa)
        ):
            balance_start_time = t
            params["alpha"] = roa["params"]["alpha"]

    check_roa(0.0, state)
    if (
        section_controller is not None
        and balance_start_time is None
        and state[0] == 0
        and state[1] >= 0
    ):
        section_controller(0.0, state.copy(), params)

    n_steps = int(np.ceil(sim_time / timestep))
    if not poincare_return:
        time_traj = np.minimum(np.arange(n_steps + 1) * timestep, sim_time)
        state_traj = np.zeros((2, n_steps + 1))
        state_traj[:, 0] = state
        torque_traj = np.full(n_steps + 1, model.clipped_ankle_torque(params))
        alpha_traj = np.full(n_steps + 1, params["alpha"])
    completed_steps = 0
    forward_steps = 0
    backward_steps = 0
    last_step = 0
    dynamics = partial(model.dynamics, params=params)

    for step in range(n_steps):
        t = step * timestep
        dt = min(timestep, sim_time - t)
        check_roa(t, state)
        if controller is not None:
            params["ankle_torque"] = controller(t, state, params)
        else:
            params["ankle_torque"] = (
                control(t, state, params) if balance_start_time is not None else 0.0
            )
        if not poincare_return:
            torque_traj[step] = torque_traj[step + 1] = model.clipped_ankle_torque(
                params
            )
            alpha_traj[step] = params["alpha"]
        if stop_at_balance and balance_start_time is not None:
            break

        next_state = integrator(state, t, dt, dynamics)
        if poincare_return and not np.all(np.isfinite(next_state)):
            return np.nan

        # Detect smooth upright crossings before applying collision resets.
        if state[0] < 0 <= next_state[0]:
            fraction = -state[0] / (next_state[0] - state[0])
            velocity = state[1] + fraction * (next_state[1] - state[1])
            if velocity > 0:
                if poincare_return and completed_steps == 1:
                    return float(velocity)
                if section_controller is not None:
                    crossing_time = t + fraction * dt
                    crossing_state = np.array([0.0, velocity])
                    check_roa(crossing_time, crossing_state)
                    if balance_start_time is None:
                        section_controller(crossing_time, crossing_state, params)

        if model.forward_collision_guard(next_state, params):
            next_state = model.forward_collision_dynamics(next_state, params)
            completed_steps += 1
            forward_steps += 1
            if poincare_return and completed_steps > 1:
                return np.nan
        elif model.backward_collision_guard(next_state, params):
            if poincare_return:
                return np.nan
            next_state = model.backward_collision_dynamics(next_state, params)
            completed_steps += 1
            backward_steps += 1

        state = next_state
        last_step = step + 1
        if not poincare_return:
            state_traj[:, last_step] = state
            alpha_traj[last_step] = params["alpha"]
            if max_collisions is not None and completed_steps >= max_collisions:
                break

    if poincare_return:
        return np.nan

    return {
        "time": time_traj[: last_step + 1],
        "state": state_traj[:, : last_step + 1],
        "torque": torque_traj[: last_step + 1],
        "alpha": alpha_traj[: last_step + 1],
        "completed_steps": completed_steps,
        "forward_steps": forward_steps,
        "backward_steps": backward_steps,
        "params": params,
        "timestep": timestep,
        "balance_start_time": balance_start_time,
    }
