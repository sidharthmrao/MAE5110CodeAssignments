"""Controlled walker simulation and GIF generation."""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

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
    initial_state, params=None, *, timestep=1e-4, sim_time=5.0, max_collisions=1
):
    """Return state and applied-torque histories; copy parameters for each trial.

    By default stop at the first collision or after five seconds.
    """
    params = dict(model.generate_params() if params is None else params)
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (2,) or not np.all(np.isfinite(initial_state)):
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
    if not lower <= initial_state[0] <= upper:
        raise ValueError("Initial angle must lie within the collision bounds")
    n_timesteps = round(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    torque_traj = np.full(n_timesteps, model.clipped_ankle_torque(params))
    completed_steps = 0
    last_step = 0
    dynamics = partial(model.dynamics, params=params)

    for step, t in enumerate(time_traj[:-1]):
        params["ankle_torque"] = control(t, state_traj[:, step], params)
        torque_traj[step] = model.clipped_ankle_torque(params)
        torque_traj[step + 1] = torque_traj[
            step
        ]  # Hold the last input at the endpoint.

        # First, calculate the regular update (no collision).
        state_traj[:, step + 1] = integrator(state_traj[:, step], t, timestep, dynamics)

        # Check for collision and wrap to the new stance leg.
        if model.forward_collision_guard(state_traj[:, step + 1], params):
            state_traj[:, step + 1] = model.forward_collision_dynamics(
                state_traj[:, step + 1], params
            )
            completed_steps += 1
        elif model.backward_collision_guard(state_traj[:, step + 1], params):
            state_traj[:, step + 1] = model.backward_collision_dynamics(
                state_traj[:, step + 1], params
            )
            completed_steps += 1

        last_step = step + 1
        if max_collisions is not None and completed_steps >= max_collisions:
            break

    time_traj = time_traj[: last_step + 1]
    state_traj = state_traj[:, : last_step + 1]
    torque_traj = torque_traj[: last_step + 1]

    return {
        "time": time_traj,
        "state": state_traj,
        "torque": torque_traj,
        "completed_steps": completed_steps,
        "params": params,
        "timestep": timestep,
    }


def save_animation(result, output_dir=OUTPUT_DIR):
    """Save the walker, state histories, and applied torque without opening a window."""
    time_traj = result["time"]
    state_traj = result["state"]
    torque_traj = result["torque"]
    params = result["params"]
    completed_steps = result["completed_steps"]
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    ax = fig.add_subplot(grid[:, 0])
    angle_ax = fig.add_subplot(grid[0, 1])
    velocity_ax = fig.add_subplot(grid[1, 1], sharex=angle_ax)
    torque_ax = fig.add_subplot(grid[2, 1], sharex=angle_ax)

    angle_ax.plot(time_traj, state_traj[0], color="tab:blue")
    velocity_ax.plot(time_traj, state_traj[1], color="tab:orange")
    angle_ax.set_ylabel("Angle [rad]")
    velocity_ax.set_ylabel("Angular velocity [rad/s]")
    torque_ax.step(time_traj, torque_traj, where="post", color="tab:green")
    for limit in (params["ankle_torque_min"], params["ankle_torque_max"]):
        torque_ax.axhline(limit, color="tab:red", linestyle=":", alpha=0.6)
    torque_ax.set_ylabel("Ankle torque [N m]")
    torque_ax.set_xlabel("Time [s]")
    velocity_ax.tick_params(labelbottom=False)
    angle_ax.set_title("State and applied torque over time")
    angle_ax.tick_params(labelbottom=False)
    time_markers = []
    state_markers = []
    plot_trajs = (state_traj[0], state_traj[1], torque_traj)
    for values, state_ax in zip(plot_trajs, (angle_ax, velocity_ax, torque_ax)):
        state_ax.grid(alpha=0.3)
        state_ax.set_xlim(0.0, max(time_traj[-1], 1e-4))
        time_markers.append(
            state_ax.axvline(0.0, color="black", linestyle="--", alpha=0.6)
        )
        (marker,) = state_ax.plot([0.0], [values[0]], "o", color="black")
        state_markers.append(marker)

    def draw_frame(index):
        # The massless swing leg is repositioned instantaneously at each impact.
        frame_params = dict(params, ankle_torque=torque_traj[index])
        model.visualize(state_traj[:, index], frame_params, ax=ax)
        ax.set_title(f"t = {time_traj[index]:.2f} s")
        for state_index, (time_marker, state_marker) in enumerate(
            zip(time_markers, state_markers)
        ):
            time_marker.set_xdata([time_traj[index], time_traj[index]])
            state_marker.set_data([time_traj[index]], [plot_trajs[state_index][index]])

    # Simulate at a small timestep, but render only 25 frames per second.
    fps = 25
    frame_stride = max(1, round(1 / (fps * result["timestep"])))
    frame_indices = list(range(0, time_traj.size, frame_stride))
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(time_traj.size - 1)

    animation = FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

    # To save an MP4 instead, install FFmpeg and use:
    # animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
    print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
    plt.close(fig)


if __name__ == "__main__":
    result = simulate([-0.22, 0.55], timestep=1e-4, sim_time=5.0, max_collisions=1)
    save_animation(result)
