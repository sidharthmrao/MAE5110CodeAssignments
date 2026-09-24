"""Table-based walker experiment and GIF generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from assignment_2.assignment_2_simulation import OUTPUT_DIR
from assignment_2.assignment_2_simulation import simulate as simulate_walker
from models import inverted_pendulum_walker as model

ROA_FILE = OUTPUT_DIR / "roa" / "roa_results.npz"
POLICY_FILE = OUTPUT_DIR / "poincare" / "step_policy.npz"
INITIAL_STATE = [0.0, 3.0]
TIMESTEP = 1e-4
SIM_TIME = 10.0
ANIMATION_FPS = 25


def simulate_policy(
    initial_state,
    params=None,
    *,
    timestep=1e-4,
    sim_time=10.0,
    max_collisions=None,
    early_convergence=False,
    roa_file=ROA_FILE,
    policy_file=POLICY_FILE,
):
    """Choose alpha during passive walking; enable ankle control upon entering the RoA.

    Non-upright starts use the supplied alpha until the first forward upright
    crossing. Check the full standing RoA at every timestep and upright crossing
    to switch to balancing as soon as it is entered. Missing policy entries
    raise an error instead of silently using an arbitrary foot placement.
    """
    params = dict(model.generate_params() if params is None else params)
    with np.load(policy_file) as data:
        policy_velocities = data["velocities"]
        min_steps = data["min_steps"]
        best_alpha = data["best_alpha"]

    def choose_alpha(t, state, model_params):
        velocity = float(state[1])
        if not policy_velocities[0] <= velocity <= policy_velocities[-1]:
            raise ValueError(
                f"Upright velocity {velocity:.6g} is outside the policy table"
            )
        index = int(np.argmin(abs(policy_velocities - velocity)))
        alpha = best_alpha[index]
        # A rounded zero-step entry cannot override the actual state RoA check.
        if min_steps[index] <= 0 or not np.isfinite(alpha):
            raise ValueError(
                f"No alpha policy found for upright velocity {velocity:.6g}"
            )
        else:
            if not model_params["alpha_min"] <= alpha <= model_params["alpha_max"]:
                raise ValueError("Saved alpha is outside the model's limits")
            model_params["alpha"] = float(alpha)

    return simulate_walker(
        initial_state,
        params,
        timestep=timestep,
        sim_time=sim_time,
        max_collisions=max_collisions,
        roa_file=roa_file,
        section_controller=choose_alpha,
        stop_at_balance=early_convergence,
    )


def save_animation(result, output_dir=OUTPUT_DIR, name="walker"):
    """Save the walker, state histories, and applied torque without opening a window."""
    time_traj = result["time"]
    state_traj = result["state"]
    torque_traj = result["torque"]
    params = result["params"]
    completed_steps = result["completed_steps"]
    alpha_degrees = np.rad2deg(result["alpha"])
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(4, 2)
    ax = fig.add_subplot(grid[:2, 0])
    roa_ax = fig.add_subplot(grid[2:, 0])
    angle_ax = fig.add_subplot(grid[0, 1])
    velocity_ax = fig.add_subplot(grid[1, 1], sharex=angle_ax)
    torque_ax = fig.add_subplot(grid[2, 1], sharex=angle_ax)
    alpha_ax = fig.add_subplot(grid[3, 1], sharex=angle_ax)

    angle_ax.plot(time_traj, state_traj[0], color="tab:blue")
    velocity_ax.plot(time_traj, state_traj[1], color="tab:orange")
    angle_ax.set_ylabel("Angle [rad]")
    velocity_ax.set_ylabel("Angular velocity [rad/s]")
    torque_ax.step(time_traj, torque_traj, where="post", color="tab:green")
    for limit in (params["ankle_torque_min"], params["ankle_torque_max"]):
        torque_ax.axhline(limit, color="tab:red", linestyle=":", alpha=0.6)
    torque_ax.set_ylabel("Ankle torque [N m]")
    alpha_ax.step(time_traj, alpha_degrees, where="post", color="tab:purple")
    for limit in (params["alpha_min"], params["alpha_max"]):
        alpha_ax.axhline(np.rad2deg(limit), color="tab:red", linestyle=":", alpha=0.6)
    alpha_ax.set_ylabel("α [deg]")
    alpha_ax.set_xlabel("Time [s]")
    torque_ax.tick_params(labelbottom=False)
    velocity_ax.tick_params(labelbottom=False)
    angle_ax.set_title("State and control inputs over time")
    angle_ax.tick_params(labelbottom=False)
    # The saved RoA describes ankle-only balancing at its original fixed alpha.
    with np.load(ROA_FILE) as data:
        roa_angles = data["angles"]
        roa_velocities = data["velocities"]
        roa_converged = data["converged"]
    roa_ax.pcolormesh(
        roa_angles,
        roa_velocities,
        roa_converged.T,
        shading="nearest",
        cmap=ListedColormap(["#dedede", "#8fd19e"]),
        vmin=0,
        vmax=1,
    )
    # Break the drawn path at impact resets rather than implying continuous motion.
    path = state_traj.copy()
    impact_indices = (
        np.flatnonzero(abs(np.diff(state_traj[0])) > params["alpha_min"]) + 1
    )
    path[:, impact_indices] = np.nan
    (roa_path,) = roa_ax.plot([], [], color="tab:blue", linewidth=1, label="Trajectory")
    roa_ax.plot(
        *state_traj[:, 0], "*", color="tab:orange", markersize=12, label="Start"
    )
    (roa_point,) = roa_ax.plot(
        [], [], "o", color="black", markersize=5, label="Current state"
    )
    balance_time = result.get("balance_start_time")
    if balance_time is not None:
        balance_index = min(
            np.searchsorted(time_traj, balance_time), len(time_traj) - 1
        )
        roa_ax.plot(
            *state_traj[:, balance_index],
            "D",
            color="tab:purple",
            markersize=5,
            label="Switch to Balancing",
        )
    x_min = min(roa_angles[0], state_traj[0].min())
    x_max = max(roa_angles[-1], state_traj[0].max())
    y_min = min(roa_velocities[0], state_traj[1].min())
    y_max = max(roa_velocities[-1], state_traj[1].max())
    roa_ax.set(
        xlim=(x_min - 0.03, x_max + 0.03),
        ylim=(y_min - 0.15, y_max + 0.15),
        xlabel="Angle [rad]",
        ylabel="Angular Velocity [rad/s]",
        title="Standing RoA and Current State",
    )
    handles, _ = roa_ax.get_legend_handles_labels()
    handles += [
        Patch(color="#8fd19e", label="Standing RoA"),
        Patch(color="#dedede", label="Collision in RoA test"),
        Patch(facecolor="white", edgecolor="0.7", label="Not sampled"),
    ]
    roa_ax.legend(handles=handles, fontsize=7, loc="lower left", ncol=2)

    time_markers = []
    state_markers = []
    plot_trajs = (state_traj[0], state_traj[1], torque_traj, alpha_degrees)
    for values, state_ax in zip(
        plot_trajs, (angle_ax, velocity_ax, torque_ax, alpha_ax)
    ):
        state_ax.grid(alpha=0.3)
        state_ax.set_xlim(0.0, max(time_traj[-1], 1e-4))
        time_markers.append(
            state_ax.axvline(0.0, color="black", linestyle="--", alpha=0.6)
        )
        (marker,) = state_ax.plot([0.0], [values[0]], "o", color="black")
        state_markers.append(marker)

    def draw_frame(index):
        roa_path.set_data(path[0, : index + 1], path[1, : index + 1])
        roa_point.set_data([state_traj[0, index]], [state_traj[1, index]])
        # The massless swing leg is repositioned instantaneously at each impact.
        frame_params = dict(
            params, ankle_torque=torque_traj[index], alpha=result["alpha"][index]
        )
        model.visualize(state_traj[:, index], frame_params, ax=ax)
        ax.set_title(
            f"t = {time_traj[index]:.2f} s | α = {np.rad2deg(result['alpha'][index]):.2f}°"
        )
        for state_index, (time_marker, state_marker) in enumerate(
            zip(time_markers, state_markers)
        ):
            time_marker.set_xdata([time_traj[index], time_traj[index]])
            state_marker.set_data([time_traj[index]], [plot_trajs[state_index][index]])

    # Simulate at a small timestep, but render only 25 frames per second.
    fps = ANIMATION_FPS
    frame_stride = max(1, round(1 / (fps * result["timestep"])))
    frame_indices = list(range(0, time_traj.size, frame_stride))
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(time_traj.size - 1)

    animation = FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    animation.save(output / f"{name}.gif", writer=PillowWriter(fps=fps))

    # To save an MP4 instead, install FFmpeg and use:
    # animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
    print(f"Saved {output / f'{name}.gif'} ({completed_steps} footstrikes).")
    draw_frame(len(time_traj) - 1)
    fig.savefig(output / f"{name}_final.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    result = simulate_policy(INITIAL_STATE, timestep=TIMESTEP, sim_time=SIM_TIME)
    save_animation(result)
