from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model

params = model.generate_params()

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0
last_step = 0

backward_collision_angle, forward_collision_angle = model.angle_collision_bounds(params)
assert backward_collision_angle <= initial_state[0] <= forward_collision_angle


dynamics = partial(model.dynamics, params=params)


for step, t in enumerate(time_traj[:-1]):
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
    if completed_steps == desired_number_of_steps:
        break

time_traj = time_traj[: last_step + 1]
state_traj = state_traj[:, : last_step + 1]

fig = plt.figure(figsize=(12, 6), layout="constrained")
grid = fig.add_gridspec(2, 2)
ax = fig.add_subplot(grid[:, 0])
angle_ax = fig.add_subplot(grid[0, 1])
velocity_ax = fig.add_subplot(grid[1, 1], sharex=angle_ax)

angle_ax.plot(time_traj, state_traj[0], color="tab:blue")
velocity_ax.plot(time_traj, state_traj[1], color="tab:orange")
angle_ax.set_ylabel("Angle [rad]")
velocity_ax.set_ylabel("Angular velocity [rad/s]")
velocity_ax.set_xlabel("Time [s]")
angle_ax.set_title("State over time")
angle_ax.tick_params(labelbottom=False)
time_markers = []
state_markers = []
for state_index, state_ax in enumerate((angle_ax, velocity_ax)):
    state_ax.grid(alpha=0.3)
    state_ax.set_xlim(0.0, max(time_traj[-1], timestep))
    time_markers.append(state_ax.axvline(0.0, color="black", linestyle="--", alpha=0.6))
    marker, = state_ax.plot([0.0], [state_traj[state_index, 0]], "o", color="black")
    state_markers.append(marker)


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")
    for state_index, (time_marker, state_marker) in enumerate(
        zip(time_markers, state_markers)
    ):
        time_marker.set_xdata([time_traj[index], time_traj[index]])
        state_marker.set_data([time_traj[index]], [state_traj[state_index, index]])


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("assignment_2/output")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.close(fig)
