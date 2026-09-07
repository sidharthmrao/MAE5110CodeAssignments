import numpy as np
import matplotlib.pyplot as plt

from integrators import rk4 as integrator
from models import spoked_wheel as model

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "slope_incline": np.pi / 4.0,
    "num_spokes": 6,
    "spoke_length": 1,
    "center_mass": 1,
}


# Starting angle must be between `slope_incline - alpha` and `slope_incline + alpha` ,
# where `alpha` is half the distance between two spokes (`pi / num_spokes`).
# This is to avoid starting in a collided position.
initial_state = np.array([np.pi / 9, 0.0])

#######################################

timestep = 1e-4

sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop
for step, t in enumerate(time_traj[:-1]):
    # First, calculate regular update (no collision).
    dynamics = lambda t, state: model.non_collision_dynamics(t, state, params)
    state_traj[:, step + 1] = integrator(state_traj[:, step], t, timestep, dynamics)

    # Check for collision and wrap to new spoke if needed.
    angle = state_traj[:, step + 1][0]
    angular_velocity = state_traj[:, step + 1][1]

    n = params["num_spokes"]
    incline = params["slope_incline"]

    # 2*alpha is the angle between two spokes
    alpha = np.pi / n

    if angle >= (alpha + incline):
        state_traj[:, step + 1][0] -= 2 * alpha
        state_traj[:, step + 1][1] = state_traj[:, step + 1][1] * np.cos(2.0 * alpha)

fig, (angle_ax, velocity_ax) = plt.subplots(2, 1, sharex=True)

velocity_ax.set_xlabel("Time [s]")
angle_ax.set_title("Spoked Wheel")

angle_ax.plot(time_traj, np.rad2deg(state_traj[0, :]))
angle_ax.set_ylabel("Spoke Angle [deg]")

velocity_ax.plot(time_traj, np.rad2deg(state_traj[1, :]))
velocity_ax.set_ylabel("Spoke Angular Velocity [deg/s]")

plt.tight_layout()
plt.show()
