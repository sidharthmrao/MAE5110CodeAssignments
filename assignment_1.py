"""Contains base simulation and spoked wheel dynamics. Running this file directly runs a single simulation with configured parameters."""

import numpy as np
import matplotlib.pyplot as plt

from integrators import rk4 as integrator
from assignment_1_visualization import plot_poincare_return_map


def spoked_wheel_non_collision_dynamics(state, params):
    g = params["gravity"]
    l = params["spoke_length"]

    angle, angular_velocity = state

    # -- DYNAMICS --

    angular_acceleration = g / l * np.sin(angle)

    state_derivative = np.array([angular_velocity, angular_acceleration])

    return state_derivative


def simulate(initial_state, simulation_params, model_params):
    # Set up simulation.
    timestep = simulation_params["timestep"]
    sim_time = simulation_params["sim_time"]
    full_stop_velocity_tolerance = simulation_params["full_stop_velocity_tolerance"]
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep

    # Poincare Section of angular velocity post-collision
    section_angular_velocity = []

    # 2*alpha is the angle between two spokes
    n = model_params["num_spokes"]
    alpha = np.pi / n

    incline = model_params["slope_incline"]

    # Ensure that initial state is within valid bounds (not colliding).
    assert (initial_state[0] >= (incline - alpha)) and (
        initial_state[0] <= (incline + alpha)
    )

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    # Simulation
    convergence = "CYCLED"  # Options: SETTLED_STABLE or SETTLED_UNSTABLE or CYCLED

    for step, t in enumerate(time_traj[:-1]):
        # First, calculate regular update (no collision).
        dynamics = lambda _t, state: spoked_wheel_non_collision_dynamics(
            state, model_params
        )
        state_traj[:, step + 1] = integrator(state_traj[:, step], t, timestep, dynamics)

        # Check for collision and wrap to new spoke if needed.
        angle = state_traj[:, step + 1][0]

        # Forward collision.
        if angle > (incline + alpha):
            state_traj[:, step + 1][0] = incline - alpha
            state_traj[:, step + 1][1] = state_traj[:, step][1] * np.cos(2.0 * alpha)

            # Collided, fully stopped.
            if abs(state_traj[:, step + 1][1]) < full_stop_velocity_tolerance:
                # Propogate current angle and 0 velocity for rest of sim and finish.
                state_traj[:, (step + 1) :][0] = state_traj[:, step + 1][0]
                state_traj[:, (step + 1) :][1] = 0.0

                section_angular_velocity.append(0.0)

                convergence = "SETTLED_STABLE"
                break

            section_angular_velocity.append(state_traj[:, step + 1][1])

        # Backward collision.
        elif angle < (incline - alpha):
            state_traj[:, step + 1][0] = incline + alpha
            state_traj[:, step + 1][1] = state_traj[:, step][1] * np.cos(2.0 * alpha)

            # Collided, fully stopped.
            if abs(state_traj[:, step + 1][1]) < full_stop_velocity_tolerance:
                # Propogate current angle and 0 velocity for rest of sim and finish.
                state_traj[:, (step + 1) :][0] = state_traj[:, step + 1][0]
                state_traj[:, (step + 1) :][1] = 0.0

                section_angular_velocity.append(0.0)

                convergence = "SETTLED_STABLE"
                break

            section_angular_velocity.append(state_traj[:, step + 1][1])

        # Stopped while in vertical orientation (unstable).
        elif (
            abs(state_traj[:, step + 1][1]) < full_stop_velocity_tolerance
            and abs(state_traj[:, step + 1][0]) < 1e-4
        ):
            convergence = "SETTLED_UNSTABLE"

            state_traj[:, (step + 1) :][0] = state_traj[:, step + 1][0]
            state_traj[:, (step + 1) :][1] = 0.0

            break

    return time_traj, state_traj, convergence, section_angular_velocity


def plot_poincare_section(section_angular_velocity):
    """Plot the step-to-step return map for post-impact angular velocity."""
    section_angular_velocity = np.asarray(section_angular_velocity)
    current_velocity = section_angular_velocity[:-1]
    next_velocity = section_angular_velocity[1:]
    figure, axis = plot_poincare_return_map(
        current_velocity,
        next_velocity,
        sample_label="Return map",
        sample_size=35,
        sample_alpha=1.0,
    )
    axis.legend()
    figure.tight_layout()
    return figure, axis


if __name__ == "__main__":
    simulation_params = {
        "timestep": 1e-4,
        "sim_time": 20.0,
        "full_stop_velocity_tolerance": 1e-5,
    }

    model_params = {
        "gravity": 9.81,  # gravity m/s^2)
        "slope_incline": np.deg2rad(20),
        "num_spokes": 6,
        "spoke_length": 1,
        "center_mass": 1,
    }

    # Starting angle must be between `slope_incline - alpha` and `slope_incline + alpha` ,
    # where `alpha` is half the distance between two spokes (`pi / num_spokes`).
    # This is to avoid starting in a collided position.
    initial_state = np.array([np.deg2rad(25.0), 0.0])

    time_traj, state_traj, convergence, poincare_section_angular_vel = simulate(
        initial_state, simulation_params, model_params
    )

    fig, (angle_ax, velocity_ax) = plt.subplots(2, 1, sharex=True)

    velocity_ax.set_xlabel("Time [s]")
    angle_ax.set_title(
        f"Spoked Wheel | Start Angle: {np.rad2deg(initial_state[0])} [deg] | Start Ang Vel: {np.rad2deg(initial_state[1])} [deg/s] | Incline: {np.rad2deg(model_params['slope_incline'])} [deg] | Num Spokes: {model_params["num_spokes"]} | Spoke Length: {model_params["spoke_length"]}"
    )

    angle_ax.plot(time_traj, np.rad2deg(state_traj[0, :]))
    angle_ax.set_ylabel("Spoke Angle [deg]")

    velocity_ax.plot(time_traj, np.rad2deg(state_traj[1, :]))
    velocity_ax.set_ylabel("Spoke Angular Velocity [deg/s]")

    plt.tight_layout()
    plot_poincare_section(poincare_section_angular_vel)
    plt.show()
