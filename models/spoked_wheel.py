import numpy as np


def non_collision_dynamics(t, state, params):
    g = params["gravity"]
    l = params["spoke_length"]

    angle, angular_velocity = state

    # -- DYNAMICS --

    angular_acceleration = g / l * np.sin(angle)

    state_derivative = np.array([angular_velocity, angular_acceleration])

    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "slope_incline": np.pi / 4.0,
        "num_spokes": 6,
        "spoke_length": 1,
        "spoke_mass": 1,
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy
