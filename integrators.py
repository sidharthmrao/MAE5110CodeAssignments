import numpy as np


def explicit_euler(start_state, current_time, timestep, dynamics):
    return start_state + timestep * dynamics(current_time, start_state)


def rk4(start_state, current_time, timestep, dynamics):
    k_1 = dynamics(current_time, start_state)
    k_2 = dynamics(current_time + timestep / 2.0, start_state + k_1 * timestep / 2.0)
    k_3 = dynamics(current_time + timestep / 2.0, start_state + k_2 * timestep / 2.0)
    k_4 = dynamics(current_time + timestep, start_state + k_3 * timestep)

    return start_state + (timestep / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
