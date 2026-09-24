"""Read upright velocity intervals from the saved balancing convergence-region table."""

import json

import numpy as np


def load_converging_velocity_intervals(
    roa_file,
) -> tuple[list[tuple[float, float]], float]:
    """
    Find successful velocity intervals in the RoA row nearest to upright position.

    Returns
    -------
    converging_velocity_intervals : list of (float, float)
        Inclusive nonnegative velocity bounds in rad/s; empty if none qualify.
    sampled_roa_angle : float
        Actual angle of the selected RoA row, in radians.
    """
    with np.load(roa_file) as data:
        roa_angles = data["angles"]
        if not roa_angles[0] <= 0 <= roa_angles[-1]:
            raise ValueError("RoA must span theta=0")
        upright_angle_index = int(np.argmin(abs(roa_angles)))
        roa_velocities = data["velocities"]
        converging_indices = np.flatnonzero(data["converged"][upright_angle_index])
        consecutive_index_groups = np.split(
            converging_indices, np.flatnonzero(np.diff(converging_indices) > 1) + 1
        )
        converging_velocity_intervals = [
            (
                max(0.0, float(roa_velocities[index_group[0]])),
                float(roa_velocities[index_group[-1]]),
            )
            for index_group in consecutive_index_groups
            if index_group.size and roa_velocities[index_group[-1]] >= 0
        ]
        return converging_velocity_intervals, float(roa_angles[upright_angle_index])


def load_standing_roa(roa_file):
    """Read the state grid, classification, and standing-model parameters."""
    with np.load(roa_file) as data:
        return {
            "angles": data["angles"],
            "velocities": data["velocities"],
            "converged": data["converged"],
            "params": json.loads(data["metadata"].item())["model_params"],
        }


def in_standing_roa(state, roa):
    """Require the enclosing RoA cell to have successful corners."""
    roa_angles = roa["angles"]
    roa_velocities = roa["velocities"]
    roa_converged = roa["converged"]
    if not (
        roa_angles[0] <= state[0] <= roa_angles[-1]
        and roa_velocities[0] <= state[1] <= roa_velocities[-1]
    ):
        return False
    # Require all corners of the enclosing cell to be marked safe. Nearest
    # neighbor alone can switch too early just outside the standing region.
    angle_low = max(0, np.searchsorted(roa_angles, state[0], side="right") - 1)
    angle_high = min(len(roa_angles) - 1, np.searchsorted(roa_angles, state[0]))
    velocity_low = max(0, np.searchsorted(roa_velocities, state[1], side="right") - 1)
    velocity_high = min(
        len(roa_velocities) - 1, np.searchsorted(roa_velocities, state[1])
    )
    return bool(
        np.all(
            roa_converged[angle_low : angle_high + 1, velocity_low : velocity_high + 1]
        )
    )
