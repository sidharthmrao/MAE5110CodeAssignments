"""Finite-horizon numerical RoA estimate for the controller in assignment_2.

Run from the repository root: uv run python -m assignment_2.assignment_2_roa
The default angle interval is halfway from upright to each collision bound.
A green cell means no collision occurred during the simulation horizon.
This operational convergence criterion does not require settling near upright.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from assignment_2.assignment_2 import OUTPUT_DIR, simulate
from models import inverted_pendulum_walker as model

# Initial-state grid (radians and radians/second).
ANGLE_STEPS = 121
VELOCITY_STEPS = 201
ANGLE_FRACTION = 1.0  # Fraction of each collision angle, measured from upright.
VELOCITY_MIN = -2.0
VELOCITY_MAX = 2.0

# Integration and convergence settings.
TIMESTEP = 0.005
SIM_TIME = 5.0
ROA_OUTPUT_DIR = OUTPUT_DIR / "roa"


def is_converged(result):
    """Count a trial as converged if it completed without a collision."""
    return result["completed_steps"] == 0


def sweep(
    angles,
    velocities,
    params,
    *,
    timestep=TIMESTEP,
    sim_time=SIM_TIME,
):
    """Classify each initial angle/velocity pair using the shared simulator."""
    converged = np.zeros((len(angles), len(velocities)), dtype=bool)
    collisions = np.zeros_like(converged, dtype=int)
    for i, angle in enumerate(angles):
        for j, velocity in enumerate(velocities):
            result = simulate(
                [angle, velocity],
                params,
                timestep=timestep,
                sim_time=sim_time,
                max_collisions=1,
            )
            converged[i, j] = is_converged(result)

            if converged[i, j]:
                print(f"Converged at angle {angle:.4f}, velocity {velocity:.4f}")

            collisions[i, j] = result["completed_steps"]
        print(f"Completed angle row {i + 1}/{len(angles)}", flush=True)
    return converged, collisions


def main():
    if ANGLE_STEPS < 2 or VELOCITY_STEPS < 2:
        raise ValueError("Each grid dimension needs at least two samples")
    if not 0 < ANGLE_FRACTION <= 1 or VELOCITY_MIN >= VELOCITY_MAX:
        raise ValueError("Use 0 < angle-fraction <= 1 and velocity-min < velocity-max")
    if (
        not np.isfinite(SIM_TIME)
        or SIM_TIME <= 0
        or not np.isfinite(TIMESTEP)
        or TIMESTEP <= 0
    ):
        raise ValueError("Use a positive finite timestep and simulation duration")
    params = model.generate_params()
    lower, upper = model.angle_collision_bounds(params)
    angles = np.linspace(ANGLE_FRACTION * lower, ANGLE_FRACTION * upper, ANGLE_STEPS)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, VELOCITY_STEPS)
    converged, collisions = sweep(
        angles,
        velocities,
        params,
        timestep=TIMESTEP,
        sim_time=SIM_TIME,
    )
    ROA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "angle_steps": ANGLE_STEPS,
        "velocity_steps": VELOCITY_STEPS,
        "angle_fraction": ANGLE_FRACTION,
        "velocity_min": VELOCITY_MIN,
        "velocity_max": VELOCITY_MAX,
        "timestep": TIMESTEP,
        "sim_time": SIM_TIME,
        "output_dir": str(ROA_OUTPUT_DIR),
        "model_params": params,
        "criterion": "No collision during the simulation horizon",
        "max_collisions": 1,
        "controller": "control in assignment_2.py",
    }
    np.savez_compressed(
        ROA_OUTPUT_DIR / "roa_results.npz",
        angles=angles,
        velocities=velocities,
        converged=converged,
        collisions=collisions,
        metadata=json.dumps(metadata),
    )
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    colors = ["#d9d9d9", "#309c63"]
    ax.pcolormesh(
        angles,
        velocities,
        converged.T,
        shading="nearest",
        cmap=ListedColormap(colors),
        norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
    )
    ax.plot(0, 0, "k+", markersize=10)
    ax.set(
        xlabel="Initial angle [rad]",
        ylabel="Initial angular velocity [rad/s]",
        title=f"Collision-free region ({SIM_TIME:g} s horizon)",
    )
    ax.legend(
        handles=[
            Patch(color=colors[1], label="No collision (converged)"),
            Patch(color=colors[0], label="Collision"),
        ]
    )
    fig.savefig(ROA_OUTPUT_DIR / "roa_map.png", dpi=180)
    plt.close(fig)
    print(
        f"Converged: {converged.sum()}/{converged.size}. Saved results to {ROA_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
