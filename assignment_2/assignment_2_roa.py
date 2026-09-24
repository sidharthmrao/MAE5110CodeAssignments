"""
Construct an Region of Attraction map for the walker with active balancing control.

Run from the repository root: uv run python -m assignment_2.assignment_2_roa
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from assignment_2.assignment_2_simulation import OUTPUT_DIR, control, simulate
from models import inverted_pendulum_walker as model

ROA_OUTPUT_DIR = OUTPUT_DIR / "roa"
ANGLE_FRACTION = 1.0
ANGLE_STEPS = 121
VELOCITY_MIN = -2.0
VELOCITY_MAX = 2.0
VELOCITY_STEPS = 201
TIMESTEP = 0.005
SIM_TIME = 5.0


def sweep(
    angles,
    velocities,
    params,
    *,
    timestep=TIMESTEP,
    sim_time=SIM_TIME,
):
    """Sweep over specified initial states, classify convergence region."""
    converged = np.zeros((len(angles), len(velocities)), dtype=bool)
    for i, angle in enumerate(angles):
        for j, velocity in enumerate(velocities):
            result = simulate(
                [angle, velocity],
                params,
                timestep=timestep,
                sim_time=sim_time,
                max_collisions=1,
                # Measure the controller basin without requiring a preexisting RoA.
                controller=control,
            )
            converged[i, j] = result["completed_steps"] == 0
        print(f"Completed angle row {i + 1}/{len(angles)}", flush=True)
    return converged


def main():
    if ANGLE_STEPS < 2 or VELOCITY_STEPS < 2:
        raise ValueError("Each grid dimension needs at least two samples")
    if not 0 < ANGLE_FRACTION <= 1 or VELOCITY_MIN >= VELOCITY_MAX:
        raise ValueError("Use 0 < angle_fraction <= 1 and velocity_min < velocity_max")
    if (
        not np.isfinite(SIM_TIME)
        or SIM_TIME <= 0
        or not np.isfinite(TIMESTEP)
        or TIMESTEP <= 0
    ):
        raise ValueError("Use a positive finite timestep and simulation duration")

    ## SIMULATE RoA MAP ##

    params = model.generate_params()
    lower, upper = model.angle_collision_bounds(params)
    angles = np.linspace(ANGLE_FRACTION * lower, ANGLE_FRACTION * upper, ANGLE_STEPS)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, VELOCITY_STEPS)
    converged = sweep(
        angles,
        velocities,
        params,
        timestep=TIMESTEP,
        sim_time=SIM_TIME,
    )

    ## SAVE DATA ##

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
        "controller": "control in assignment_2_simulation.py",
    }
    np.savez_compressed(
        ROA_OUTPUT_DIR / "roa_results.npz",
        angles=angles,
        velocities=velocities,
        converged=converged,
        metadata=json.dumps(metadata),
    )

    ## PLOT ##

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
        title="Region of Attraction for Upright Balancer",
    )
    ax.legend(
        handles=[
            Patch(color=colors[1], label="Balanced"),
            Patch(color=colors[0], label="Failed"),
        ]
    )
    fig.savefig(ROA_OUTPUT_DIR / "roa_map.png", dpi=180)
    plt.close(fig)
    print(
        f"Converged: {converged.sum()}/{converged.size}. Saved results to {ROA_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
