"""Plot simulated policy step counts over initial angle and velocity.

Run: uv run python -m assignment_2.assignment_2_policy_steps
"""

import json
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from assignment_2.assignment_2 import POLICY_FILE, ROA_FILE, simulate_policy
from assignment_2.assignment_2_config import (
    POINCARE_ALPHA_SAMPLES,
    POINCARE_VELOCITY_SAMPLES,
)
from assignment_2.assignment_2_simulation import OUTPUT_DIR
from models import inverted_pendulum_walker as model

RESULTS_DIR = OUTPUT_DIR / "policy_steps"
ANGLE_SAMPLES = 31
VELOCITY_SAMPLES = 61
WORKERS = 8
TIMESTEP = 0.001
SIM_TIME = 10.0


def simulate_start(start):
    """Return forward steps to RoA, or -1 when the policy does not converge."""
    try:
        result = simulate_policy(
            start,
            timestep=TIMESTEP,
            sim_time=SIM_TIME,
            early_convergence=True,
            roa_file=ROA_FILE,
            policy_file=POLICY_FILE,
        )
    except ValueError as error:
        if not str(error).startswith(("No alpha policy found", "Upright velocity")):
            raise
        return -1
    if result["balance_start_time"] is None or not np.all(np.isfinite(result["state"])):
        return -1
    return result["forward_steps"]


def main():
    if ANGLE_SAMPLES < 2 or VELOCITY_SAMPLES < 2:
        raise ValueError("Use at least two samples in each dimension")

    params = model.generate_params()
    lower_angle, upper_angle = model.angle_collision_bounds(params)
    max_velocity = np.sqrt(2 * params["gravity"] / params["spoke_length"])
    angles = np.linspace(lower_angle, upper_angle, ANGLE_SAMPLES)
    velocities = np.linspace(0.0, max_velocity, VELOCITY_SAMPLES)
    starts = np.array(np.meshgrid(angles, velocities, indexing="ij")).reshape(2, -1).T

    with Pool(WORKERS) as pool:
        values = pool.imap(simulate_start, starts, chunksize=10)
        step_counts = np.empty(len(starts), dtype=int)
        for index, value in enumerate(values):
            step_counts[index] = value
            if (index + 1) % 100 == 0:
                print(f"Simulated {index + 1}/{len(starts)} states", flush=True)
    step_counts = step_counts.reshape(ANGLE_SAMPLES, VELOCITY_SAMPLES)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "angle_samples": ANGLE_SAMPLES,
        "velocity_samples": VELOCITY_SAMPLES,
        "timestep": TIMESTEP,
        "sim_time": SIM_TIME,
        "failure_value": -1,
        "policy_file": str(POLICY_FILE),
        "roa_file": str(ROA_FILE),
    }
    np.savez_compressed(
        RESULTS_DIR / "policy_step_counts.npz",
        angles=angles,
        velocities=velocities,
        step_counts=step_counts,
        metadata=json.dumps(metadata),
    )

    max_steps = max(0, int(step_counts.max()))
    step_colors = [
        plt.get_cmap("viridis")(x) for x in np.linspace(0.15, 0.9, max_steps + 1)
    ]
    cmap = ListedColormap(["#bdbdbd", *step_colors])
    norm = BoundaryNorm(np.arange(-1.5, max_steps + 1.5), cmap.N)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    image = ax.pcolormesh(
        angles,
        velocities,
        step_counts.T,
        shading="nearest",
        cmap=cmap,
        norm=norm,
    )
    colorbar = fig.colorbar(image, ax=ax, ticks=range(max_steps + 1))
    colorbar.set_label("Forward footstrikes before entering RoA")
    ax.set(
        xlabel="Initial angle [rad]",
        ylabel="Initial angular velocity [rad/s]",
        title=(
            f"Simulated {POINCARE_ALPHA_SAMPLES} × "
            f"{POINCARE_VELOCITY_SAMPLES} policy: steps to standing RoA\n"
            f"{np.count_nonzero(step_counts >= 0)}/{step_counts.size} states converged"
        ),
    )
    ax.legend(handles=[Patch(color="#bdbdbd", label="Did not reach RoA")])
    fig.savefig(RESULTS_DIR / "policy_step_counts.png", dpi=220)
    plt.close(fig)

    print(
        f"Converged: {np.count_nonzero(step_counts >= 0)}/{step_counts.size}; "
        f"maximum successful step count: {max_steps}"
    )
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
