"""
Sweeps initial states and generates a return map for angular velocity when pendulum is upright.
Enables the ankle controller when the state enters the standing RoA.

Run: uv run python -m assignment_2.assignment_2_poincare
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assignment_2.assignment_2_config import (
    POINCARE_ALPHA_SAMPLES,
    POINCARE_VELOCITY_SAMPLES,
)
from assignment_2.assignment_2_simulation import simulate
from assignment_2.assignment_2_tables import (
    load_converging_velocity_intervals,
    load_standing_roa,
)
from models import inverted_pendulum_walker as model

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "poincare"
ROA_FILE = OUTPUT_DIR.parent / "roa" / "roa_results.npz"
TIMESTEP = 0.001
MAX_RETURN_TIME = 10.0


def next_return(
    initial_velocity,
    alpha,
    params,
    *,
    timestep=TIMESTEP,
    max_time=MAX_RETURN_TIME,
    roa=None,
):
    """Return the next upright velocity, or NaN if no forward return occurs."""

    if timestep <= 0 or max_time <= 0:
        raise ValueError("timestep and max_time must be positive")
    if not np.isfinite(initial_velocity) or initial_velocity < 0:
        raise ValueError("initial_velocity must be finite and nonnegative")
    if not params["alpha_min"] <= alpha <= params["alpha_max"]:
        raise ValueError("alpha is outside its permitted range")

    if initial_velocity == 0:
        return np.nan
    return simulate(
        [0.0, initial_velocity],
        dict(params, alpha=float(alpha)),
        timestep=timestep,
        sim_time=max_time,
        roa_file=ROA_FILE,
        roa=roa,
        poincare_return=True,
    )


def main():
    if POINCARE_ALPHA_SAMPLES < 2 or POINCARE_VELOCITY_SAMPLES < 2:
        raise ValueError("Use at least two samples per sweep dimension")

    ## SIMULATE RETURN MAP ##

    params = model.generate_params()
    roa = load_standing_roa(ROA_FILE)
    alphas = np.linspace(
        params["alpha_min"], params["alpha_max"], POINCARE_ALPHA_SAMPLES
    )

    # Velocities from 0 to speed with Froude number of 2.
    velocities = np.linspace(
        0.0,
        np.sqrt(2 * params["gravity"] / params["spoke_length"]),
        POINCARE_VELOCITY_SAMPLES,
    )
    returns = np.full((POINCARE_ALPHA_SAMPLES, POINCARE_VELOCITY_SAMPLES), np.nan)
    for i, alpha in enumerate(alphas):
        for j, velocity in enumerate(velocities):
            returns[i, j] = next_return(velocity, alpha, params, roa=roa)
        print(
            f"alpha={np.rad2deg(alpha):.2f} deg: "
            f"{np.count_nonzero(np.isfinite(returns[i]))}/"
            f"{POINCARE_VELOCITY_SAMPLES} returns",
            flush=True,
        )

    ## SAVE DATA ##

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_params": params,
        "controller": "roa_gated",
        "timestep": TIMESTEP,
        "max_return_time": MAX_RETURN_TIME,
        "section": "theta=0, angular velocity > 0; one forward impact",
    }
    np.savez_compressed(
        OUTPUT_DIR / "poincare_results.npz",
        alphas=alphas,
        initial_velocities=velocities,
        return_velocities=returns,
        metadata=json.dumps(metadata),
    )

    ## PLOT RETURN MAP ##

    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(alphas)))
    for i, (alpha, color) in enumerate(zip(alphas, colors)):
        ax.plot(
            velocities,
            returns[i],
            "-",
            color=color,
            label=rf"$\alpha={np.rad2deg(alpha):.2f}^\circ$",
        )
    ax.plot(velocities, velocities, "k--", alpha=0.6, label="Identity")
    ax.set(
        xlim=(velocities[0], velocities[-1]),
        xlabel=r"Initial velocity $\dot\theta_k$ [rad/s]",
        ylabel=r"Return velocity $\dot\theta_{k+1}$ [rad/s]",
        title="Forward return maps at θ = 0° (ankle control after RoA entry)",
    )
    converging_velocity_intervals, sampled_angle = load_converging_velocity_intervals(
        ROA_FILE
    )
    for interval_index, (lower, upper) in enumerate(converging_velocity_intervals):
        label = "Standing RoA at θ ≈ 0° (saved grid)" if interval_index == 0 else None
        ax.axhspan(
            lower,
            upper,
            color="tab:green",
            alpha=0.16,
            hatch="//",
            label=label,
            zorder=0,
        )
        ax.axhline(upper, color="tab:green", linestyle=":", linewidth=1)
        print(
            f"RoA target band: {lower:g} to {upper:g} rad/s "
            f"(nearest sampled θ = {np.rad2deg(sampled_angle):.5f} deg)"
        )
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR / "poincare_return_maps.png", dpi=240)
    plt.close(fig)
    print(f"Saved return-map plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
