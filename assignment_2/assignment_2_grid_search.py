"""Compare policy grid resolutions on the same random starting states.

Run: uv run python -m assignment_2.assignment_2_grid_search
Uses the existing standing RoA. Success means reaching balancing within
WALK_TIME, then surviving BALANCE_TIME without another collision.
"""

import csv
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assignment_2.assignment_2 import ROA_FILE, simulate_policy
from assignment_2.assignment_2_poincare import MAX_RETURN_TIME, next_return
from assignment_2.assignment_2_simulation import control
from assignment_2.assignment_2_simulation import simulate as simulate_standing
from assignment_2.assignment_2_step_policy import build_table
from assignment_2.assignment_2_tables import (
    load_converging_velocity_intervals,
    load_standing_roa,
)
from models import inverted_pendulum_walker as model

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "grid_search"
GRID_RESOLUTIONS = [
    (2, 11),
    (2, 15),
    (2, 18),
    (2, 19),
    (2, 20),
    (2, 21),
    (2, 22),
    (2, 23),
    (2, 24),
    (3, 15),
    (4, 22),
    (8, 45),
    (15, 90),
    (30, 180),
]
WORKERS = 8
NUM_STARTS = 4000
SEED = 42
TARGET_SUCCESS = 0.95
RANDOM_ANGLES = False
TIMESTEP = 0.001
WALK_TIME = 10.0
BALANCE_TIME = 5.0


def plot_success_regions(results_dir):
    """Plot saved grid-search outcomes without rerunning simulations."""
    with np.load(results_dir / "trials.npz") as data:
        starts = data["starts"]
        grids = data["grids"]
        outcomes = data["outcomes"]
        settings = json.loads(data["metadata"].item())

    styles = {
        "success": ("Balanced successfully", "#21864a", "o"),
        "no_policy": ("No valid policy action", "#c94444", "x"),
        "timeout": ("Did not reach balancing in time", "#c18116", "^"),
        "balance_collision": ("Collision while balancing", "#8b4aa8", "s"),
        "nonfinite": ("Nonfinite simulation", "#333333", "+"),
    }
    images_dir = results_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    upright_starts = np.all(starts[:, 0] == 0)
    for grid, results in zip(grids, outcomes):
        fig, ax = plt.subplots(figsize=(9, 5) if upright_starts else (8, 7))
        for outcome, (label, color, marker) in styles.items():
            mask = results == outcome
            if np.any(mask):
                ax.scatter(
                    starts[mask, 1] if upright_starts else starts[mask, 0],
                    (
                        np.full(np.count_nonzero(mask), int(outcome == "success"))
                        if upright_starts
                        else starts[mask, 1]
                    ),
                    color=color,
                    marker=marker,
                    label=label,
                    s=42,
                    alpha=0.85,
                    linewidths=1.3,
                )
        successes = np.count_nonzero(results == "success")
        ax.set(
            title=f"{grid[0]} alpha × {grid[1]} velocity samples\n"
            f"{successes}/{len(starts)} succeeded ({successes / len(starts):.1%})",
        )
        if upright_starts:
            ax.set(
                xlabel="Initial angular velocity [rad/s] (θ = 0)",
                ylabel="Outcome",
                xlim=settings["velocity_range"],
                ylim=(-0.3, 1.3),
                yticks=[0, 1],
                yticklabels=["Failed", "Balanced"],
            )
        else:
            ax.set(
                xlabel="Initial angle [rad]",
                ylabel="Initial angular velocity [rad/s]",
                ylim=settings["velocity_range"],
            )
            ax.axvline(0, color="0.6", linewidth=0.7, zorder=0)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        fig.suptitle("Walking policy and balancing control", fontsize=14)
        fig.legend(
            loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2, frameon=False
        )
        fig.tight_layout(rect=(0, 0.2, 1, 0.94))
        fig.savefig(images_dir / f"success_regions_{grid[0]}x{grid[1]}.png", dpi=220)
        plt.close(fig)
    print(f"Saved success-region images to {images_dir}")


def evaluate_start(start, *, params, policy_file):
    """Test one start through walking and five seconds of continued balancing."""
    try:
        result = simulate_policy(
            start,
            params,
            timestep=TIMESTEP,
            sim_time=WALK_TIME,
            early_convergence=True,
            roa_file=ROA_FILE,
            policy_file=policy_file,
        )
    except ValueError as error:
        # Missing/out-of-range policy actions are failures; other errors are bugs.
        if not str(error).startswith(("No alpha policy found", "Upright velocity")):
            raise
        outcome = "no_policy"
    else:
        if not np.all(np.isfinite(result["state"])):
            outcome = "nonfinite"
        elif result["balance_start_time"] is None:
            outcome = "timeout"
        else:
            standing = simulate_standing(
                result["state"][:, -1],
                result["params"],
                timestep=TIMESTEP,
                sim_time=BALANCE_TIME,
                # This rollout continues a controller already enabled by RoA entry.
                controller=control,
            )
            if not np.all(np.isfinite(standing["state"])):
                outcome = "nonfinite"
            elif standing["completed_steps"]:
                outcome = "balance_collision"
            else:
                outcome = "success"
    return outcome


def main():
    if NUM_STARTS < 1 or not 0 < TARGET_SUCCESS <= 1:
        raise ValueError("Use positive NUM_STARTS and 0 < TARGET_SUCCESS <= 1")
    grids = sorted(set(GRID_RESOLUTIONS), key=lambda grid: (grid[0] * grid[1], grid))
    if not grids or any(a < 2 or v < 2 for a, v in grids):
        raise ValueError("Each grid needs at least two samples per dimension")
    if min(TIMESTEP, WALK_TIME, BALANCE_TIME) <= 0:
        raise ValueError("Timestep and simulation durations must be positive")

    # Use the same model as the saved standing region.
    with np.load(ROA_FILE) as data:
        params = json.loads(data["metadata"].item())["model_params"]
    roa = load_standing_roa(ROA_FILE)
    intervals, sampled_angle = load_converging_velocity_intervals(ROA_FILE)
    max_velocity = np.sqrt(2 * params["gravity"] / params["spoke_length"])
    rng = np.random.default_rng(SEED)
    starts = np.zeros((NUM_STARTS, 2))
    starts[:, 1] = rng.uniform(0, max_velocity, NUM_STARTS)
    if RANDOM_ANGLES:
        lower, upper = model.angle_collision_bounds(params)
        starts[:, 0] = rng.uniform(lower, upper, NUM_STARTS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outcomes = np.full((len(grids), NUM_STARTS), "not_run", dtype="U24")
    rows = []
    selected = None
    for grid_index, (alpha_count, velocity_count) in enumerate(grids):
        print(f"Building {alpha_count} × {velocity_count} grid", flush=True)
        alphas = np.linspace(params["alpha_min"], params["alpha_max"], alpha_count)
        velocities = np.linspace(0, max_velocity, velocity_count)
        with Pool(WORKERS) as pool:
            returns = np.array(
                pool.starmap(
                    partial(next_return, timestep=TIMESTEP, roa=roa),
                    ((v, a, params) for a in alphas for v in velocities),
                    chunksize=32,
                )
            ).reshape(alpha_count, velocity_count)
        min_steps, alpha_indices, _ = build_table(velocities, returns, intervals)
        best_alpha = np.full(velocity_count, np.nan)
        valid = alpha_indices >= 0
        best_alpha[valid] = alphas[alpha_indices[valid]]
        policy_file = OUTPUT_DIR / f"policy_{alpha_count}x{velocity_count}.npz"
        np.savez_compressed(
            policy_file,
            velocities=velocities,
            alphas=alphas,
            min_steps=min_steps,
            best_alpha=best_alpha,
            metadata=json.dumps(
                {
                    "standing_intervals": intervals,
                    "roa_sample_angle": sampled_angle,
                    "return_map_settings": {
                        "model_params": params,
                        "timestep": TIMESTEP,
                        "max_return_time": MAX_RETURN_TIME,
                        "controller": "roa_gated",
                    },
                }
            ),
        )

        with Pool(WORKERS) as pool:
            trials = pool.imap(
                partial(evaluate_start, params=params, policy_file=policy_file),
                starts,
                chunksize=5,
            )
            for start_index, outcome in enumerate(trials):
                outcomes[grid_index, start_index] = outcome
                if (start_index + 1) % 50 == 0:
                    print(f"  Tested {start_index + 1}/{NUM_STARTS} starts", flush=True)

        successes = int(np.count_nonzero(outcomes[grid_index] == "success"))
        rate = successes / NUM_STARTS
        rows.append(
            (
                alpha_count,
                velocity_count,
                alpha_count * velocity_count,
                successes,
                NUM_STARTS,
                rate,
            )
        )
        print(
            f"{alpha_count} × {velocity_count}: {successes}/{NUM_STARTS} = {rate:.1%}",
            flush=True,
        )
        if selected is None and rate >= TARGET_SUCCESS:
            selected = {
                "alpha_samples": alpha_count,
                "velocity_samples": velocity_count,
                "success_rate": rate,
                "policy_file": str(policy_file),
            }

    with (OUTPUT_DIR / "success_rates.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "alpha_samples",
                "velocity_samples",
                "table_size",
                "successes",
                "trials",
                "success_rate",
            ]
        )
        writer.writerows(rows)
    settings = {
        "seed": SEED,
        "num_starts": NUM_STARTS,
        "random_angles": RANDOM_ANGLES,
        "velocity_range": [0.0, float(max_velocity)],
        "timestep": TIMESTEP,
        "walk_time": WALK_TIME,
        "balance_time": BALANCE_TIME,
        "target_success": TARGET_SUCCESS,
        "roa_file": str(ROA_FILE),
        "selection": "Smallest tested table meeting the observed success target",
        "success_criterion": "Reach balancing, then no collisions during balance_time",
        "controller": "roa_gated",
        "balancing_check": "Every timestep and upright crossing; all surrounding RoA corners successful",
        "selected": selected,
    }
    np.savez_compressed(
        OUTPUT_DIR / "trials.npz",
        starts=starts,
        grids=grids,
        outcomes=outcomes,
        metadata=json.dumps(settings),
    )
    (OUTPUT_DIR / "selection.json").write_text(json.dumps(settings, indent=2) + "\n")

    fig, ax = plt.subplots(layout="constrained")
    ax.plot([row[2] for row in rows], [100 * row[5] for row in rows], "o-")
    ax.axhline(
        100 * TARGET_SUCCESS,
        linestyle="--",
        color="gray",
        label=f"Target: {TARGET_SUCCESS:.0%}",
    )
    ax.set(
        xlabel="Table entries (alpha samples × velocity samples)",
        ylabel="Successful starts [%]",
        ylim=(0, 101),
        title=f"Policy grid comparison ({NUM_STARTS} random starts)",
    )
    ax.legend()
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(images_dir / "success_rates.png", dpi=180)
    plt.close(fig)
    plot_success_regions(OUTPUT_DIR)
    if selected is None:
        print(f"No tested grid reached {TARGET_SUCCESS:.0%} success.")
    else:
        print(
            f"Selected {selected['alpha_samples']} × {selected['velocity_samples']} "
            f"with {selected['success_rate']:.1%} observed success."
        )
    print(f"Saved experiment to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
