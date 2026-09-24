"""
Find the shortest/longest converging route through the saved step map for a given initial state.

Run: uv run python -m assignment_2.assignment_2_longest_path

Uses DFS to figure out the shortest/longest converging route for the given initial state, and then plots both.
"""

import csv
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assignment_2.assignment_2 import save_animation
from assignment_2.assignment_2_simulation import simulate
from assignment_2.assignment_2_tables import load_converging_velocity_intervals
from models import inverted_pendulum_walker as model

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "poincare"
RETURN_FILE = OUTPUT_DIR / "poincare_results.npz"
ROA_FILE = OUTPUT_DIR.parent / "roa" / "roa_results.npz"
START_VELOCITY = 3.0
TIMESTEP = 0.001
SIM_TIME = 10.0


def nearest_indices(grid, values):
    """Return nearest grid indices for values known to lie inside the grid."""
    upper = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
    lower = upper - 1
    return np.where(
        abs(values - grid[lower]) <= abs(values - grid[upper]), lower, upper
    )


def converging_paths(start_velocity, velocities, alphas, returns, intervals):
    """Return the shortest and longest decreasing paths into the standing RoA."""
    if not velocities[0] <= start_velocity <= velocities[-1]:
        raise ValueError("Starting velocity is outside the return-map grid")

    converging = np.zeros(len(velocities), dtype=bool)
    for lower, upper in intervals:
        converging |= (velocities >= lower) & (velocities <= upper)

    valid = (
        np.isfinite(returns) & (returns >= velocities[0]) & (returns <= velocities[-1])
    )
    successors = np.full(returns.shape, -1, dtype=int)
    successors[valid] = nearest_indices(velocities, returns[valid])

    @cache
    def dfs(velocity_index):
        if converging[velocity_index]:
            return (), ()

        shortest = None
        longest = None
        for alpha_index in range(len(alphas)):
            next_index = successors[alpha_index, velocity_index]
            # Match build_table(): only use transitions that strictly descend
            # through the velocity grid. This excludes cycles and stalls.
            if next_index < 0 or next_index >= velocity_index:
                continue
            suffixes = dfs(int(next_index))
            if suffixes is None:
                continue
            short_candidate = (alpha_index,) + suffixes[0]
            long_candidate = (alpha_index,) + suffixes[1]
            if shortest is None or len(short_candidate) < len(shortest):
                shortest = short_candidate
            if longest is None or len(long_candidate) > len(longest):
                longest = long_candidate
        return None if shortest is None else (shortest, longest)

    start_index = int(nearest_indices(velocities, np.array([start_velocity]))[0])
    alpha_paths = dfs(start_index)
    if alpha_paths is None:
        return None

    paths = []
    for alpha_path in alpha_paths:
        velocity_indices = [start_index]
        for alpha_index in alpha_path:
            velocity_indices.append(int(successors[alpha_index, velocity_indices[-1]]))
        paths.append((np.asarray(velocity_indices), np.asarray(alpha_path, dtype=int)))
    return tuple(paths)


def simulate_path(alpha_indices, alphas):
    """Simulate one chosen alpha sequence from the configured starting state."""
    action = 0

    def choose_alpha(time, state, params):
        nonlocal action
        if action >= len(alpha_indices):
            raise RuntimeError("Alpha sequence ended before the state reached the RoA")
        params["alpha"] = float(alphas[alpha_indices[action]])
        action += 1

    result = simulate(
        [0.0, START_VELOCITY],
        model.generate_params(),
        timestep=TIMESTEP,
        sim_time=SIM_TIME,
        max_collisions=None,
        roa_file=ROA_FILE,
        section_controller=choose_alpha,
    )
    if result["balance_start_time"] is None:
        raise RuntimeError("The selected map path did not reach the RoA in simulation")
    if action != len(alpha_indices):
        raise RuntimeError(
            f"Simulation entered the RoA after {action} actions, but the map path "
            f"contains {len(alpha_indices)}"
        )
    return result


def save_path(kind, path, velocities, alphas, returns):
    """Save one path as a CSV, plot, and simulated GIF."""
    velocity_indices, alpha_indices = path
    rows = []
    for step, alpha_index in enumerate(alpha_indices, start=1):
        source_index = velocity_indices[step - 1]
        destination_index = velocity_indices[step]
        rows.append(
            (
                step,
                velocities[source_index],
                alphas[alpha_index],
                returns[alpha_index, source_index],
                velocities[destination_index],
            )
        )

    output_file = OUTPUT_DIR / f"{kind}_converging_path.csv"
    with output_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "grid_velocity_rad_s",
                "alpha_rad",
                "continuous_return_velocity_rad_s",
                "next_grid_velocity_rad_s",
            ]
        )
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    path_velocities = velocities[velocity_indices]
    ax.plot(range(len(path_velocities)), path_velocities, "o-")
    for step, alpha_index in enumerate(alpha_indices):
        ax.annotate(
            rf"$\alpha={np.rad2deg(alphas[alpha_index]):.2f}^\circ$",
            (step + 0.5, (path_velocities[step] + path_velocities[step + 1]) / 2),
            fontsize=8,
        )
    ax.set(
        xlabel="Step",
        ylabel="Upright velocity-grid state [rad/s]",
        title=f"{kind.title()} converging path: {len(rows)} steps",
        xticks=range(len(path_velocities)),
    )
    ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR / f"{kind}_converging_path.png", dpi=180)
    plt.close(fig)

    result = simulate_path(alpha_indices, alphas)
    save_animation(result, OUTPUT_DIR, f"{kind}_converging_path")
    print(f"{kind.title()} converging path: {len(rows)} steps")


def main():
    with np.load(RETURN_FILE) as data:
        velocities = data["initial_velocities"]
        alphas = data["alphas"]
        returns = data["return_velocities"]
    intervals, _ = load_converging_velocity_intervals(ROA_FILE)

    paths = converging_paths(START_VELOCITY, velocities, alphas, returns, intervals)
    if paths is None:
        print(f"No converging path found from {START_VELOCITY:g} rad/s")
        return

    for kind, path in zip(("shortest", "longest"), paths):
        save_path(kind, path, velocities, alphas, returns)


if __name__ == "__main__":
    main()
