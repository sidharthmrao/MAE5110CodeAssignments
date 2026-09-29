"""Ascending-velocity minimum-step lookup from saved upright return maps and standing RoA.

Run: uv run python -m assignment_2.assignment_2_step_policy

A step is one forward touchdown followed by a forward upright crossing.
Zero steps means the standing controller can be used immediately at the current
upright velocity. Success uses the saved RoA's balancing convergence region.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from assignment_2.assignment_2_tables import load_converging_velocity_intervals

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "poincare"
RETURN_FILE = OUTPUT_DIR / "poincare_results.npz"
ROA_FILE = OUTPUT_DIR.parent / "roa" / "roa_results.npz"


def build_table(
    initial_velocities, return_velocities, converging_velocity_intervals
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find minimum step counts in one pass from low to high velocity.

    Visit initial velocities from smallest to largest, trying every alpha.
    Look up each return at its nearest velocity-grid point, including returns
    near the standing region. Use 1 + the stored step count if it is known.
    Skip returns rounded to the current or a higher velocity index.

    initial_velocities: sorted 1D velocity grid.
    return_velocities: [alpha index, initial velocity index] return-map table.
    converging_velocity_intervals: inclusive (min, max) ranges where the standing controller can be used immediately.

    Returns
    -------
    min_steps_to_converge : ndarray of int, shape (number_of_velocities,)
        Minimum estimated footsteps before switching to the standing controller.
        0 means switch immediately; -1 means no sequence of sampled alpha
        choices was found to reach the standing region.
    best_alpha_indices : ndarray of int, shape (number_of_velocities,)
        First alpha index to choose; -1 means no action assigned, including
        states where the standing controller can be used immediately.
    steps_for_each_alpha : ndarray of int, shape return_velocities.shape
        Footsteps before switching to the standing controller, when this alpha
        is chosen first and the best available alpha is chosen on later steps.
        Rows index alphas, columns index initial velocities; -1 means no route
        or an action not evaluated because the initial state already converges.
    """
    # Filter velocities and figure out where they lie in the return map grid.
    valid_returns = (
        np.isfinite(return_velocities)
        & (return_velocities >= initial_velocities[0])
        & (return_velocities <= initial_velocities[-1])
    )
    return_velocities_for_lookup = np.where(
        valid_returns, return_velocities, initial_velocities[0]
    )
    upper_velocity_indices = np.clip(
        np.searchsorted(initial_velocities, return_velocities_for_lookup),
        1,
        len(initial_velocities) - 1,
    )
    lower_velocity_indices = upper_velocity_indices - 1
    next_velocity_indices = np.where(
        abs(return_velocities_for_lookup - initial_velocities[lower_velocity_indices])
        <= abs(
            return_velocities_for_lookup - initial_velocities[upper_velocity_indices]
        ),
        lower_velocity_indices,
        upper_velocity_indices,
    )

    # Velocities in the RoA can switch to the standing controller with zero steps.
    already_converging = np.zeros(initial_velocities.shape, dtype=bool)
    for lower, upper in converging_velocity_intervals:
        already_converging |= (initial_velocities >= lower) & (
            initial_velocities <= upper
        )

    min_steps_to_converge = np.full(len(initial_velocities), -1, dtype=int)
    min_steps_to_converge[already_converging] = 0
    best_alpha_indices = np.full(len(initial_velocities), -1, dtype=int)

    steps_for_each_alpha = np.full(return_velocities.shape, -1, dtype=int)

    # Work from low to high velocity so results for slower states are available.
    for velocity_index in range(len(initial_velocities)):
        if min_steps_to_converge[velocity_index] == 0:
            continue  # Switch to the standing controller immediately.

        for alpha_index in range(return_velocities.shape[0]):
            if not valid_returns[alpha_index, velocity_index]:
                continue

            next_velocity_index = next_velocity_indices[alpha_index, velocity_index]
            if next_velocity_index >= velocity_index:
                continue

            remaining_steps = min_steps_to_converge[next_velocity_index]
            if remaining_steps < 0:
                continue  # No known way to reach the standing region from here.

            # One footstep now, plus the steps stored for the return velocity.
            action_steps = 1 + remaining_steps
            steps_for_each_alpha[alpha_index, velocity_index] = action_steps
            current_best = min_steps_to_converge[velocity_index]
            if current_best < 0 or action_steps < current_best:
                min_steps_to_converge[velocity_index] = action_steps
                best_alpha_indices[velocity_index] = alpha_index

    return (
        min_steps_to_converge,
        best_alpha_indices,
        steps_for_each_alpha,
    )


def plot_step_map(
    initial_velocities,
    alphas,
    return_velocities,
    min_steps_to_converge,
    steps_for_each_alpha,
    converging_velocity_intervals,
):
    """Plot action step counts, minimum state costs, and the failure-region span."""
    max_step_count = int(
        max(min_steps_to_converge.max(), steps_for_each_alpha.max(), 0)
    )
    step_colors = ["#239b56"] + [
        plt.get_cmap("plasma")(color_position)
        for color_position in np.linspace(0.1, 0.9, max_step_count)
    ]
    cmap = ListedColormap(["#bcbcbc"] + step_colors)
    norm = BoundaryNorm(np.arange(-1.5, max_step_count + 1.5), cmap.N)
    fig, (return_map_axis, state_steps_axis) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, height_ratios=[5, 1], layout="constrained"
    )
    for alpha_index in range(len(alphas)):
        return_map_axis.plot(
            initial_velocities,
            return_velocities[alpha_index],
            color="0.8",
            linewidth=0.6,
            zorder=1,
        )
        valid_returns = np.isfinite(return_velocities[alpha_index])
        return_map_axis.scatter(
            initial_velocities[valid_returns],
            return_velocities[alpha_index, valid_returns],
            c=steps_for_each_alpha[alpha_index, valid_returns],
            cmap=cmap,
            norm=norm,
            s=8,
            zorder=2,
        )
    for min_velocity, max_velocity in converging_velocity_intervals:
        return_map_axis.axhspan(
            min_velocity, max_velocity, color=step_colors[0], alpha=0.18, zorder=0
        )
    # Measure the vertical spread of gray return points, not their initial velocities.
    # Zero-step states skip action evaluation, so exclude those placeholder costs.
    gray_points = (
        np.isfinite(return_velocities)
        & (steps_for_each_alpha < 0)
        & (min_steps_to_converge[np.newaxis, :] != 0)
    )
    if np.any(gray_points):
        gray_min_velocity = float(return_velocities[gray_points].min())
        gray_max_velocity = float(return_velocities[gray_points].max())
        gray_height = gray_max_velocity - gray_min_velocity
        bracket_x = initial_velocities[0] + 0.45 * (
            initial_velocities[-1] - initial_velocities[0]
        )
        return_map_axis.hlines(
            [gray_min_velocity, gray_max_velocity],
            initial_velocities[0],
            bracket_x,
            colors="0.4",
            linestyles=":",
            linewidth=0.8,
        )
        return_map_axis.annotate(
            "",
            xy=(bracket_x, gray_max_velocity),
            xytext=(bracket_x, gray_min_velocity),
            arrowprops={"arrowstyle": "|-|", "color": "0.25"},
        )
        return_map_axis.annotate(
            f"Failure Region Height ≈ {gray_height:.4f} rad/s\n"
            f"({gray_min_velocity:.4f}–{gray_max_velocity:.4f} rad/s)",
            xy=(bracket_x, (gray_min_velocity + gray_max_velocity) / 2),
            xytext=(12, 20),
            textcoords="offset points",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        print(f"Sampled gray velocity height: {gray_height:.6f} rad/s")
    return_map_axis.plot(initial_velocities, initial_velocities, "k--", alpha=0.4)
    return_map_axis.set(
        ylabel="Next upright velocity [rad/s]",
        title="Step Return Map per Alpha",
    )
    legend_entries = [
        Patch(
            color=step_colors[step_count],
            label=f"{step_count} steps" if step_count != 1 else "1 step",
        )
        for step_count in range(max_step_count + 1)
    ]
    legend_entries.append(Patch(color="#bcbcbc", label="No route found"))
    return_map_axis.legend(handles=legend_entries, ncol=4, fontsize=8)
    return_map_axis.grid(alpha=0.2)
    state_steps_axis.scatter(
        initial_velocities,
        np.zeros_like(initial_velocities),
        c=min_steps_to_converge,
        cmap=cmap,
        norm=norm,
        s=40,
        marker="s",
    )
    state_steps_axis.set(
        xlabel="Initial upright velocity [rad/s]",
        yticks=[],
        xlim=(initial_velocities[0], initial_velocities[-1]),
        title="Grid-estimated minimum steps over all α choices (green = 0)",
    )
    fig.savefig(OUTPUT_DIR / "poincare_steps.png", dpi=240)
    plt.close(fig)


def main():
    with np.load(RETURN_FILE) as data:
        initial_velocities = data["initial_velocities"]
        alphas = data["alphas"]
        return_velocities = data["return_velocities"]
        return_map_metadata = json.loads(data["metadata"].item())
    converging_velocity_intervals, sampled_roa_angle = (
        load_converging_velocity_intervals(ROA_FILE)
    )

    ## Calculate Steps to Converge ##
    (
        min_steps_to_converge,
        best_alpha_indices,
        steps_for_each_alpha,
    ) = build_table(
        initial_velocities, return_velocities, converging_velocity_intervals
    )

    best_alpha_values = np.full(min_steps_to_converge.shape, np.nan)
    best_alpha_values[best_alpha_indices >= 0] = alphas[
        best_alpha_indices[best_alpha_indices >= 0]
    ]

    ## SAVE DATA ##
    metadata = {
        "standing_intervals": converging_velocity_intervals,
        "roa_sample_angle": sampled_roa_angle,
        "lookup": "nearest velocity for all returns, including standing-region membership",
        "unreachable_value": -1,
        "return_map_settings": return_map_metadata,
        "criterion": "Reach saved collision-free standing region; not proof of asymptotic convergence",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "step_policy.npz",
        velocities=initial_velocities,
        alphas=alphas,
        min_steps=min_steps_to_converge,
        best_alpha=best_alpha_values,
        action_steps=steps_for_each_alpha,
        metadata=json.dumps(metadata),
    )
    with (OUTPUT_DIR / "step_policy.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["velocity_rad_s", "min_steps", "best_alpha_rad", "best_alpha_deg"]
        )
        writer.writerows(
            zip(
                initial_velocities,
                min_steps_to_converge,
                best_alpha_values,
                np.rad2deg(best_alpha_values),
            )
        )
    with (OUTPUT_DIR / "state_action_steps.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "velocity_rad_s",
                "alpha_rad",
                "return_velocity_rad_s",
                "steps_with_this_first_alpha",
            ]
        )
        for alpha_index, alpha in enumerate(alphas):
            writer.writerows(
                zip(
                    initial_velocities,
                    np.full(len(initial_velocities), alpha),
                    return_velocities[alpha_index],
                    steps_for_each_alpha[alpha_index],
                )
            )

    ## PLOT ##
    plot_step_map(
        initial_velocities,
        alphas,
        return_velocities,
        min_steps_to_converge,
        steps_for_each_alpha,
        converging_velocity_intervals,
    )

    print(f"Saved policy tables and poincare_steps.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
