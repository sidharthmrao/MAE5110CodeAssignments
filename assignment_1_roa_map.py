"""
Analyze and plot results saved by assignment_1_roa_simulate.py.

Overall process:
 - Load results.
 - Plot an RoA map.
 - Plot a Poincare return map. While doing this:
   - Find the largest point where the return map crosses the identity (this corresponds to limit cycle fixed point when Floquet multiplier is positive)
   - Estimate Floquet multiplier by regressing around the fixed point
   - If the Floquet multiplier is negative, discard this fixed point since it most likely corresponds to a stopping convergence point.
 - Plot overall Floquet multiplier vs incline / num spokes maps.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from assignment_1_visualization import plot_poincare_return_map

from assignment_1_roa_simulate import (
    MODEL_PARAMS,
    NUMPY_DATA_DIR,
    NUM_SPOKES_SWEEP_INCLINE_DEGREES,
    OUTPUT_DIR,
    make_test_tag,
)

RESULTS_PATTERN = "assignment_1_roa_results_*.npz"
METADATA_DIR = OUTPUT_DIR / "metadata"
ROA_MAP_DIR = OUTPUT_DIR / "roa_maps"
POINCARE_MAP_DIR = OUTPUT_DIR / "poincare_maps"
MULTIPLIER_SWEEP_FIGURE = OUTPUT_DIR / "floquet_multiplier_vs_incline.png"
SPOKE_MULTIPLIER_SWEEP_FIGURE = OUTPUT_DIR / "floquet_multiplier_vs_num_spokes.png"

CONVERGENCE_COLORS = ["green", "red", "blue"]
CONVERGENCE_LABELS = ["Settled stably", "Settled unstable", "Was cycling"]
RETURN_MAP_BIN_COUNT = 200
IDENTITY_CONTACT_TOLERANCE = np.deg2rad(0.1)
LOCAL_FIT_HALF_WIDTH = np.deg2rad(2.0)
LOCAL_FIT_BIN_COUNT = 20
MULTIPLIER_LINE_HALF_WIDTH = np.deg2rad(20.0)


def load_results(results_file):
    """Load the saved numeric arrays and run configuration."""
    if not results_file.exists():
        raise FileNotFoundError(
            f"{results_file.name} does not exist. Run "
            "assignment_1_roa_simulate.py first."
        )

    with np.load(results_file, allow_pickle=False) as data:
        return {
            "initial_angles": data["initial_angles"],
            "initial_velocities": data["initial_velocities"],
            "convergence_map": data["convergence_map"],
            "poincare_current": data["poincare_current"],
            "poincare_next": data["poincare_next"],
            "metadata": json.loads(data["metadata"].item()),
            "results_file": results_file,
        }


def get_output_files(results):
    """Build matching output names from this test's saved parameters."""
    test_tag = make_test_tag(results["metadata"]["model_params"])
    return {
        "roa": ROA_MAP_DIR / f"assignment_1_roa_map_{test_tag}.png",
        "poincare": (POINCARE_MAP_DIR / f"assignment_1_poincare_map_{test_tag}.png"),
        "summary": METADATA_DIR / f"assignment_1_roa_analysis_{test_tag}.json",
    }


def plot_roa_map(results):
    ROA_MAP_DIR.mkdir(parents=True, exist_ok=True)
    convergence_map = results["convergence_map"]
    metadata = results["metadata"]
    model = metadata["model_params"]
    simulation = metadata["simulation_params"]
    colormap = ListedColormap(CONVERGENCE_COLORS)
    normalization = BoundaryNorm(
        np.arange(len(CONVERGENCE_COLORS) + 1) - 0.5,
        colormap.N,
    )

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.pcolormesh(
        np.rad2deg(results["initial_angles"]),
        np.rad2deg(results["initial_velocities"]),
        convergence_map.T,
        cmap=colormap,
        norm=normalization,
        shading="nearest",
    )
    axis.set_xlabel("Initial spoke angle [deg]")
    axis.set_ylabel("Initial angular velocity [deg/s]")
    axis.set_title(
        "Spoked Wheel RoA Map\n"
        f"Incline: {np.rad2deg(model['slope_incline']):g} [deg] | "
        f"Spokes: {model['num_spokes']} | "
        f"Simulation time: {simulation['sim_time']:g} s | "
        f"dt: {simulation['timestep']:g} s | "
        f"Grid: {metadata['angle_steps']} × {metadata['angular_velocity_steps']}"
    )
    axis.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(CONVERGENCE_COLORS, CONVERGENCE_LABELS)
        ],
        title="Convergence",
        loc="best",
    )
    figure.tight_layout()
    figure.savefig(get_output_files(results)["roa"], dpi=300, bbox_inches="tight")
    return figure, axis


def bin_return_map(current_velocities, next_velocities, bin_count):
    """
    Reduce repeated/noisy samples to one median point per velocity bin.
    We naturally get a lot of samples around the fixed point, so don't want to over-weight it when regressing.
    """
    finite = np.isfinite(current_velocities) & np.isfinite(next_velocities)
    current = np.asarray(current_velocities)[finite]
    following = np.asarray(next_velocities)[finite]
    if len(current) < 2:
        return current, following

    lower, upper = current.min(), current.max()
    if upper == lower:
        return np.array([lower]), np.array([np.median(following)])

    edges = np.linspace(lower, upper, bin_count + 1)
    bin_indices = np.clip(np.digitize(current, edges) - 1, 0, bin_count - 1)
    binned_current = []
    binned_next = []
    for bin_index in range(bin_count):
        in_bin = bin_indices == bin_index
        if np.any(in_bin):
            binned_current.append(np.median(current[in_bin]))
            binned_next.append(np.median(following[in_bin]))

    return np.asarray(binned_current), np.asarray(binned_next)


def estimate_largest_fixed_point(current_velocities, next_velocities):
    """
    Estimate the largest intersection after median-binning the noisy map.
    When this point has a positive Floquet Multiplier, it corresponds to a limit cycling fixed point.
    """
    current, following = bin_return_map(
        current_velocities, next_velocities, RETURN_MAP_BIN_COUNT
    )
    if len(current) < 2:
        return None

    error = following - current
    candidates = list(current[np.isclose(error, 0.0, atol=1e-10)])

    # Find points within a small threshold of identity line.
    contact_indices = np.flatnonzero(np.abs(error) <= IDENTITY_CONTACT_TOLERANCE)
    candidates.extend(current[contact_indices])

    # Find places where the error from identity line changes direction.
    for index in np.flatnonzero(error[:-1] * error[1:] < 0.0):
        fraction = -error[index] / (error[index + 1] - error[index])
        candidates.append(
            current[index] + fraction * (current[index + 1] - current[index])
        )

    # Select largest fixed point.
    if candidates:
        return float(np.max(candidates))

    return float(current[np.argmin(np.abs(error))])


def estimate_floquet_multiplier(fixed_point, results):
    """Fit line to velocity bins within +-2 deg/s of the fixed point."""
    if fixed_point is None:
        return None, None, np.array([], dtype=int), 0

    current = results["poincare_current"]
    following = results["poincare_next"]
    local_indices = np.flatnonzero(
        np.isfinite(current)
        & np.isfinite(following)
        & (np.abs(current - fixed_point) <= LOCAL_FIT_HALF_WIDTH)
    )
    local_current = current[local_indices]
    local_following = following[local_indices]
    if len(local_current) < 2 or np.ptp(local_current) <= np.finfo(float).eps:
        return None, None, local_indices, 0

    fit_current, fit_following = bin_return_map(
        local_current, local_following, LOCAL_FIT_BIN_COUNT
    )
    if len(fit_current) < 2:
        return None, None, local_indices, len(fit_current)

    multiplier, intercept = np.polyfit(fit_current, fit_following, 1)
    if multiplier < 0.0:
        return None, None, local_indices, len(fit_current)

    return float(multiplier), float(intercept), local_indices, len(fit_current)


def plot_poincare_map(results):
    """Plot all return samples, the largest fixed point (corresponding to limit cycling), and its Floquet Multiplier estimate, if a limit cycle case exists."""
    POINCARE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    current = results["poincare_current"]
    following = results["poincare_next"]
    metadata = results["metadata"]
    model = metadata["model_params"]
    simulation = metadata["simulation_params"]
    fixed_point = estimate_largest_fixed_point(current, following)
    (
        multiplier,
        line_intercept,
        local_indices,
        local_bin_count,
    ) = estimate_floquet_multiplier(fixed_point, results)

    figure, axis = plot_poincare_return_map(
        current,
        following,
        title=(
            "Post-impact Angular Velocity Return Map\n"
            f"Incline: {np.rad2deg(model['slope_incline']):g} [deg] | "
            f"Spokes: {model['num_spokes']} | "
            f"Simulation time: {simulation['sim_time']:g} s | "
            f"dt: {simulation['timestep']:g} s | "
            f"Grid: {metadata['angle_steps']} × "
            f"{metadata['angular_velocity_steps']}"
        ),
    )
    if len(current) != 0:
        if fixed_point is not None and multiplier is not None:
            fixed_degrees = np.rad2deg(fixed_point)
            axis.scatter(
                [fixed_degrees],
                [fixed_degrees],
                marker="o",
                s=110,
                color="red",
                edgecolor="black",
                zorder=5,
                label=f"Cycling Fixed Point: {fixed_degrees:.3f} deg/s",
            )

        if multiplier is not None and line_intercept is not None:
            line_x = np.array(
                [
                    fixed_point - MULTIPLIER_LINE_HALF_WIDTH,
                    fixed_point + MULTIPLIER_LINE_HALF_WIDTH,
                ]
            )
            line_y = multiplier * line_x + line_intercept
            axis.plot(
                np.rad2deg(line_x),
                np.rad2deg(line_y),
                color="darkorange",
                linewidth=2.5,
                linestyle=":",
                label=rf"Floquet Multiplier: $\lambda={multiplier:.4f}$",
            )

    axis.legend()
    figure.tight_layout()
    figure.savefig(get_output_files(results)["poincare"], dpi=300, bbox_inches="tight")
    return (
        figure,
        axis,
        fixed_point,
        multiplier,
        line_intercept,
        len(local_indices),
        local_bin_count,
    )


def save_summary(
    results,
    fixed_point,
    multiplier,
    line_intercept,
    local_fit_point_count,
    local_fit_bin_count,
):
    """Save fixed-point, multiplier, counts, and run parameters as JSON."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    convergences = results["metadata"]["convergences"]
    class_counts = {
        name: int(np.count_nonzero(results["convergence_map"] == index))
        for index, name in enumerate(convergences)
    }
    output_files = get_output_files(results)
    summary = {
        "results_file": results["results_file"].name,
        "roa_figure": output_files["roa"].name,
        "poincare_figure": output_files["poincare"].name,
        "fixed_point_rad_s": fixed_point,
        "fixed_point_deg_s": (
            None if fixed_point is None else float(np.rad2deg(fixed_point))
        ),
        "floquet_multiplier": multiplier,
        "multiplier_fit_intercept_rad_s": line_intercept,
        "multiplier_method": "median-binned least-squares fit around fixed point",
        "multiplier_fit_half_width_deg_s": float(np.rad2deg(LOCAL_FIT_HALF_WIDTH)),
        "multiplier_fit_point_count": local_fit_point_count,
        "multiplier_fit_bin_count": local_fit_bin_count,
        "return_map_pair_count": len(results["poincare_current"]),
        "convergence_counts": class_counts,
        **results["metadata"],
    }
    output_files["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved RoA plot to {output_files['roa']}")
    print(f"Saved return-map plot to {output_files['poincare']}")
    print(f"Saved analysis data to {output_files['summary']}")
    if fixed_point is not None:
        print(f"Largest fixed point: {np.rad2deg(fixed_point):.6f} deg/s")
    if multiplier is not None:
        print(f"Floquet multiplier: {multiplier:.6f}")


def plot_multiplier_vs_incline(inclines_degrees, multipliers, num_spokes):
    """Plot and save the Floquet multiplier across all valid tests."""
    inclines_degrees = np.asarray(inclines_degrees)
    multipliers = np.asarray(multipliers)
    valid = np.isfinite(inclines_degrees) & np.isfinite(multipliers)
    inclines_degrees = inclines_degrees[valid]
    multipliers = multipliers[valid]
    if len(inclines_degrees) == 0:
        print("No valid Floquet multipliers were available for the sweep plot.")
        return None, None

    order = np.argsort(inclines_degrees)
    inclines_degrees = inclines_degrees[order]
    multipliers = multipliers[order]

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(inclines_degrees, multipliers, "o-", color="tab:purple")
    axis.axhline(
        1.0, color="black", linestyle="--", linewidth=1.25, label=r"$\lambda=1$"
    )
    axis.axhline(
        -1.0,
        color="black",
        linestyle="--",
        linewidth=1.25,
        label=r"$\lambda=-1$",
    )
    axis.set_xlabel("Slope incline [deg]")
    axis.set_ylabel(r"Floquet multiplier $\lambda$")
    axis.set_title(
        "Floquet Multiplier vs. Slope Incline\n" f"Number of spokes: {num_spokes}"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(MULTIPLIER_SWEEP_FIGURE, dpi=300, bbox_inches="tight")
    print(f"Saved multiplier sweep plot to {MULTIPLIER_SWEEP_FIGURE}")
    return figure, axis


def plot_multiplier_vs_num_spokes(num_spokes, multipliers, incline_degrees):
    """Plot and save the Floquet multiplier across the spoke-count sweep."""
    num_spokes = np.asarray(num_spokes)
    multipliers = np.asarray(multipliers)
    valid = np.isfinite(num_spokes) & np.isfinite(multipliers)
    num_spokes = num_spokes[valid]
    multipliers = multipliers[valid]
    if len(num_spokes) == 0:
        print("No valid Floquet multipliers were available for the spoke plot.")
        return None, None

    order = np.argsort(num_spokes)
    num_spokes = num_spokes[order]
    multipliers = multipliers[order]

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(num_spokes, multipliers, "o-", color="tab:green")
    axis.axhline(
        1.0, color="black", linestyle="--", linewidth=1.25, label=r"$\lambda=1$"
    )
    axis.axhline(
        -1.0,
        color="black",
        linestyle="--",
        linewidth=1.25,
        label=r"$\lambda=-1$",
    )
    axis.set_xticks(num_spokes)
    axis.set_xlabel("Number of spokes")
    axis.set_ylabel(r"Floquet multiplier $\lambda$")
    axis.set_title(
        "Floquet Multiplier vs. Number of Spokes\n"
        f"Slope incline: {incline_degrees:g} [deg]"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(SPOKE_MULTIPLIER_SWEEP_FIGURE, dpi=300, bbox_inches="tight")
    print(f"Saved spoke sweep plot to {SPOKE_MULTIPLIER_SWEEP_FIGURE}")
    return figure, axis


if __name__ == "__main__":
    result_files = sorted(NUMPY_DATA_DIR.glob(RESULTS_PATTERN))
    if not result_files:
        raise FileNotFoundError(
            f"No saved tests matching {RESULTS_PATTERN} were found in "
            f"{NUMPY_DATA_DIR}. "
            "Run assignment_1_roa_simulate.py first."
        )

    print(f"Plotting {len(result_files)} saved test(s).")
    incline_values = []
    spoke_values = []
    multiplier_values = []
    for result_file in result_files:
        print(f"\nLoading {result_file.name}")
        saved_results = load_results(result_file)
        roa_figure, _ = plot_roa_map(saved_results)
        (
            poincare_figure,
            _,
            fixed_point,
            multiplier,
            line_intercept,
            local_fit_point_count,
            local_fit_bin_count,
        ) = plot_poincare_map(saved_results)
        save_summary(
            saved_results,
            fixed_point,
            multiplier,
            line_intercept,
            local_fit_point_count,
            local_fit_bin_count,
        )
        if multiplier is not None:
            incline_values.append(
                np.rad2deg(saved_results["metadata"]["model_params"]["slope_incline"])
            )
            spoke_values.append(saved_results["metadata"]["model_params"]["num_spokes"])
            multiplier_values.append(multiplier)
        plt.close(roa_figure)
        plt.close(poincare_figure)

    incline_values = np.asarray(incline_values)
    spoke_values = np.asarray(spoke_values)
    multiplier_values = np.asarray(multiplier_values)

    incline_sweep_spokes = MODEL_PARAMS["num_spokes"]
    incline_sweep = spoke_values == incline_sweep_spokes
    multiplier_figure, _ = plot_multiplier_vs_incline(
        incline_values[incline_sweep],
        multiplier_values[incline_sweep],
        incline_sweep_spokes,
    )
    if multiplier_figure is not None:
        plt.close(multiplier_figure)

    spoke_sweep = np.isclose(incline_values, NUM_SPOKES_SWEEP_INCLINE_DEGREES)
    spoke_figure, _ = plot_multiplier_vs_num_spokes(
        spoke_values[spoke_sweep],
        multiplier_values[spoke_sweep],
        NUM_SPOKES_SWEEP_INCLINE_DEGREES,
    )
    if spoke_figure is not None:
        plt.close(spoke_figure)
