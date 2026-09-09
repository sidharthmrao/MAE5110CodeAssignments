"""Run simulation sweeps over different parameters (incline, number of spokes) and then save data for visualization."""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from assignment_1 import simulate

CONVERGENCES = ["SETTLED_STABLE", "SETTLED_UNSTABLE", "CYCLED"]

SIMULATION_PARAMS = {
    "timestep": 1e-3,
    "sim_time": 20.0,
    "full_stop_velocity_tolerance": 1e-5,
}
MODEL_PARAMS = {
    "gravity": 9.81,
    "slope_incline": np.deg2rad(20.0),
    "num_spokes": 6,
    "spoke_length": 1.0,
    "center_mass": 1.0,
}

ANGLE_STEPS = 45
MIN_ANGULAR_VELOCITY_DEGREES = -500.0
MAX_ANGULAR_VELOCITY_DEGREES = 300.0

ANGULAR_VELOCITY_RESOLUTION_DEGREES = 4.0
ANGULAR_VELOCITY_STEPS = (
    int(
        (MAX_ANGULAR_VELOCITY_DEGREES - MIN_ANGULAR_VELOCITY_DEGREES)
        / ANGULAR_VELOCITY_RESOLUTION_DEGREES
    )
    + 1
)
MIN_ANGULAR_VELOCITY = np.deg2rad(MIN_ANGULAR_VELOCITY_DEGREES)
MAX_ANGULAR_VELOCITY = np.deg2rad(MAX_ANGULAR_VELOCITY_DEGREES)

TRIAL_COUNT = ANGLE_STEPS * ANGULAR_VELOCITY_STEPS

INCLINATION_SWEEP_DEGREES = np.arange(0.0, 90.0 + 3.0, 3.0)
NUM_SPOKES_SWEEP = range(6, 13)
NUM_SPOKES_SWEEP_INCLINE_DEGREES = 20.0

NUM_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 1))
OUTPUT_DIR = Path(__file__).with_name("assignment_1_results")
NUMPY_DATA_DIR = OUTPUT_DIR / "numpy_data"


def make_test_tag(model_params):
    incline_degrees = np.rad2deg(model_params["slope_incline"])
    incline_tag = f"{incline_degrees:g}".replace("-", "neg").replace(".", "p")
    return f"incline_{incline_tag}deg_{model_params['num_spokes']}_spokes"


def make_results_file(model_params):
    return (
        NUMPY_DATA_DIR / f"assignment_1_roa_results_{make_test_tag(model_params)}.npz"
    )


def simulate_initial_state(task):
    angle_index, velocity_index, angle, angular_velocity, model_params = task
    _, _, convergence, section_velocities = simulate(
        np.array([angle, angular_velocity]),
        SIMULATION_PARAMS,
        model_params,
    )
    return angle_index, velocity_index, convergence, section_velocities


def get_simulation_results(model_params):
    alpha = np.pi / model_params["num_spokes"]
    incline = model_params["slope_incline"]
    initial_angles = np.linspace(incline - alpha, incline + alpha, ANGLE_STEPS)
    initial_velocities = np.linspace(
        MIN_ANGULAR_VELOCITY,
        MAX_ANGULAR_VELOCITY,
        ANGULAR_VELOCITY_STEPS,
    )
    convergence_map = np.empty((ANGLE_STEPS, ANGULAR_VELOCITY_STEPS), dtype=np.int8)
    poincare_current = []
    poincare_next = []
    tasks = [
        (angle_index, velocity_index, angle, velocity, model_params)
        for angle_index, angle in enumerate(initial_angles)
        for velocity_index, velocity in enumerate(initial_velocities)
    ]
    chunksize = max(1, len(tasks) // (NUM_WORKERS * 8))
    last_percentage = -1

    # Parallelize to speed stuff up, these sims are independent.
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = executor.map(simulate_initial_state, tasks, chunksize=chunksize)
        for completed, result in enumerate(results, start=1):
            angle_index, velocity_index, convergence, section_velocities = result
            convergence_map[angle_index, velocity_index] = CONVERGENCES.index(
                convergence
            )
            if len(section_velocities) >= 2:
                poincare_current.extend(section_velocities[:-1])
                poincare_next.extend(section_velocities[1:])

            percentage = int(100 * completed / len(tasks))
            if percentage != last_percentage:
                print(f"Simulating: {percentage:3d}%", end="\r", flush=True)
                last_percentage = percentage

    print("Simulating: 100%")
    return (
        initial_angles,
        initial_velocities,
        convergence_map,
        np.asarray(poincare_current),
        np.asarray(poincare_next),
    )


def save_results(
    initial_angles,
    initial_velocities,
    convergence_map,
    poincare_current,
    poincare_next,
    model_params,
    results_file,
):
    NUMPY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "simulation_params": SIMULATION_PARAMS,
        "model_params": model_params,
        "angle_steps": ANGLE_STEPS,
        "angular_velocity_steps": ANGULAR_VELOCITY_STEPS,
        "num_workers": NUM_WORKERS,
        "trial_count": TRIAL_COUNT,
        "convergences": CONVERGENCES,
    }
    np.savez_compressed(
        results_file,
        initial_angles=initial_angles,
        initial_velocities=initial_velocities,
        convergence_map=convergence_map,
        poincare_current=poincare_current,
        poincare_next=poincare_next,
        metadata=np.array(json.dumps(metadata)),
    )
    print(f"Saved simulation results to {results_file}")


def run_simulation_if_needed(model_params):
    """Run the sweep only when its result file is not already present."""
    results_file = make_results_file(model_params)
    if results_file.exists():
        print(f"Using existing simulation results: {results_file}")
        return

    incline_degrees = np.rad2deg(model_params["slope_incline"])
    print(
        f"Starting incline {incline_degrees:g} [deg] with "
        f"{model_params['num_spokes']} spokes"
    )
    save_results(
        *get_simulation_results(model_params),
        model_params=model_params,
        results_file=results_file,
    )


def run_inclination_sweep():
    """Run a sweep over different inclinations."""
    for incline_degrees in INCLINATION_SWEEP_DEGREES:
        model_params = MODEL_PARAMS.copy()
        model_params["slope_incline"] = np.deg2rad(incline_degrees)
        run_simulation_if_needed(model_params)


def run_num_spokes_sweep():
    """Run a sweep over different numbers of spokes."""
    for num_spokes in NUM_SPOKES_SWEEP:
        model_params = MODEL_PARAMS.copy()
        model_params["slope_incline"] = np.deg2rad(NUM_SPOKES_SWEEP_INCLINE_DEGREES)
        model_params["num_spokes"] = num_spokes
        run_simulation_if_needed(model_params)


if __name__ == "__main__":
    run_inclination_sweep()
    run_num_spokes_sweep()
