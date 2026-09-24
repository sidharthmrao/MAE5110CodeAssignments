import json

import numpy as np
import pytest

from assignment_2 import assignment_2 as policy
from assignment_2.assignment_2_simulation import simulate
from models.inverted_pendulum_walker import generate_params


@pytest.fixture
def tables(tmp_path):
    params = generate_params()
    roa_file = tmp_path / "roa.npz"
    policy_file = tmp_path / "policy.npz"
    np.savez(
        roa_file,
        angles=[-0.1, 0.0, 0.1],
        velocities=[0.0, 0.2, 0.4],
        converged=[[True, True, False]] * 3,
        metadata=json.dumps({"model_params": params}),
    )
    np.savez(
        policy_file,
        velocities=[0.0, 0.2, 0.4],
        min_steps=[0, 0, 1],
        best_alpha=[np.nan, np.nan, params["alpha_min"]],
    )
    return {"roa_file": roa_file, "policy_file": policy_file}


def test_policy_starts_balancing_between_upright_crossings(tables, monkeypatch):
    from assignment_2 import assignment_2_simulation as physics

    def advance(state, t, dt, dynamics):
        return np.array([0.05, 0.1])

    monkeypatch.setattr(physics, "integrator", advance)
    result = policy.simulate_policy(
        [0.05, 0.4], timestep=0.001, early_convergence=True, **tables
    )
    assert result["balance_start_time"] == 0.001
    assert result["time"][-1] == 0.001
    assert result["torque"][0] == 0.0
    assert result["torque"][-1] != 0.0


def test_rounded_zero_step_entry_does_not_trigger_balancing(tables):
    # 0.21 rounds to the zero-step entry at 0.2, but lies in a mixed RoA cell.
    with pytest.raises(ValueError, match="No alpha policy found"):
        policy.simulate_policy([0.0, 0.21], sim_time=0.001, **tables)


@pytest.mark.parametrize("angle", [0.0, 0.05])
def test_initial_safe_state_starts_balancing_immediately(tables, angle):
    result = policy.simulate_policy(
        [angle, 0.1], early_convergence=True, sim_time=0.01, **tables
    )
    assert result["balance_start_time"] == 0.0
    np.testing.assert_array_equal(result["time"], [0.0])


def test_physics_stops_at_roa_entry_before_advancing(tables):
    params = generate_params()
    params["alpha"] = params["alpha_min"]
    result = simulate(
        [0.0, 0.1], params, roa_file=tables["roa_file"], stop_at_balance=True
    )
    np.testing.assert_array_equal(result["time"], [0.0])
    np.testing.assert_array_equal(result["state"][:, 0], [0.0, 0.1])
    assert result["alpha"][0] == generate_params()["alpha"]
    assert result["torque"][0] != 0.0


def test_default_simulator_waits_for_roa_entry(tables, monkeypatch):
    from assignment_2 import assignment_2_simulation as physics

    def advance(state, t, dt, dynamics):
        return np.array([0.05, 0.1])

    monkeypatch.setattr(physics, "integrator", advance)
    params = generate_params()
    params["ankle_torque"] = 0.4  # A supplied torque cannot bypass the default gate.
    result = simulate(
        [0.05, 0.4],
        params,
        timestep=0.001,
        sim_time=0.002,
        roa_file=tables["roa_file"],
    )
    assert result["torque"][0] == 0.0
    assert result["torque"][1] != 0.0


def test_roa_construction_does_not_need_a_saved_roa(monkeypatch):
    from assignment_2 import assignment_2_roa as roa
    from assignment_2 import assignment_2_simulation as physics

    def no_saved_roa(path):
        raise AssertionError("RoA generation must not load an existing RoA")

    monkeypatch.setattr(physics, "load_standing_roa", no_saved_roa)
    result = roa.sweep([0.0], [0.0], generate_params(), timestep=0.001, sim_time=0.001)
    assert result[0, 0]
