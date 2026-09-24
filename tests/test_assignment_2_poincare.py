import numpy as np
import pytest

from assignment_2 import assignment_2_simulation as physics
from assignment_2.assignment_2_poincare import next_return
from assignment_2.assignment_2_simulation import simulate
from models.inverted_pendulum_walker import generate_params


@pytest.mark.parametrize("alpha_key", ["alpha_min", "alpha_max"])
def test_passive_return_outside_roa_matches_shared_physics_simulator(alpha_key):
    params = generate_params()
    params["alpha"] = params[alpha_key]
    roa = {
        "angles": np.array([-0.1, 0.1]),
        "velocities": np.array([0.0, 0.2]),
        "converged": np.zeros((2, 2), dtype=bool),
        "params": params,
    }
    expected = next_return(3.0, params["alpha"], params, roa=roa)
    result = simulate(
        [0.0, 3.0],
        params,
        timestep=0.001,
        sim_time=2.0,
        max_collisions=None,
        controller=lambda t, state, params: 0.0,
    )
    angle, velocity = result["state"]
    first_impact = np.flatnonzero(np.diff(angle) < -0.5)[0] + 1
    crossings = np.flatnonzero((angle[:-1] < 0) & (angle[1:] >= 0))
    index = crossings[crossings >= first_impact][0]
    fraction = -angle[index] / (angle[index + 1] - angle[index])
    actual = velocity[index] + fraction * (velocity[index + 1] - velocity[index])
    assert np.all(result["torque"] == 0.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert (
        params["ankle_torque"] == 0
    )  # Neither simulator mutates the caller's parameters.


def test_return_map_enables_torque_only_after_roa_entry(monkeypatch):
    params = generate_params()
    roa = {
        "angles": np.array([-0.1, 0.1]),
        "velocities": np.array([0.0, 0.2]),
        "converged": np.ones((2, 2), dtype=bool),
        "params": params,
    }
    torques = []
    alphas = []

    def advance(state, t, dt, dynamics):
        step_params = dynamics.keywords["params"]
        torques.append(step_params["ankle_torque"])
        alphas.append(step_params["alpha"])
        return np.array([0.01, 0.1])

    monkeypatch.setattr(physics, "integrator", advance)
    next_return(3.0, params["alpha_min"], params, max_time=0.002, roa=roa)
    assert torques[0] == 0.0
    assert torques[1] != 0.0
    assert alphas == [params["alpha_min"], params["alpha"]]


@pytest.mark.parametrize("next_angle", [-1.0, 1.0])
def test_collision_reset_is_not_an_upright_return(monkeypatch, next_angle):
    params = generate_params()
    monkeypatch.setattr(
        physics,
        "integrator",
        lambda state, t, dt, dynamics: np.array([next_angle, 1.0]),
    )
    result = simulate(
        [0.0, 1.0],
        params,
        controller=lambda t, state, p: 0.0,
        timestep=0.001,
        sim_time=0.001,
        poincare_return=True,
    )
    assert np.isnan(result)


def test_return_mode_respects_partial_final_timestep(monkeypatch):
    steps = []

    def advance(state, t, dt, dynamics):
        steps.append(dt)
        return state

    monkeypatch.setattr(physics, "integrator", advance)
    value = simulate(
        [0.0, 1.0],
        controller=lambda t, state, p: 0.0,
        timestep=0.001,
        sim_time=0.0015,
        poincare_return=True,
    )
    assert np.isnan(value)
    np.testing.assert_allclose(steps, [0.001, 0.0005])
