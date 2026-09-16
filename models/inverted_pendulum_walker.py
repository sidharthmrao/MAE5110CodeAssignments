"""InvertedPendulumWalker starter model, with visualization provided.

Implement the model functions for Assignment 2. The visualizer works independently
of those functions; it draws a supplied state without advancing the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_params():
    return {
        # Constant environment parameters.
        "gravity": 9.81,  # m/s^2
        "incline_angle": 0.06,  # rad
        # Constant wheel parameters.
        "mass": 1.0,  # kg
        "spoke_length": 1.0,  # m
        # Control of angle between spokes.
        "alpha": np.pi / 7.0,  # rad
        "alpha_min": np.pi / 8.0,  # rad
        "alpha_max": np.pi / 7.0,  # rad
        # Control of contact spoke torque.
        "ankle_torque": 0,  # Nm
        "ankle_torque_min": -0.1 * 1.0 * 9.8 * 1.0,  # Nm
        "ankle_torque_max": 0.05 * 1.0 * 9.8 * 1.0,  # Nm
    }


def angle_collision_bounds(params):
    """Get angle collision bounds for spoke based on parameters."""
    incline = params["incline_angle"]
    alpha = params["alpha"]

    return incline - alpha, incline + alpha


def clipped_ankle_torque(params):
    """Clamp ankle torque to min/max limits."""
    return np.clip(
        params["ankle_torque"], params["ankle_torque_min"], params["ankle_torque_max"]
    )


def dynamics(t, state, params):
    m = params["mass"]
    g = params["gravity"]
    l = params["spoke_length"]

    T = clipped_ankle_torque(params)

    angle, angular_velocity = state

    # -- DYNAMICS --

    angular_acceleration = g / l * np.sin(angle) + T / (m * l * l)

    state_derivative = np.array([angular_velocity, angular_acceleration])

    return state_derivative


def forward_collision_guard(state, params):
    """Detect a candidate state beyond the forward collision bound."""
    _, forward_collision_angle = angle_collision_bounds(params)
    return state[0] > forward_collision_angle


def backward_collision_guard(state, params):
    """Detect a candidate state beyond the backward collision bound."""
    backward_collision_angle, _ = angle_collision_bounds(params)
    return state[0] < backward_collision_angle


def forward_collision_dynamics(state, params):
    """Apply a forward impact after its guard fires, without modifying the input."""
    state = np.array(state, dtype=float, copy=True)
    backward_collision_angle, _ = angle_collision_bounds(params)
    state[0] = backward_collision_angle
    state[1] *= np.cos(2.0 * params["alpha"])
    return state


def backward_collision_dynamics(state, params):
    """Apply a backward impact after its guard fires, without modifying the input."""
    state = np.array(state, dtype=float, copy=True)
    _, forward_collision_angle = angle_collision_bounds(params)
    state[0] = forward_collision_angle
    state[1] *= np.cos(2.0 * params["alpha"])
    return state


def calculate_energy(state, params):
    pass


def visualize(
    state,
    params,
    ax=None,
    *,
    show_swing=True,
    stance_position=(0.0, 0.0),
    view_limits=None,
):
    """Draw one walker pose and return a Matplotlib Axes.

    Parameters
    ----------
    state : array-like, shape (2,)
        [theta, angular_velocity], in radians and radians/second. Theta is
        measured clockwise from upward vertical; positive x points right.
    params : dict
        ``length`` is the leg length in meters. ``incline`` is the ground's
        downhill slope angle in radians (positive slopes descend to the right).
        ``angle_of_attack`` is HALF the angle between the stance and forward swing
        legs, in radians; it is needed only when show_swing=True.
        ``ankle_torque`` (optional, default 0) is displayed in N m, with positive
        torque acting in the positive theta direction. Other keys are ignored.
    ax : matplotlib.axes.Axes, optional
        Axes to clear and reuse. If omitted, create a figure. This function
        neither shows nor saves it: use plt.show() or ax.figure.savefig(...).
    show_swing : bool
        Draw a straight forward swing leg at the supplied angle_of_attack. Set False
        while the swing leg is held clear or while balancing. Swing motion is
        not part of the two-state model and is not inferred from theta.
    stance_position : pair of floats
        Current stance foot's (x, y) in meters, default (0, 0). The two-state
        model does not track translation; supply foot positions if desired.
        Ground passes through this point at the supplied incline.
    view_limits : (xmin, xmax, ymin, ymax), optional
        Fixed camera bounds in meters. By default the view follows the stance
        foot with bounds that fit both legs at any angle. Supply the same bounds
        each frame for a stationary world view.

    Notes
    -----
    Draws the supplied pose; contact events belong in the simulation.
    Reuse ax for frame sequences; use evenly spaced simulation times for playback
    at a fixed frame rate, and pass the parameters actually used at each frame.
    """
    state = np.asarray(state, dtype=float)
    foot = np.asarray(stance_position, dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("state must contain two finite values: [theta, velocity].")
    if foot.shape != (2,) or not np.all(np.isfinite(foot)):
        raise ValueError("stance_position must contain two finite values: [x, y].")
    length = float(params["spoke_length"])
    incline = float(params["incline_angle"])
    torque = float(params.get("ankle_torque", 0.0))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive.")
    if not np.isfinite(incline) or abs(incline) >= np.pi / 2:
        raise ValueError("incline must be finite and between -pi/2 and pi/2.")
    if not np.isfinite(torque):
        raise ValueError("ankle_torque must be finite.")
    angle_of_attack = None
    if show_swing:
        angle_of_attack = float(params["alpha"])
        if not np.isfinite(angle_of_attack):
            raise ValueError("angle_of_attack must be finite.")

    if view_limits is None:
        radius = 2.15 * length
        view_limits = (
            foot[0] - radius,
            foot[0] + radius,
            foot[1] - radius,
            foot[1] + radius,
        )
    limits = np.asarray(view_limits, dtype=float)
    if (
        limits.shape != (4,)
        or not np.all(np.isfinite(limits))
        or limits[0] >= limits[1]
        or limits[2] >= limits[3]
    ):
        raise ValueError(
            "view_limits must be (xmin, xmax, ymin, ymax) with increasing bounds."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.clear()
    theta, angular_velocity = state
    hub = foot + length * np.array([np.sin(theta), np.cos(theta)])

    ground_x = np.array(limits[:2])
    ground_y = foot[1] - np.tan(incline) * (ground_x - foot[0])
    ax.fill_between(ground_x, ground_y, limits[2], color="#eee7dc", zorder=0)
    ax.plot(ground_x, ground_y, color="#7b6651", linewidth=2, label="Ground")
    ax.plot(
        [foot[0], foot[0]],
        [foot[1], foot[1] + 1.25 * length],
        ":",
        color="0.7",
        linewidth=1,
        label="Vertical",
    )

    if show_swing and angle_of_attack is not None:
        swing_angle = theta - 2 * angle_of_attack
        swing_foot = hub - length * np.array([np.sin(swing_angle), np.cos(swing_angle)])
        swing_color = "#df8a25"
        ax.plot(
            [hub[0], swing_foot[0]],
            [hub[1], swing_foot[1]],
            "--",
            color=swing_color,
            linewidth=2.5,
            label="Swing leg",
            zorder=3,
        )
        ax.plot(
            *swing_foot,
            "o",
            color=swing_color,
            markersize=7,
            zorder=4,
            label="Swing foot",
        )

    stance_color = "#23699b"
    ax.plot(
        [foot[0], hub[0]],
        [foot[1], hub[1]],
        color=stance_color,
        linewidth=4,
        label="Stance leg",
        zorder=4,
    )
    ax.plot(*foot, "s", color="#333333", markersize=8, zorder=5, label="Stance foot")
    ax.plot(
        *hub,
        "o",
        color=stance_color,
        markeredgecolor="white",
        markersize=17,
        zorder=6,
        label="Hub",
    )
    ax.text(
        0.03,
        0.97,
        f"$\\theta$ = {theta:.3f} rad\n"
        f"$\\dot\\theta$ = {angular_velocity:.3f} rad/s\n"
        f"$\\tau$ = {torque:.3f} N m",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set(
        xlim=limits[:2],
        ylim=limits[2:],
        xlabel="x (m)",
        ylabel="y (m)",
        title="Inverted pendulum walker",
    )
    ax.set_aspect("equal", adjustable="box")
    return ax
