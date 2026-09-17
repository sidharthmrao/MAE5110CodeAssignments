"""Shared plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np


def plot_poincare_return_map(
    current_velocity,
    next_velocity,
    *,
    title="Poincare Return Map",
    sample_label="Return-map samples",
    sample_size=10,
    sample_alpha=0.2,
):
    """Plot angular velocity at one impact against the following impact."""
    current_velocity = np.asarray(current_velocity)
    next_velocity = np.asarray(next_velocity)
    finite = np.isfinite(current_velocity) & np.isfinite(next_velocity)
    current_degrees = np.rad2deg(current_velocity[finite])
    next_degrees = np.rad2deg(next_velocity[finite])

    figure, axis = plt.subplots(figsize=(7, 7))
    if len(current_degrees) == 0:
        axis.text(
            0.5,
            0.5,
            "No return-map pairs were recorded.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        plot_limits = (-1.0, 1.0)
    else:
        lower = min(current_degrees.min(), next_degrees.min())
        upper = max(current_degrees.max(), next_degrees.max())
        span = upper - lower
        padding = 0.05 * span if span > 0 else max(1.0, 0.05 * abs(lower))
        plot_limits = (lower - padding, upper + padding)
        axis.scatter(
            current_degrees,
            next_degrees,
            s=sample_size,
            alpha=sample_alpha,
            edgecolors="none",
            rasterized=True,
            label=sample_label,
        )

    axis.plot(
        plot_limits,
        plot_limits,
        "k--",
        linewidth=1.25,
        label="Identity Line",
    )
    axis.set_xlim(plot_limits)
    axis.set_ylim(plot_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.set_xlabel(r"$\dot{\theta}_k^+$ [deg/s]")
    axis.set_ylabel(r"$\dot{\theta}_{k+1}^+$ [deg/s]")
    axis.set_title(title)
    return figure, axis
