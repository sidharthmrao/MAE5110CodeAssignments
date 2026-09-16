# Assignment 2

Run from the repository root:

```bash
uv run python -m assignment_2.assignment_2
uv run python -m assignment_2.assignment_2_roa
```

The first command uses `simulate()` and saves `output/walker.gif` in this folder, without opening a window. Importing the module does not run the simulation. `simulate(initial_state, params=None, timestep=1e-4, sim_time=5.0, max_collisions=1)` returns a dictionary containing time, state, torque, collision count, parameters, and timestep. Pass the timing and collision options as keyword arguments. Each call copies the model parameters.

The RoA script tests collision-free behavior using the same controller and collision handling. Initial angles range from half the backward collision angle to half the forward collision angle, measured from upright. Initial angular velocities span −1 to +1 rad/s. Defaults are an 81×121 grid and a five-second horizon, terminating at the first collision.

A trial counts as converged if it completes the five-second horizon without a collision. There are no angle/velocity settling tolerances or dwell-time requirements. This measures collision-free behavior over the chosen horizon, rather than asymptotic convergence to upright. The sweep uses a 0.005-second timestep; refine the timestep and grid near the boundary before drawing conclusions.

Configure the constants at the top of `assignment_2_roa.py` to change grid resolution, angle fraction, velocity bounds, timestep, duration, or output directory. For a coarse preview, set `ANGLE_STEPS = 9` and `VELOCITY_STEPS = 11`, then run the same command above.

Results are saved in `output/roa/roa_map.png` and `output/roa/roa_results.npz` relative to this folder. The NPZ contains the grid, classification, collision counts, and run settings. Re-running overwrites these outputs; change `ROA_OUTPUT_DIR` to retain separate experiments.
