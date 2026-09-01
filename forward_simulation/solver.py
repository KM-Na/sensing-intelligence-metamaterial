"""Shared forward-dynamics solver.

This factors out the "run one forward simulation for a fixed design, driven
by a real force time series, and read out a handful of sensor blocks" logic
that is duplicated across the original co-optimization scripts (one local
copy of ``setup_dynamic_solver`` per structure type). The one difference
that matters for reproducing their recorded ``input_data`` is that those
scripts read out acceleration (x, y) and angular velocity at an explicit list
of sensor blocks (``measure_idx``) rather than either every block or a single
hardcoded block -- so that is made an explicit argument here instead of a
global closed over from argparse, as the originals did.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import diffrax
import jax.numpy as jnp
from jax import jacobian, vmap

from blockymetamaterials.energy import constrain_energy
from blockymetamaterials.geometry import Geometry, compute_inertia
from blockymetamaterials.utils import ControlParams
from cooptimization_real import (
    build_RHS,
    build_constrained_kinematics,
    build_loading,
    build_viscous_damping,
)


def setup_dynamic_solver_multi(
    geometry: Geometry,
    energy_fn: Callable,
    free_DOF_ids: jnp.ndarray,
    constrained_DOF_ids: jnp.ndarray,
    all_DOF_ids: jnp.ndarray,
    measure_idx: jnp.ndarray,
    loaded_DOF_ids: Optional[jnp.ndarray] = None,
    loading_fn: Optional[Callable] = None,
    constrained_block_DOF_pairs: jnp.ndarray = jnp.array([]),
    constrained_DOFs_fn: Callable = lambda t: 0,
    damped_DOF_ids: Optional[jnp.ndarray] = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Builds a dynamics solver that reads out (accel_x, accel_y, ang_vel)
    at each block in ``measure_idx``, matching the format used throughout
    the original training-dataset pickles (`n_measure_idx * 3` channels).
    """

    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs,
        constrained_DOFs_fn=constrained_DOFs_fn,
        free_DOF_ids=free_DOF_ids,
        constrained_DOF_ids=constrained_DOF_ids,
        all_DOF_ids=all_DOF_ids,
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    if loaded_DOF_ids is not None and loading_fn is not None:
        _loading_fn = build_loading(
            geometry=geometry,
            loaded_DOF_ids=loaded_DOF_ids,
            loading_fn=loading_fn,
            free_DOF_ids=free_DOF_ids,
            constrained_DOF_ids=constrained_DOF_ids,
            all_DOF_ids=all_DOF_ids,
        )
    else:
        def _loading_fn(state, t, loading_params):
            return 0

    if damped_DOF_ids is not None:
        damping_fn = build_viscous_damping(
            geometry=geometry,
            damped_DOF_ids=damped_DOF_ids,
            free_DOF_ids=free_DOF_ids,
            constrained_DOF_ids=constrained_DOF_ids,
            all_DOF_ids=all_DOF_ids,
        )
    else:
        def damping_fn(state, t, damping):
            return 0

    def loading_fn_total(state, t, loading_params, damping):
        return _loading_fn(state, t, loading_params) + damping_fn(state, t, damping)

    rhs = build_RHS(energy_fn=constrained_energy, loading_fn=loading_fn_total)

    jac_kinematics = jacobian(kinematics, argnums=(0, 1))

    def velocity_fn(free_DOFs, free_DOFs_dot, t, constraint_params):
        du_dfree, du_dt = jac_kinematics(free_DOFs, t, constraint_params)
        return du_dfree @ free_DOFs_dot + du_dt

    jac_velocity = jacobian(velocity_fn, argnums=(0, 1, 2))

    def acceleration_fn(free_DOFs_total, t, control_params, inertia):
        free_DOFs = free_DOFs_total[0, :]
        free_DOFs_dot = free_DOFs_total[1, :]
        dvdx, dvdx_dot, dvdt = jac_velocity(free_DOFs, free_DOFs_dot, t, control_params.constraint_params)
        args = (control_params, inertia)
        acc_freeDOFs = rhs(t, free_DOFs_total, args)[1, :]
        return dvdx @ free_DOFs_dot + dvdx_dot @ acc_freeDOFs + dvdt

    velocity_history_fn = vmap(velocity_fn, in_axes=(0, 0, 0, None))
    acceleration_history_fn = vmap(acceleration_fn, in_axes=(0, 0, None, None))

    measure_idx = jnp.asarray(measure_idx)
    n_measure = measure_idx.shape[0]

    def solve_dynamics(state0: jnp.ndarray, timepoints: jnp.ndarray, control_params: ControlParams):
        _state0 = state0.reshape((2, geometry.n_blocks * 3))[:, free_DOF_ids]

        if control_params.mechanical_params.inertia is None:
            _inertia = compute_inertia(
                vertices=control_params.geometrical_params.centroid_node_vectors,
                density=control_params.mechanical_params.density,
            ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
        else:
            _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

        ode_solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=timepoints)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            ode_solver,
            timepoints[0],
            timepoints[-1],
            dt0=None,
            y0=_state0,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=saveat,
            args=(control_params, _inertia),
            max_steps=1_000_000,
        )
        free_DOFs_solution = solution.ys

        velocity_history = velocity_history_fn(
            free_DOFs_solution[:, 0, :],
            free_DOFs_solution[:, 1, :],
            timepoints,
            control_params.constraint_params,
        )
        acceleration_history = acceleration_history_fn(
            free_DOFs_solution[:, 0:2, :],
            timepoints,
            control_params,
            _inertia,
        )

        out = jnp.zeros((len(timepoints), n_measure, 3))
        out = out.at[:, :, :2].set(acceleration_history[:, measure_idx, 0:2])
        out = out.at[:, :, -1].set(velocity_history[:, measure_idx, -1])
        return out.reshape((len(timepoints), -1))

    return solve_dynamics


def build_forward_problem(structure, output_data_transform: Optional[Callable] = None) -> Callable:
    """Given a ``StructureSetup`` (see ``forward_simulation.structures``),
    returns ``forward_problem(design, output_data)`` -- driven by a real
    force time series, vmap-able over a leading sample axis of
    ``output_data`` via ``jax.vmap(forward_problem, in_axes=(None, 0))``.
    """

    timepoints = jnp.linspace(0.0, structure.simulation_time, structure.n_timepoints)
    transform = output_data_transform or structure.output_data_transform or (lambda x: x)

    @eqx.filter_jit
    def forward_problem(design, output_data):
        horizontal_shifts, vertical_shifts = design
        force = transform(output_data)

        def loading_fn(state, t, **kwargs):
            return vmap(jnp.interp, in_axes=(None, None, 1))(t, timepoints, force)

        solve_dynamics = setup_dynamic_solver_multi(
            geometry=structure.geometry,
            energy_fn=structure.potential_energy,
            free_DOF_ids=structure.free_DOF_ids,
            constrained_DOF_ids=structure.constrained_DOF_ids,
            all_DOF_ids=structure.all_DOF_ids,
            measure_idx=structure.measure_idx,
            loaded_DOF_ids=structure.loaded_DOF_ids,
            loading_fn=loading_fn,
            constrained_block_DOF_pairs=structure.constrained_block_DOF_pairs,
            constrained_DOFs_fn=structure.constrained_DOFs_fn,
            damped_DOF_ids=structure.damped_DOF_ids,
            atol=structure.atol,
            rtol=structure.rtol,
        )

        control_params = structure.build_control_params(horizontal_shifts, vertical_shifts)
        state0 = 1e-10 * jnp.ones((2, structure.geometry.n_blocks, 3))
        return solve_dynamics(state0=state0, timepoints=timepoints, control_params=control_params)

    return forward_problem
