import os,sys
sys.path.append(os.path.dirname(os.getcwd()))

from pathlib import Path
from typing import NamedTuple, Any, Optional, List, Union, Tuple, Dict
import dataclasses
from dataclasses import dataclass
from jax import hessian, jacobian, jit, vmap, grad, random
from functools import partial

import numpy as np
import time
import jax.numpy as jnp
import jax.nn as jnn
import jax
import pickle
jax.config.update("jax_enable_x64", True)  # enable float64 type

from blockymetamaterials.utils import SolutionType, SolutionData, ControlParams, GeometricalParams, MechanicalParams, LigamentParams, ContactParams
from blockymetamaterials.geometry import Geometry, QuadGeometry, compute_inertia, compute_edge_angles, compute_edge_lengths, DOFsInfo
from blockymetamaterials.energy import build_strain_energy, kinetic_energy, ligament_energy, ligament_energy_linearized, build_contact_energy, combine_block_energies, compute_ligament_strains_history, constrain_energy
from typing import Any, Literal, Optional, List, Union, Tuple, Dict
import dataclasses
from dataclasses import dataclass
from itertools import combinations

import diffrax
from typing import Callable, Optional, Union
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from jax_md.quantity import force
import equinox as eqx
from tqdm import trange, tqdm



def build_viscous_damping(
        geometry: Geometry,
        damped_DOF_ids: jnp.ndarray,
        free_DOF_ids: jnp.ndarray,
        constrained_DOF_ids: jnp.ndarray,
        all_DOF_ids: jnp.ndarray):

    # This is to ensure correct shape of loading vector when damping is either a scalar or an array of shape (n_damped_blocks, 3)
    reshaping_array = jnp.ones((damped_DOF_ids.shape[0]//3, 3))

    def loading_fn(state, t, damping: jnp.ndarray):
        _, velocity = state
        loading_vector = jnp.zeros((len(all_DOF_ids),))
        loading_vector = loading_vector.at[damped_DOF_ids].set(
            (damping * reshaping_array).reshape(damped_DOF_ids.shape)
        )
        loading_vector = loading_vector[free_DOF_ids]

        return -loading_vector * velocity

    return loading_fn


def build_loading(
        geometry: Geometry,
        loaded_DOF_ids: jnp.ndarray,
        loading_fn: Callable,
        free_DOF_ids: jnp.ndarray,
        constrained_DOF_ids: jnp.ndarray,
        all_DOF_ids: jnp.ndarray):

    def global_loading_fn(state, t, loading_params: Dict):

        loading_vector = jnp.zeros((len(all_DOF_ids),))
        
        loading_vector = loading_vector.at[loaded_DOF_ids].set(
            loading_fn(state, t, **loading_params)
        )
        loading_vector = loading_vector[free_DOF_ids]  # Reduce loading vector to the free DOFs

        return loading_vector

    return global_loading_fn


def build_constrained_kinematics(
        geometry: Geometry,
        constrained_block_DOF_pairs: jnp.ndarray,
        free_DOF_ids: jnp.ndarray,
        constrained_DOF_ids: jnp.ndarray,
        all_DOF_ids: jnp.ndarray,
        constrained_DOFs_fn: Callable = lambda t,
        **kwargs: 0):

    def constrained_kinematics(free_DOFs: jnp.ndarray, t, constraint_params: Dict = dict()):

        all_DOFs = jnp.zeros((len(all_DOF_ids),))
        # Assign imposed displacements along the constrained DOFs
        if len(constrained_DOF_ids) != 0:
            all_DOFs = all_DOFs.at[constrained_DOF_ids].set(
                constrained_DOFs_fn(t, **constraint_params)
            )

        # Simply assign the free_DOFs along the free DOFs (this acts as the identity operator)
        all_DOFs = all_DOFs.at[free_DOF_ids].set(
            free_DOFs
        )
        return all_DOFs.reshape((geometry.n_blocks, 3))

    return constrained_kinematics

def build_RHS(energy_fn: Callable, loading_fn: Callable):

    potential_force = force(energy_fn)

    @jit
    def rhs(t, state: jnp.ndarray, args):
        control_params, inertia = args
        loading_params = control_params.loading_params
        damping = control_params.mechanical_params.damping
        displacement, velocity = state
        dxdt = jnp.array([
            velocity,
            (potential_force(displacement, t, control_params) + loading_fn(state, t, loading_params, damping)) / inertia
        ])
        return dxdt

    return rhs


def setup_dynamic_solver_all(
        geometry: Geometry,
        energy_fn: Callable,
        free_DOF_ids: jnp.ndarray,
        constrained_DOF_ids: jnp.ndarray,
        all_DOF_ids: jnp.ndarray,
        loaded_DOF_ids: Optional[jnp.ndarray] = None,
        loading_fn: Optional[Callable] = None,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([]),
        constrained_DOFs_fn: Callable = lambda t: 0,
        damped_DOF_ids: Optional[jnp.ndarray] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8):

    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs,
        constrained_DOFs_fn=constrained_DOFs_fn,
        free_DOF_ids=free_DOF_ids,
        constrained_DOF_ids=constrained_DOF_ids,
        all_DOF_ids=all_DOF_ids
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Canonicalize loading function
    if loaded_DOF_ids is not None and loading_fn is not None:
        _loading_fn = build_loading(
            geometry=geometry,
            loaded_DOF_ids=loaded_DOF_ids,
            loading_fn=loading_fn,
            free_DOF_ids=free_DOF_ids,
            constrained_DOF_ids=constrained_DOF_ids,
            all_DOF_ids=all_DOF_ids
        )
    else:
        def _loading_fn(state, t, loading_params): return 0

    # Canonicalize damping
    if damped_DOF_ids is not None:
        damping_fn = build_viscous_damping(
            geometry=geometry,
            damped_DOF_ids=damped_DOF_ids,
            free_DOF_ids=free_DOF_ids,
            constrained_DOF_ids=constrained_DOF_ids,
            all_DOF_ids=all_DOF_ids
        )
    else:
        def damping_fn(state, t, damping): return 0

    # Combine all loading functions
    def loading_fn_total(state, t, loading_params, damping):
        return _loading_fn(state, t, loading_params) + damping_fn(state, t, damping)

    rhs = build_RHS(energy_fn=constrained_energy, loading_fn=loading_fn_total)
    
    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    # free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Utility functions to reconstruct the full state array from the solution of the free DOFs
    displacement_history_fn = vmap(kinematics, in_axes=(0, 0, None)) #-> TRY OBTAIN DISPLACEMENTS OF ALL BLOCKS (N,x) ,
    jac_kinematics = jacobian(kinematics, argnums=(0, 1))
    
    def velocity_fn(free_DOFs, free_DOFs_dot, t, constraint_params):
        du_dfree, du_dt = jac_kinematics(free_DOFs, t, constraint_params)
        return du_dfree @ free_DOFs_dot + du_dt

    jac_velocity = jacobian(velocity_fn, argnums=(0, 1, 2))
    def acceleration_fn(free_DOFs_total, t, control_params, inertia):
        free_DOFs = free_DOFs_total[0,:]
        free_DOFs_dot = free_DOFs_total[1,:]
    
        dvdx, dvdx_dot, dvdt = jac_velocity(free_DOFs, free_DOFs_dot, t, control_params.constraint_params)
        args = (control_params, inertia)
        acc_freeDOFs = rhs(t, free_DOFs_total, args)[1,:]
        return dvdx @ free_DOFs_dot + dvdx_dot @ acc_freeDOFs + dvdt

    velocity_history_fn = vmap(velocity_fn, in_axes=(0, 0, 0, None))
    acceleration_history_fn = vmap(acceleration_fn, in_axes=(0, 0, None, None))

    def solve_dynamics(state0: jnp.ndarray, timepoints: jnp.ndarray, control_params: ControlParams):
        
        # Reduce state0 and inertia to the free DOFs
        _state0 = state0.reshape((2, geometry.n_blocks * 3))[:, free_DOF_ids]
        
        if control_params.mechanical_params.inertia is None:
            _inertia = compute_inertia(
                vertices=control_params.geometrical_params.centroid_node_vectors,
                density=control_params.mechanical_params.density
            ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
        else:
            _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

        # Solve ODE
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=timepoints)

        solution = diffrax.diffeqsolve(diffrax.ODETerm(rhs), solver,
                                                timepoints[0], timepoints[-1], dt0=None, y0=_state0, 
                                                stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                                                saveat=saveat,
                                                args=(control_params, _inertia),
                                                max_steps=1000000) 

        free_DOFs_solution = solution.ys

        # Reshape solution to global state.
        displacement_history = displacement_history_fn(
            free_DOFs_solution[:, 0, :],
            timepoints,
            control_params.constraint_params
        )
        
        velocity_history = velocity_history_fn(
            free_DOFs_solution[:, 0, :],
            free_DOFs_solution[:, 1, :],
            timepoints,
            control_params.constraint_params
        )

        acceleration_history = acceleration_history_fn(
            free_DOFs_solution[:, 0:2, :],
            timepoints,
            control_params,
            _inertia
        )

        solution = jnp.zeros((len(timepoints), 3, geometry.n_blocks, 3))
        solution = solution.at[:, 0, :, :].set(displacement_history)
        solution = solution.at[:, 1, :, :].set(velocity_history)
        # Compute acceleration history
        solution = solution.at[:, 2, :, :].set(acceleration_history)
        
        return solution

    return solve_dynamics

def setup_dynamic_solver(
        geometry: Geometry,
        energy_fn: Callable,
        free_DOF_ids: jnp.ndarray,
        constrained_DOF_ids: jnp.ndarray,
        all_DOF_ids: jnp.ndarray,
        loaded_DOF_ids: Optional[jnp.ndarray] = None,
        loading_fn: Optional[Callable] = None,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([]),
        constrained_DOFs_fn: Callable = lambda t: 0,
        damped_DOF_ids: Optional[jnp.ndarray] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8):

    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs,
        constrained_DOFs_fn=constrained_DOFs_fn,
        free_DOF_ids=free_DOF_ids,
        constrained_DOF_ids=constrained_DOF_ids,
        all_DOF_ids=all_DOF_ids
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Canonicalize loading function
    if loaded_DOF_ids is not None and loading_fn is not None:
        _loading_fn = build_loading(
            geometry=geometry,
            loaded_DOF_ids=loaded_DOF_ids,
            loading_fn=loading_fn,
            free_DOF_ids=free_DOF_ids,
            constrained_DOF_ids=constrained_DOF_ids,
            all_DOF_ids=all_DOF_ids
        )
    else:
        def _loading_fn(state, t, loading_params): return 0

    # Canonicalize damping
    if damped_DOF_ids is not None:
        damping_fn = build_viscous_damping(
            geometry=geometry,
            damped_DOF_ids=damped_DOF_ids,
            free_DOF_ids=free_DOF_ids,
            constrained_DOF_ids=constrained_DOF_ids,
            all_DOF_ids=all_DOF_ids
        )
    else:
        def damping_fn(state, t, damping): return 0

    # Combine all loading functions
    def loading_fn_total(state, t, loading_params, damping):
        return _loading_fn(state, t, loading_params) + damping_fn(state, t, damping)

    rhs = build_RHS(energy_fn=constrained_energy, loading_fn=loading_fn_total)

    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    # free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Utility functions to reconstruct the full state array from the solution of the free DOFs
    displacement_history_fn = vmap(kinematics, in_axes=(0, 0, None)) #-> TRY OBTAIN DISPLACEMENTS OF ALL BLOCKS (N,x) ,
    jac_kinematics = jacobian(kinematics, argnums=(0, 1))
    
    def velocity_fn(free_DOFs, free_DOFs_dot, t, constraint_params):
        du_dfree, du_dt = jac_kinematics(free_DOFs, t, constraint_params)
        return du_dfree @ free_DOFs_dot + du_dt

    jac_velocity = jacobian(velocity_fn, argnums=(0, 1, 2))
    def acceleration_fn(free_DOFs_total, t, control_params, inertia):
        free_DOFs = free_DOFs_total[0,:]
        free_DOFs_dot = free_DOFs_total[1,:]
    
        dvdx, dvdx_dot, dvdt = jac_velocity(free_DOFs, free_DOFs_dot, t, control_params.constraint_params)
        acc_freeDOFs = rhs(free_DOFs_total, t, control_params, inertia)[1,:]
        return dvdx @ free_DOFs_dot + dvdx_dot @ acc_freeDOFs + dvdt

    velocity_history_fn = vmap(velocity_fn, in_axes=(0, 0, 0, None))
    acceleration_history_fn = vmap(acceleration_fn, in_axes=(0, 0, None, None))

    def solve_dynamics(state0: jnp.ndarray, timepoints: jnp.ndarray, control_params: ControlParams):
        
        # Reduce state0 and inertia to the free DOFs
        _state0 = state0.reshape((2, geometry.n_blocks * 3))[:, free_DOF_ids]
        
        if control_params.mechanical_params.inertia is None:
            _inertia = compute_inertia(
                vertices=control_params.geometrical_params.centroid_node_vectors,
                density=control_params.mechanical_params.density
            ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
        else:
            _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

        # Solve ODE
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=timepoints)

        solution = diffrax.diffeqsolve(diffrax.ODETerm(rhs), solver,
                                                timepoints[0], timepoints[-1], dt0=None, y0=_state0, 
                                                stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                                                saveat=saveat,
                                                args=(control_params, _inertia),
                                                max_steps=1000000) 

        free_DOFs_solution = solution.ys

        # Reshape solution to global state.
        displacement_history = displacement_history_fn(
            free_DOFs_solution[:, 0, :],
            timepoints,
            control_params.constraint_params
        )
        
        velocity_history = velocity_history_fn(
            free_DOFs_solution[:, 0, :],
            free_DOFs_solution[:, 1, :],
            timepoints,
            control_params.constraint_params
        )

        # solution = jnp.zeros((len(timepoints), 3)) # (len(timepoints), 2, geometry.n_blocks, 3) -> (len(timepoints), 3, geometry.n_blocks, 3)
        # solution = solution.at[:, 0, :].set(displacement_history[:,12,:])
        # solution = solution.at[:, 1, :].set(velocity_history[:,12,:])
        center_idx = 48 # if n1_blocks = 5, n2_blocks = 5, then 12
        solution = velocity_history[:,center_idx,:] # (len(timepoints), 3)
        
        return solution

    return solve_dynamics
