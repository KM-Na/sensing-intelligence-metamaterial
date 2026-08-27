"""
Compared with the `dynamics` module, this module allows different shape of the blocks.
In previous version, the blocks were assumed to have same number of nodes.
"""

from typing import Callable, Optional, Union

import jax.numpy as jnp
import scipy
from jax import hessian, jacobian, jit, vmap, grad
from jax.experimental.ode import odeint
from jax_md.quantity import force
import diffrax

from blockymetamaterials.energy import constrain_energy
from blockymetamaterials.geometry import DOFsInfo, Geometry, compute_inertia
from blockymetamaterials.kinematics import build_constrained_kinematics
from blockymetamaterials.loading import build_loading, build_viscous_damping
from blockymetamaterials.utils import ControlParams, is_scalar


def build_RHS(energy_fn: Callable, loading_fn: Callable):
    """Defines the RHS of dynamic problem dydt = RHS for a system governed by the potential energy functional `energy_fn`.

    Args:
        energy_fn (Callable): potential energy functional.
        loading_fn (Callable): function including any external forces.

    Returns:
        Callable: RHS function of dynamic problem dydt = RHS.
    """

    potential_force = force(energy_fn)

    # def rhs(state: jnp.ndarray, t, control_params: ControlParams, inertia: jnp.ndarray):
    @jit
    def rhs(t, state: jnp.ndarray, args):
        # args = control_params: ControlParams, inertia: jnp.ndarray
        control_params, inertia = args
        """Computes RHS of dynamic problem dydt = RHS.

        Args:
            state (jnp.ndarray): array of shape (2, n_free_DOFs) where the first axis represents displacement (first position) and velocity (second position).
            t (float): time value to be passed to time dependent loadings.
            control_params (ControlParams): control parameters. See `utils.ControlParams` for details.
            inertia (jnp.ndarray): array of shape (n_free_DOFs) collecting the inertia of the blocks.

        Returns:
            jnp.ndarray: array representing the RHS of dynamic problem dydt = RHS.
        """

        loading_params = control_params.loading_params
        damping = control_params.mechanical_params.damping

        displacement, velocity = state

        return jnp.array([
            velocity,
            (potential_force(displacement, t, control_params) + loading_fn(state, t, loading_params, damping)) / inertia
        ])

    return rhs

# Return the full state of the system (displacement, velocity, acceleration)
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
        # solver = diffrax.Kvaerno5()
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

# Return the acceleration (x,y) and angular velocity of the center block 
# (we assume that we have only one IMU sensor on the center block but you can modify it to have multiple IMU sensors)
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
        # print(free_DOFs_total.shape)
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
        # solver = diffrax.Kvaerno5()
        saveat = diffrax.SaveAt(ts=timepoints)

        solution = diffrax.diffeqsolve(diffrax.ODETerm(rhs), solver,
                                                timepoints[0], timepoints[-1], dt0=None, y0=_state0, 
                                                stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                                                saveat=saveat,
                                                args=(control_params, _inertia),
                                                max_steps=1000000) 

        free_DOFs_solution = solution.ys

        # Reshape solution to global state.
        # displacement_history = displacement_history_fn(
        #     free_DOFs_solution[:, 0, :],
        #     timepoints,
        #     control_params.constraint_params
        # )
        
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

        solution = jnp.zeros((len(timepoints), 3)) # (len(timepoints), 2, geometry.n_blocks, 3) -> (len(timepoints), 3, geometry.n_blocks, 3)
        center_idx = geometry.n1_blocks * (geometry.n2_blocks//2 - 1) + geometry.n1_blocks // 2
        print( "center_idx", center_idx )

        solution = solution.at[:,:2].set( acceleration_history[:,center_idx,0:2] )
        solution = solution.at[:,-1].set( velocity_history[:,center_idx,-1] )
        
        return solution

    return solve_dynamics


def linear_mode_analysis(
        displacement: jnp.ndarray,
        geometry: Geometry,
        energy_fn: Callable,
        control_params: ControlParams,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([]),):
    """Computes eigenvalues and eigenmodes of K @ q = w^2 M @ q.

    Args:
        displacement (jnp.ndarray): configuration around which linearization is performed.
        geometry (Geometry): Geometry of the structure.
        energy_fn (Callable): Potential energy functional.
        centroid_node_vectors (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes.
        inertia (Union[jnp.ndarray, float]): either a scalar or an array of shape (n_blocks, 3) collecting the inertia of the blocks.
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        tuple: eigenvalues and eigenmodes. The eigenmodes are returned as an array of shape (n_modes, n_blocks, 3)
    """

    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Retrieve free DOFs from constraints info
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Reduce displacement and inertia to the free DOFs
    _displacement = displacement.reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    if control_params.mechanical_params.inertia is None:
        _inertia = compute_inertia(
            vertices=control_params.geometrical_params.centroid_node_vectors,
            density=control_params.mechanical_params.density
        ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    else:
        _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

    stiffness_matrix = hessian(constrained_energy)(_displacement, 0, control_params)
    # eigenvectors given by scipy are organized column-wise
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        stiffness_matrix,
        jnp.diag(_inertia),
    )  # jnp.linalg.eigh does not currently implement generalized eigenvalue problems
    # Normalize and transpose eigenvectors
    eigenvectors = vmap(lambda v: v / jnp.linalg.norm(v))(eigenvectors.T)

    # Reshape eigenvectors to global state. all_DOFs_modes are organized row-wise.
    all_DOFs_modes = jnp.zeros((len(free_DOF_ids), len(all_DOF_ids)))
    all_DOFs_modes = all_DOFs_modes.at[:, free_DOF_ids].set(
        eigenvectors
    )

    # NOTE: return eigenfrequency squared and modes
    return jnp.array(eigenvalues), all_DOFs_modes.reshape((len(free_DOF_ids), geometry.n_blocks, 3))
