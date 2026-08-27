import os
import sys
import datetime
import argparse
import pickle
import time
import dataclasses
from dataclasses import dataclass
from itertools import combinations
import datetime
from pathlib import Path
# Create parser
parser = argparse.ArgumentParser(description="Example script that takes arguments.")
# Add arguments
# outer num iter, inner num iter
parser.add_argument('--inneriter', type=int, default=1000, help='Number of iteration for inner loop')
parser.add_argument('--outeriter', type=int, default=100, help='Number of iteration for outer loop')
parser.add_argument('--outerlr', type=float, default=0.1, help='Learning rate for outer loop')
parser.add_argument('--innerlr', type=float, default=0.01, help='Learning rate for inner loop')
parser.add_argument(
    '--measure-idx', '--measure_idx', dest='measure_idx', type=int, nargs='+',
    default=[48], metavar='IDX',
    help='One or more measurement indices (e.g. --measure-idx 25 29 41).'
)
parser.add_argument('--gpu', type=int, default=0, help='GPU number')
args = parser.parse_args()
measure_idx_tag = "_".join(str(idx) for idx in args.measure_idx)

# print(args.innerlr)
t = datetime.date.today()
time_string = t.strftime('%m%d%Y')
print(time_string)

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# JAX config
from jax import config
config.update("jax_enable_x64", True)

# JAX and related imports
import jax
import jax.numpy as jnp
from jax import random, vmap, hessian, jacobian, jit, grad, value_and_grad
from jax.tree_util import Partial, tree_map
from jax_md.quantity import force
# Equinox
import equinox as eqx
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from jaxopt import LBFGS, GradientDescent
from jaxopt import linear_solve
from jaxopt import OptaxSolver
from jaxopt import tree_util

# Diffrax
import diffrax
from typing import Callable, Optional, Union, Any, Sequence, Literal, List, Tuple, Dict

# tqdm for progress bars
from tqdm.auto import tqdm

# Local project imports
from blockymetamaterials.utils import (
    SolutionType, SolutionData, ControlParams, GeometricalParams, MechanicalParams, LigamentParams, ContactParams
)
from blockymetamaterials.geometry import (
    Geometry, QuadGeometry, compute_inertia, compute_edge_angles, compute_edge_lengths, DOFsInfo,
    QuadGeometry_InputSource, polygon_area, polygon_polar_moment
)
from blockymetamaterials.energy import (
    build_strain_energy, kinetic_energy, ligament_energy, ligament_energy_linearized,
    build_contact_energy, combine_block_energies, compute_ligament_strains_history, constrain_energy
)

from blockymetamaterials.kinematics import build_constrained_kinematics
from blockymetamaterials.loading import build_loading, build_viscous_damping
from blockymetamaterials.dynamics import build_RHS


from blockymetamaterials.utils import SolutionType, SolutionData, ControlParams, GeometricalParams, MechanicalParams, LigamentParams, ContactParams
# from blockymetamaterials.geometry import Geometry, QuadGeometry, compute_inertia, compute_edge_angles, compute_edge_lengths, DOFsInfo, QuadGeometry_Circle_InputSource, polygon_area, polygon_polar_moment
from blockymetamaterials.geometry import Geometry, QuadGeometry, compute_inertia, compute_edge_angles, compute_edge_lengths, DOFsInfo, QuadGeometry_Circle_InputSource, polygon_area, polygon_polar_moment
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
        saveat = diffrax.SaveAt(ts=timepoints)

        solution = diffrax.diffeqsolve(diffrax.ODETerm(rhs), solver,
                                                timepoints[0], timepoints[-1], dt0=None, y0=_state0, 
                                                stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                                                saveat=saveat,
                                                args=(control_params, _inertia),
                                                max_steps=1000000) 

        free_DOFs_solution = solution.ys
        
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

        measure_idx = jnp.array(args.measure_idx)
        n_measure = measure_idx.shape[0]
        solution = jnp.zeros((len(timepoints), n_measure, 3))

        solution = solution.at[:,:,:2].set( acceleration_history[:,measure_idx,0:2] )
        solution = solution.at[:,:,-1].set( velocity_history[:,measure_idx,-1] )
        solution = solution.reshape((len(timepoints), -1))
        print( "solution shape : ", solution.shape )
        return solution

    return solve_dynamics

class minmax_scaler():
  def __init__(self, input_data, output_data):
    # assume that 
    self.data_max = jnp.max(input_data, axis=(0, 1), keepdims=True)
    self.data_min = jnp.min(input_data, axis=(0, 1), keepdims=True)
    print(self.data_max.shape, self.data_min.shape)

    self.output_data_max = jnp.max(output_data, axis=(0, 1), keepdims=True)
    self.output_data_min = jnp.min(output_data, axis=(0, 1), keepdims=True)
    print(self.output_data_max.shape, self.output_data_min.shape)

  def transform(self, input_data, output_data):
    data_normalized = 2.0 * ( input_data - self.data_min ) / ( self.data_max - self.data_min ) - 1.0  # Add a small value to avoid division by zero
    output_normalized = (output_data - self.output_data_min) / (self.output_data_max - self.output_data_min)
    return data_normalized, output_normalized
  
  def inverse_transform(self, data_normalized, output_normalized):
    data_original = (data_normalized + 1.0) * (self.data_max - self.data_min) / 2.0 + self.data_min
    output_original = output_normalized * (self.output_data_max - self.output_data_min) + self.output_data_min
    return data_original, output_original

  def information(self):
    print("data min : ", self.data_min)
    print("data max : ", self.data_max)
    print("test data min : ", self.output_data_min)
    print("test data max : ", self.output_data_max)
    return self.data_min, self.data_max, self.output_data_min, self.output_data_max

class TimeSeriesCNN(nn.Module):
    features: int = 64
    output_features: int = 2

    @nn.compact
    def __call__(self, x):
        # Input shape: (batch_size, time_steps, 2)
        x = nn.Conv(self.features, kernel_size=(3,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.output_features, kernel_size=(1,), padding='SAME')(x)  # Final output: 2 channels
        return x  # Shape: (batch_size, time_steps, 2)
    

n1_blocks = 15
n2_blocks = 15
n_piece = 8
n_source = n_piece * 2
spacing = 15. + 0.075*15.  # 1.0  # 15 mm
hinge_length = 0.075*15.  # Same as bond length
initial_angle = 25.0*jnp.pi/180

horizontal_shifts, vertical_shifts = QuadGeometry_Circle_InputSource(n1_blocks, n2_blocks, spacing=spacing, bond_length=hinge_length, n_piece=n_piece).get_design_from_rotated_square(
    angle=initial_angle,
)  # Initial design

# use previous parameters
# Mechanical params
k_stretch = 4.00  # stretching stiffness 120. N/mm
k_shear = k_stretch * 2.1e-01 # shearing stiffness 1.19 N/mm
k_rot = k_stretch * 2.6902e-03 * 20.0 # rotational stiffness 1.50 Nmm
density = 3.2325226e-08 # 0.03859661 * 1e-6  # Mg/mm^2

# Dynamic loading (I don't use these parameters)----------
amplitude = 0.5 * spacing  # 0.5 * spacing default 
loading_rate = 30.0 # jnp.array([30., 0., 20., 20., 20.])  #  Hz loading frequency for dynamic input
input_delay = 0.1*30.0**-1 * jnp.ones((n1_blocks,))
#---------------------------------------------------------
min_angle=-15*jnp.pi/180
cutoff_angle=-10*jnp.pi/180

simulation_time = 1.0
n_timepoints = 200
bond_length = hinge_length
linearized_strains = False
use_contact = False
k_contact = k_rot

geometry = QuadGeometry_Circle_InputSource(
    n1_blocks=n1_blocks,
    n2_blocks=n2_blocks,
    spacing=spacing,
    bond_length=bond_length
)

block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = geometry.get_parametrization()
_bond_connectivity = bond_connectivity()  # Compute bond connectivity once as it is constant
_reference_bond_vectors = reference_bond_vectors(horizontal_shifts, vertical_shifts)  # Compute reference bond vectors once as they are constant

k_stretch_arr = k_stretch * jnp.ones((_bond_connectivity.shape[0],))  # stretching stiffness 120. N/mm
k_shear_arr = k_shear * jnp.ones((_bond_connectivity.shape[0],)) # shearing stiffness 1.19 N/mm
k_rot_arr = k_rot * jnp.ones((_bond_connectivity.shape[0],)) # rotational stiffness 1.50 Nmm

# NOTE: Damping is assumed to be the same for all blocks as it is small enough that the inertia change during optimization is negligible.
# The reference is taken to be zero angle rotated square geometry.
# 0.36125, 0.02175026 = mass and inertia of a single square of zero angle and unitary spacing and density and 0.15 bond length.
dampall = 0.624118 
dampxy = 0.884512
damprot = 0.85392

damping_params = jnp.array([dampall * 0.186, dampxy * 0.36125, damprot * 0.02175026])
print( "initial damping params : ", damping_params )
damping = damping_params[0] * jnp.array([
    2 * (damping_params[1] * density * spacing**2 * k_shear)**0.5,
    2 * (damping_params[1] * density * spacing**2 * k_shear)**0.5,
    2 * (damping_params[2] * density * spacing**4 * k_rot)**0.5
]) * jnp.ones((geometry.n_blocks, 3))


# Damping
damped_blocks = jnp.arange(geometry.n_blocks)

n_excited_blocks = geometry.n_piece
idx_input_source = jnp.arange(geometry.n_quads,geometry.n_quads+n_excited_blocks)
driven_block_DOF_pairs = jnp.array([
    jnp.tile(
        idx_input_source,
        2),
    jnp.array([0]*n_excited_blocks + [1]*n_excited_blocks)
]).T
print("driven_block_DOF_pairs", driven_block_DOF_pairs)

# No constrained blocks
constrained_block_DOF_pairs = jnp.zeros((0,2), dtype=int)

driven_blocks_ids = jnp.unique( driven_block_DOF_pairs[:, 0] )
constrained_DOFs_loading_vector = jnp.zeros((len(constrained_block_DOF_pairs),))

allowed_blocks = jnp.arange( geometry.n_quads+n_excited_blocks )
moving_blocks_ids = allowed_blocks
# moving_blocks_ids = jnp.delete( allowed_blocks, clamped_blocks_ids ) # or jnp.delete( jnp.arange(geometry.n_blocks), constrained_blocks_ids )

loaded_DOF_ids = jnp.array([block_id * 3 + DOF_id for block_id, DOF_id in driven_block_DOF_pairs])

def pulse(t, amplitude, loading_rate):
    return amplitude * jnp.where(
        (t > 0.) & (t < loading_rate**-1),
        (1 - jnp.cos(2*jnp.pi * loading_rate * t))/2,
        0.
    )
    
excited_blocks_fn = None

if excited_blocks_fn is None:
    # Apply sinthetic pulse loading
    # NOTE: This is used for optimization.
    def constrained_DOFs_fn(t, amplitude, loading_rate, input_delay):
        return 0.0 * constrained_DOFs_loading_vector
else:
    # Apply user-defined loading
    # NOTE: This can be used to apply the experimental loading
    def constrained_DOFs_fn(t, **kwargs):
        return excited_blocks_fn(t) * constrained_DOFs_loading_vector
    
# Construct strain energy
strain_energy = build_strain_energy(
    bond_connectivity=_bond_connectivity,
    bond_energy_fn=ligament_energy_linearized if linearized_strains else ligament_energy,
)
contact_energy = build_contact_energy(bond_connectivity=_bond_connectivity)
potential_energy = combine_block_energies(strain_energy, contact_energy) if use_contact else strain_energy

atol = 1e-4
rtol = 1e-4

free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)
damped_DOF_ids = jnp.concatenate([jnp.arange(block_id * 3, (block_id + 1) * 3) for block_id in damped_blocks])

ts = jnp.linspace(0, simulation_time, n_timepoints)

vertices = centroid_node_vectors(horizontal_shifts, vertical_shifts)
areas = jnp.stack([ polygon_area(vertice) for vertice in vertices ], axis=0)
area_moments = jnp.stack([ polygon_polar_moment(vertice) for vertice in vertices ], axis=0)

density_arr = jnp.ones( (len(areas)) ) * density
density_arr = density_arr.at[geometry.n_quads:].set(2.0 * density) # shell blocks are heavier than inner blocks

target_blocks = jnp.array(args.measure_idx)
print( "target blocks : ", target_blocks )
density_arr = density_arr.at[target_blocks].set( density + 0.007301278265064265 * 1e-6  )
translational_inertia = density_arr * areas
rotational_inertia = density_arr * area_moments


_inertia = jnp.column_stack((translational_inertia, translational_inertia, rotational_inertia))

@eqx.filter_jit
def forward_problem(inputs, output_data, free_DOF_ids, constrained_DOF_ids, all_DOF_ids):
    
    # Analysis params
    timepoints = jnp.linspace(0.0, simulation_time, n_timepoints)
    horizontal_shifts, vertical_shifts = inputs

    def loading_fn(state, t, **kwargs):
        return vmap(jnp.interp, in_axes=(None,None,1))( t, timepoints, output_data )

    # Setup solver
    solve_dynamics = setup_dynamic_solver(
        geometry=geometry,
        energy_fn=potential_energy,
        # add new arguments
        loaded_DOF_ids=loaded_DOF_ids,
        loading_fn=loading_fn,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs,
        constrained_DOFs_fn=constrained_DOFs_fn,
        damped_DOF_ids=damped_DOF_ids,
        free_DOF_ids=free_DOF_ids,
        constrained_DOF_ids=constrained_DOF_ids,
        all_DOF_ids=all_DOF_ids,
        atol=atol,
        rtol=rtol,
    )

    # Define control params
    control_params = ControlParams(
        geometrical_params=GeometricalParams(
            block_centroids=block_centroids(horizontal_shifts, vertical_shifts),
            centroid_node_vectors=centroid_node_vectors(horizontal_shifts, vertical_shifts),
        ),
        mechanical_params=MechanicalParams(
            bond_params=LigamentParams(
                k_stretch=k_stretch_arr,
                k_shear=k_shear_arr,
                k_rot=k_rot_arr,
                reference_vector=_reference_bond_vectors,
            ),
            density=density,
            damping=damping,
            contact_params=ContactParams(
                k_contact=k_contact,
                min_angle=min_angle,
                cutoff_angle=cutoff_angle,
            ),
            inertia=_inertia
        ),
        constraint_params=dict( # I don't use this parameter
            amplitude=amplitude,
            loading_rate=loading_rate,
            input_delay=input_delay,
        ),
        loading_params=dict( # I don't use this parameter
            amplitude=amplitude,
            loading_rate=loading_rate,
            input_delay=input_delay,
        ),
    )

    # Initial conditions
    # state0 = jnp.zeros((2, geometry.n_blocks, 3))
    state0 = 1e-10 * jnp.ones((2, geometry.n_blocks, 3))

    start = time.time()
    # Solve dynamics
    solution = solve_dynamics(
        state0=state0,
        timepoints=timepoints,
        control_params=control_params,
    )
    print("duration : ", time.time() - start)
    
    return solution


key = random.PRNGKey(0)
N = 300


measure_idx = jnp.array(args.measure_idx)
n_measure = measure_idx.shape[0]

filename = f"012326_circular_structure_dataset_n_source_8_exp_param_free_set_4.pickle"
angle_cut = 360. / geometry.n_piece
def map_func(output_data_single):
    output_data_temp = jnp.zeros( (n_timepoints, 2*n_excited_blocks) ) # (ts, boundary blocks * 2)
    for i in range(n_excited_blocks):
        normal_angle = (+180 - angle_cut/2. - angle_cut * i) * jnp.pi / 180.
        output_data_temp = output_data_temp.at[:,i].set( output_data_single[:,i] * jnp.cos(normal_angle) )
        output_data_temp = output_data_temp.at[:,n_excited_blocks+i].set( output_data_single[:,i] * jnp.sin(normal_angle) )
    return output_data_temp

if len(measure_idx) == 1:
    print("Checking for data file at:", filename)
    with open(filename, "rb") as f:
        data_dict = pickle.load(f)

    print("input data shape : ", data_dict["input_data"].shape)
    input_data = jnp.zeros_like(data_dict["input_data"][:,:,2,measure_idx,:].squeeze()) # (N, time_steps, 3)
    input_data = input_data.at[:,:,0:2].set(data_dict["input_data"][:,:,2,measure_idx,0:2].squeeze()) # (N, time_steps, 2)
    input_data = input_data.at[:,:,2].set(data_dict["input_data"][:,:,1,measure_idx,2].squeeze()) # (N, time_steps, 2)
    output_data = data_dict["output_data"]
    output_data_all = vmap(map_func)(output_data)
else:
    filename = f"012326_circular_structure_dataset_n_source_8_exp_param_free_set_4_idx_{measure_idx_tag}.pickle"
    if os.path.exists(filename):
        print("Loading data from file...")
        with open(filename, "rb") as f:
            data_dict = pickle.load(f)
            input_data = data_dict["input_data"]
            output_data_all = vmap(map_func)(data_dict["output_data"])
        print("Data loaded from file.")
        nest_forward_problem = lambda inputs : vmap(forward_problem, in_axes=(None,0,None,None,None))(inputs, output_data_all, free_DOF_ids, constrained_DOF_ids, all_DOF_ids)
    else:
        print(f"{filename} file not found. Generating data and saving to file...")
        print("Generating data and saving to file...")

        data_path = f"012326_circular_structure_dataset_n_source_8_exp_param_free_set_4.pickle"
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)
            output_data = data_dict["output_data"]

        output_data_all = vmap(map_func)(output_data) # force function set
        print("output data shape : ", output_data_all.shape)
        nest_forward_problem = lambda inputs : vmap(forward_problem, in_axes=(None,0,None,None,None))(inputs, output_data_all, free_DOF_ids, constrained_DOF_ids, all_DOF_ids)
        input_data = nest_forward_problem((horizontal_shifts, vertical_shifts))
        print("Input data shape:", input_data.shape)
        data_dict = {"input_data": input_data, "output_data": output_data} # input data shape (N, time_steps, 2*3)
        with open(filename, "xb") as f:
            pickle.dump(data_dict, f)
        print("Data generated and saved to file.")


test_N = input_data.shape[0] // 10 # 10 % of total data will be used as test data
test_idx = jax.random.randint(key, shape=(test_N,), minval=0, maxval=input_data.shape[0])

mask = jnp.ones(input_data.shape[0], dtype=bool).at[test_idx].set(False)

training_input_data = input_data[mask]
training_output_data = output_data[mask]

test_input_data = input_data[test_idx]
test_output_data = output_data[test_idx]

print("input data size : ", input_data.shape)
print("output data size : ", output_data.shape)

mm_scaler = minmax_scaler(training_input_data, training_output_data)
train_data_normalized, train_output_normalized = mm_scaler.transform(training_input_data, training_output_data)
test_data_normalized, test_output_normalized = mm_scaler.transform(test_input_data, test_output_data)

model = TimeSeriesCNN(output_features=n_piece)
init_params = model.init(key, input_data)
preds = model.apply(init_params, input_data)

def inner_loss(params, outer_parameters, data):
  inputs, targets = data
  predictions = model.apply(params, inputs)
  mse = jnp.mean((predictions - targets)**2)  # this is L(phi_prime, D^{tr}_i)
  return mse

maxiter = 20
tol = 1e-7
inner_solver = OptaxSolver(opt=optax.adam(args.innerlr),
                            fun=inner_loss,
                            maxiter=args.inneriter,
                            tol=1e-12,
                            implicit_diff=True,
                            implicit_diff_solve=Partial(
                            linear_solve.solve_cg,
                            maxiter=maxiter, # 5
                            tol=tol),
                            )

hori_shifts_length = (n1_blocks+1) * n2_blocks * 2

nest_forward_problem = lambda inputs : vmap(forward_problem, in_axes=(None,0,None,None,None))(inputs, output_data_all, free_DOF_ids, constrained_DOF_ids, all_DOF_ids)
def outer_loss(meta_params):

    horizontal_shifts = meta_params[:hori_shifts_length].reshape((n1_blocks+1, n2_blocks, 2))
    vertical_shifts = meta_params[hori_shifts_length:].reshape((n1_blocks, n2_blocks+1, 2))
    inputs = (horizontal_shifts, vertical_shifts)
    input_data = nest_forward_problem(inputs)

    training_input_data = input_data[mask]
    training_output_data = output_data[mask]
    test_input_data = input_data[test_idx]
    test_output_data = output_data[test_idx]
    train_data_normalized, train_output_normalized = mm_scaler.transform(training_input_data, training_output_data)
    test_data_normalized, test_output_normalized = mm_scaler.transform(test_input_data, test_output_data)

    model = TimeSeriesCNN(output_features=n_piece)
    nn_params = model.init(key, input_data)

    in_params_sol, state = inner_solver.run(
        nn_params, 
        meta_params,
        (train_data_normalized, train_output_normalized)
        ) 

    prediction = model.apply(in_params_sol, test_data_normalized)
    loss = jnp.mean((prediction - test_output_normalized)**2)  # L(\phi, D^{te}_i)

    return loss, (in_params_sol, state, training_input_data, training_output_data, test_input_data, test_output_data)
    # return loss, (in_params_sol, state)

# horizontal_shifts, vertical_shifts = inputs
meta_params = jnp.concatenate( (horizontal_shifts.flatten(), vertical_shifts.flatten()))
print("meta params shape : ", meta_params.shape)

solver = OptaxSolver(
  opt=optax.adam(args.outerlr), 
  fun=outer_loss, 
  maxiter=args.outeriter, 
  has_aux=True,
  tol=1e-8,
)
state = solver.init_state(meta_params)
pbar = tqdm(range(solver.maxiter))

gradient_subopt = []
outer_losses = []
param_record = []
in_param_record = []

start_time = time.time()
for it in pbar:
    param_record.append(meta_params)
    start_time_outer = time.time()
    meta_params, state = solver.update(meta_params, state)
    print(f"{it}th iteration takes {time.time()-start_time_outer} seconds")
    outer_losses.append(state.value)
    in_param_record.append(state.aux)
    pbar.set_description(f"Outer loss {state.value:.3e}")

    record_dict = {"outer_iter": args.outeriter,
                    "inner_iter": args.inneriter,
                    "outer_lr": args.outerlr,
                    "inner_lr": args.innerlr,
                    "design_record": param_record,
                    "nn_record": in_param_record,
                    "loss_record": outer_losses}

    with open(time_string+f"_circular_robot_cooptimization_n_source_{n_source}_measure_idx_{measure_idx_tag}.pickle", "wb") as f:
        pickle.dump(record_dict, f)

print("time : ", time.time()-start_time)
