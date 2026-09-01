"""Per-structure geometry, mechanical-parameter, and DOF setup.

Each ``build_*`` function returns a ``StructureSetup`` describing one
structure type's initial design, mechanical parameters, driven/measured/
constrained DOFs, and (for the circular structure) how to turn the raw
per-panel force dataset into the x/y loading the solver expects. This is
the "structure-specific" half of the forward-simulation package; the
solver mechanics themselves live in ``forward_simulation.solver`` and are
shared by all four.

Values here are taken directly from the original co-optimization scripts
(now archived) for each structure -- see the plan/commit history for the
side-by-side comparison across structures.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp

from blockymetamaterials.energy import (
    build_contact_energy,
    build_strain_energy,
    combine_block_energies,
    ligament_energy,
    ligament_energy_linearized,
)
from blockymetamaterials.geometry import (
    DOFsInfo,
    Geometry,
    QuadGeometry_Circle_InputSource,
    QuadGeometry_InputSource,
    polygon_area,
    polygon_polar_moment,
)
from blockymetamaterials.utils import (
    ContactParams,
    ControlParams,
    GeometricalParams,
    LigamentParams,
    MechanicalParams,
)

# Shared across every structure in the original scripts.
_K_STRETCH = 4.00
_K_SHEAR = _K_STRETCH * 2.1e-01
_K_ROT = _K_STRETCH * 2.6902e-03 * 20.0
_DENSITY = 3.2325226e-08
_MIN_ANGLE = -15 * jnp.pi / 180
_CUTOFF_ANGLE = -10 * jnp.pi / 180
_SIMULATION_TIME = 1.0
_N_TIMEPOINTS = 200
_ATOL = _RTOL = 1e-4
_SENSOR_MASS = 8.58e-6
_SENSOR_MOMENT = 1 / 12 * (32**2 + 23**2) * _SENSOR_MASS
_SENSOR_DENSITY_BUMP = 0.007301278265064265e-6

_DAMP_ALL = 0.624118
_DAMP_XY = 0.884512
_DAMP_ROT = 0.85392


def _damping_array(n_blocks: int) -> jnp.ndarray:
    damping_params = jnp.array([_DAMP_ALL * 0.186, _DAMP_XY * 0.36125, _DAMP_ROT * 0.02175026])
    return damping_params[0] * jnp.array([
        2 * (damping_params[1] * _DENSITY * _SPACING**2 * _K_SHEAR) ** 0.5,
        2 * (damping_params[1] * _DENSITY * _SPACING**2 * _K_SHEAR) ** 0.5,
        2 * (damping_params[2] * _DENSITY * _SPACING**4 * _K_ROT) ** 0.5,
    ]) * jnp.ones((n_blocks, 3))


_SPACING = 15.0 + 0.075 * 15.0
_HINGE_LENGTH = 0.075 * 15.0


@dataclass
class StructureSetup:
    name: str
    geometry: Geometry
    initial_design: Tuple[jnp.ndarray, jnp.ndarray]
    potential_energy: Callable
    free_DOF_ids: jnp.ndarray
    constrained_DOF_ids: jnp.ndarray
    all_DOF_ids: jnp.ndarray
    loaded_DOF_ids: jnp.ndarray
    measure_idx: jnp.ndarray
    constrained_block_DOF_pairs: jnp.ndarray
    constrained_DOFs_fn: Callable
    damped_DOF_ids: jnp.ndarray
    simulation_time: float
    n_timepoints: int
    atol: float
    rtol: float
    output_data_transform: Optional[Callable]
    _block_centroids_fn: Callable
    _centroid_node_vectors_fn: Callable
    _bond_params: LigamentParams
    _density_arr: jnp.ndarray
    _damping: jnp.ndarray
    _contact_params: ContactParams
    _inertia: jnp.ndarray

    def build_control_params(self, horizontal_shifts: jnp.ndarray, vertical_shifts: jnp.ndarray) -> ControlParams:
        return ControlParams(
            geometrical_params=GeometricalParams(
                block_centroids=self._block_centroids_fn(horizontal_shifts, vertical_shifts),
                centroid_node_vectors=self._centroid_node_vectors_fn(horizontal_shifts, vertical_shifts),
            ),
            mechanical_params=MechanicalParams(
                bond_params=self._bond_params,
                density=self._density_arr,
                damping=self._damping,
                contact_params=self._contact_params,
                inertia=self._inertia,
            ),
            constraint_params={},
            loading_params={},
        )


def _constrained_DOFs_fn(n_constrained: int):
    zeros = jnp.zeros((n_constrained,))
    return lambda t, **kwargs: zeros


def _square_structure(
    name: str,
    n1_blocks: int,
    n2_blocks: int,
    n_source: int,
    measure_idx,
    linearized_strains: bool,
    use_contact: bool,
    k_contact: Optional[float] = None,
) -> StructureSetup:
    measure_idx = jnp.asarray(measure_idx)

    geometry = QuadGeometry_InputSource(
        n1_blocks=n1_blocks, n2_blocks=n2_blocks, spacing=_SPACING,
        bond_length=_HINGE_LENGTH, n_source=n_source,
    )
    horizontal_shifts, vertical_shifts = QuadGeometry_InputSource(
        n1_blocks, n2_blocks, spacing=_SPACING, bond_length=_HINGE_LENGTH, n_source=n_source,
    ).get_design_from_rotated_square(angle=-25.0 * jnp.pi / 180)

    block_centroids_fn, centroid_node_vectors_fn, bond_connectivity_fn, reference_bond_vectors_fn = (
        geometry.get_parametrization()
    )
    bond_connectivity = bond_connectivity_fn()
    reference_bond_vectors = reference_bond_vectors_fn(horizontal_shifts, vertical_shifts)

    k_stretch_arr = _K_STRETCH * jnp.ones((bond_connectivity.shape[0],))
    k_shear_arr = _K_SHEAR * jnp.ones((bond_connectivity.shape[0],))
    k_rot_arr = _K_ROT * jnp.ones((bond_connectivity.shape[0],))
    bond_params = LigamentParams(
        k_stretch=k_stretch_arr, k_shear=k_shear_arr, k_rot=k_rot_arr, reference_vector=reference_bond_vectors,
    )

    density_arr = _DENSITY * jnp.ones((geometry.n_blocks,))
    density_arr = density_arr.at[measure_idx].set(_DENSITY + _SENSOR_DENSITY_BUMP)

    vertices = centroid_node_vectors_fn(horizontal_shifts, vertical_shifts)
    areas = jnp.stack([polygon_area(v) for v in vertices], axis=0)
    area_moments = jnp.stack([polygon_polar_moment(v) for v in vertices], axis=0)
    translational_inertia = density_arr * areas
    rotational_inertia = density_arr * area_moments
    translational_inertia = translational_inertia.at[measure_idx].set(translational_inertia[measure_idx] + _SENSOR_MASS)
    rotational_inertia = rotational_inertia.at[measure_idx].set(rotational_inertia[measure_idx] + _SENSOR_MOMENT)
    inertia = jnp.column_stack((translational_inertia, translational_inertia, rotational_inertia))

    driven_block_DOF_pairs = jnp.array([
        jnp.tile(jnp.arange(geometry.n_quads, geometry.n_quads + n_source), 1),
        jnp.array([1] * n_source),
    ]).T
    clamped_block_DOF_pairs = jnp.array([
        jnp.tile(jnp.arange(geometry.n_quads + n_source, geometry.n_quads + n_source + 1), 3),
        jnp.array([0, 1, 2]),
    ]).T
    loaded_DOF_ids = jnp.array([block_id * 3 + dof_id for block_id, dof_id in driven_block_DOF_pairs])

    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, clamped_block_DOF_pairs)
    damped_DOF_ids = jnp.concatenate([jnp.arange(b * 3, (b + 1) * 3) for b in range(geometry.n_blocks)])

    strain_energy = build_strain_energy(
        bond_connectivity=bond_connectivity,
        bond_energy_fn=ligament_energy_linearized if linearized_strains else ligament_energy,
    )
    contact_energy = build_contact_energy(bond_connectivity=bond_connectivity)
    potential_energy = combine_block_energies(strain_energy, contact_energy) if use_contact else strain_energy

    contact_params = ContactParams(
        k_contact=k_contact if k_contact is not None else _K_ROT, min_angle=_MIN_ANGLE, cutoff_angle=_CUTOFF_ANGLE,
    )

    return StructureSetup(
        name=name,
        geometry=geometry,
        initial_design=(horizontal_shifts, vertical_shifts),
        potential_energy=potential_energy,
        free_DOF_ids=free_DOF_ids,
        constrained_DOF_ids=constrained_DOF_ids,
        all_DOF_ids=all_DOF_ids,
        loaded_DOF_ids=loaded_DOF_ids,
        measure_idx=measure_idx,
        constrained_block_DOF_pairs=clamped_block_DOF_pairs,
        constrained_DOFs_fn=_constrained_DOFs_fn(clamped_block_DOF_pairs.shape[0]),
        damped_DOF_ids=damped_DOF_ids,
        simulation_time=_SIMULATION_TIME,
        n_timepoints=_N_TIMEPOINTS,
        atol=_ATOL,
        rtol=_RTOL,
        output_data_transform=None,
        _block_centroids_fn=block_centroids_fn,
        _centroid_node_vectors_fn=centroid_node_vectors_fn,
        _bond_params=bond_params,
        _density_arr=density_arr,
        _damping=_damping_array(geometry.n_blocks),
        _contact_params=contact_params,
        _inertia=inertia,
    )


def build_input_source_2(measure_idx=(7,)) -> StructureSetup:
    return _square_structure(
        "input_source_2", n1_blocks=5, n2_blocks=4, n_source=2, measure_idx=measure_idx,
        linearized_strains=True, use_contact=False,
    )


def build_input_source_4(measure_idx=(25, 29)) -> StructureSetup:
    return _square_structure(
        "input_source_4", n1_blocks=11, n2_blocks=6, n_source=4, measure_idx=measure_idx,
        linearized_strains=True, use_contact=False,
    )


def build_input_source_8(measure_idx=(75,)) -> StructureSetup:
    return _square_structure(
        "input_source_8", n1_blocks=23, n2_blocks=6, n_source=8, measure_idx=measure_idx,
        linearized_strains=True, use_contact=False,
    )


def build_circular_structure(measure_idx=(48,)) -> StructureSetup:
    measure_idx = jnp.asarray(measure_idx)
    n1_blocks, n2_blocks, n_piece = 15, 15, 8
    n_excited_blocks = n_piece

    geometry = QuadGeometry_Circle_InputSource(
        n1_blocks=n1_blocks, n2_blocks=n2_blocks, spacing=_SPACING, bond_length=_HINGE_LENGTH,
    )
    horizontal_shifts, vertical_shifts = QuadGeometry_Circle_InputSource(
        n1_blocks, n2_blocks, spacing=_SPACING, bond_length=_HINGE_LENGTH, n_piece=n_piece,
    ).get_design_from_rotated_square(angle=25.0 * jnp.pi / 180)

    block_centroids_fn, centroid_node_vectors_fn, bond_connectivity_fn, reference_bond_vectors_fn = (
        geometry.get_parametrization()
    )
    bond_connectivity = bond_connectivity_fn()
    reference_bond_vectors = reference_bond_vectors_fn(horizontal_shifts, vertical_shifts)

    k_stretch_arr = _K_STRETCH * jnp.ones((bond_connectivity.shape[0],))
    k_shear_arr = _K_SHEAR * jnp.ones((bond_connectivity.shape[0],))
    k_rot_arr = _K_ROT * jnp.ones((bond_connectivity.shape[0],))
    bond_params = LigamentParams(
        k_stretch=k_stretch_arr, k_shear=k_shear_arr, k_rot=k_rot_arr, reference_vector=reference_bond_vectors,
    )

    density_arr = jnp.ones((geometry.n_blocks,)) * _DENSITY
    density_arr = density_arr.at[geometry.n_quads:].set(2.0 * _DENSITY)  # shell blocks are heavier
    density_arr = density_arr.at[measure_idx].set(_DENSITY + _SENSOR_DENSITY_BUMP)

    vertices = centroid_node_vectors_fn(horizontal_shifts, vertical_shifts)
    areas = jnp.stack([polygon_area(v) for v in vertices], axis=0)
    area_moments = jnp.stack([polygon_polar_moment(v) for v in vertices], axis=0)
    translational_inertia = density_arr * areas
    rotational_inertia = density_arr * area_moments
    inertia = jnp.column_stack((translational_inertia, translational_inertia, rotational_inertia))

    idx_input_source = jnp.arange(geometry.n_quads, geometry.n_quads + n_excited_blocks)
    driven_block_DOF_pairs = jnp.array([
        jnp.tile(idx_input_source, 2),
        jnp.array([0] * n_excited_blocks + [1] * n_excited_blocks),
    ]).T
    loaded_DOF_ids = jnp.array([block_id * 3 + dof_id for block_id, dof_id in driven_block_DOF_pairs])

    constrained_block_DOF_pairs = jnp.zeros((0, 2), dtype=int)  # no clamped blocks
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)
    damped_DOF_ids = jnp.concatenate([jnp.arange(b * 3, (b + 1) * 3) for b in range(geometry.n_blocks)])

    strain_energy = build_strain_energy(bond_connectivity=bond_connectivity, bond_energy_fn=ligament_energy)
    potential_energy = strain_energy  # use_contact = False

    contact_params = ContactParams(k_contact=_K_ROT, min_angle=_MIN_ANGLE, cutoff_angle=_CUTOFF_ANGLE)

    angle_cut = 360.0 / n_piece

    def output_data_transform(output_data_single: jnp.ndarray) -> jnp.ndarray:
        """Projects raw per-panel scalar force onto (x, y) loading components."""
        out = jnp.zeros((output_data_single.shape[0], 2 * n_excited_blocks))
        for i in range(n_excited_blocks):
            normal_angle = (180 - angle_cut / 2.0 - angle_cut * i) * jnp.pi / 180.0
            out = out.at[:, i].set(output_data_single[:, i] * jnp.cos(normal_angle))
            out = out.at[:, n_excited_blocks + i].set(output_data_single[:, i] * jnp.sin(normal_angle))
        return out

    return StructureSetup(
        name="circular_structure",
        geometry=geometry,
        initial_design=(horizontal_shifts, vertical_shifts),
        potential_energy=potential_energy,
        free_DOF_ids=free_DOF_ids,
        constrained_DOF_ids=constrained_DOF_ids,
        all_DOF_ids=all_DOF_ids,
        loaded_DOF_ids=loaded_DOF_ids,
        measure_idx=measure_idx,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs,
        constrained_DOFs_fn=_constrained_DOFs_fn(0),
        damped_DOF_ids=damped_DOF_ids,
        simulation_time=_SIMULATION_TIME,
        n_timepoints=_N_TIMEPOINTS,
        atol=_ATOL,
        rtol=_RTOL,
        output_data_transform=output_data_transform,
        _block_centroids_fn=block_centroids_fn,
        _centroid_node_vectors_fn=centroid_node_vectors_fn,
        _bond_params=bond_params,
        _density_arr=density_arr,
        _damping=_damping_array(geometry.n_blocks),
        _contact_params=contact_params,
        _inertia=inertia,
    )


STRUCTURES = {
    "input_source_2": build_input_source_2,
    "input_source_4": build_input_source_4,
    "input_source_8": build_input_source_8,
    "circular_structure": build_circular_structure,
}
