"""
The `geometry` module implements some geometries.
"""

# NOTE: This module acts as a sort of kitchen for geometric design spaces.


from typing import Callable, Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap


# Utility functions


def rotation_matrix(angle):
    """
    docstring
    """

    return jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                      [jnp.sin(angle), jnp.cos(angle)]])


def current_coordinates(vertices, centroids, angles, displacements):
    """
    Computes the deformed configuration coordinates.
    """

    def _current_coordinates(v, Q, c, d):
        return (Q @ v.T).T + c + d

    rotations = vmap(rotation_matrix)(angles) # (N, 2, 2)
    # current_coordinates_v = vmap(_current_coordinates, in_axes=(0, 0, 0, 0))  # Vectorize over blocks
    current_coordinates_v = jnp.concatenate([ _current_coordinates(vertices[i], rotations[i], centroids[i], displacements[i]) for i in range(len(vertices)) ])
    # return current_coordinates_v(vertices, rotations, centroids, displacements)
    return current_coordinates_v


def get_point_ids_in_bounding_box(points: jnp.ndarray, bounding_box: jnp.ndarray):
    """Returns the indices of the points that lie within the bounding box.

    Args:
        points (jnp.ndarray): array of shape (n_points, 2) collecting the coordinates of the points.
        bounding_box (jnp.ndarray): array of shape (2, 2) collecting the coordinates of the bounding box. The first row collects the coordinates of the bottom-left corner and the second row collects the coordinates of the top-right corner.

    Returns:
        jnp.ndarray: array of shape (n_points_in_bounding_box,) collecting the indices of the points that lie within the bounding box.
    """

    return jnp.where(
        (points[:, 0] >= bounding_box[0, 0]) & (points[:, 0] <= bounding_box[1, 0]) &
        (points[:, 1] >= bounding_box[0, 1]) & (points[:, 1] <= bounding_box[1, 1])
    )[0]


def get_point_ids_in_circle(points: jnp.ndarray, center: jnp.ndarray, radius: float):
    """Returns the indices of the points that lie within the circle.

    Args:
        points (jnp.ndarray): array of shape (n_points, 2) collecting the coordinates of the points.
        center (jnp.ndarray): array of shape (2,) collecting the coordinates of the center of the circle.
        radius (float): radius of the circle.

    Returns:
        jnp.ndarray: array of shape (n_points_in_circle,) collecting the indices of the points that lie within the circle.
    """

    return jnp.where(jnp.linalg.norm(points - center, axis=1) <= radius)[0]


def polygon_area(vertices: jnp.ndarray):
    """Computes area of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (jnp.ndarray): array of shape (n_vertices, 2).

    Returns:
        float: Area of the polygon.
    """

    v1 = jnp.roll(vertices, shift=1, axis=0)
    v2 = vertices

    return jnp.abs(jnp.sum(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2)


def polygon_centroid(vertices: jnp.ndarray):
    """Computes centroid of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (jnp.ndarray): array of shape (n_vertices, 2).

    Returns:
        jnp.ndarray: Centroid of the polygon.
    """

    area = polygon_area(vertices)
    v1 = jnp.roll(vertices, shift=1, axis=0)
    v2 = vertices
    x_plus_y = v1 + v2
    v_cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    return jnp.array([
        jnp.sum(x_plus_y[:, 0] * v_cross),
        jnp.sum(x_plus_y[:, 1] * v_cross)
    ]) / (6 * area)


def polygon_polar_moment(vertices: jnp.ndarray):
    """Computes polar moment of area of a polygon with `vertices` ordered counter-clockwise.

    Args:
        vertices (jnp.ndarray): array of shape (n_vertices, 2).

    Returns:
        float: Polar moment of area of the polygon.
    """

    centroid = polygon_centroid(vertices)
    v1 = jnp.roll(vertices, shift=1, axis=0) - centroid
    v2 = vertices - centroid

    return jnp.abs(
        jnp.sum((v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) * (
            v1[:, 0]**2 + v1[:, 0] * v2[:, 0] + v2[:, 0]**2 + v1[:, 1]**2 + v1[:, 1] * v2[:, 1] + v2[:, 1]**2
        )) / 12
    )


@vmap
def polygons_geometric_properties(vertices: jnp.ndarray):
    """Computes area, centroid, and polar moment of area of an array of polygons defined by `vertices`.

    Args:
        vertices (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: centroid, area, and polar moment of area of the polygons.
    """

    return polygon_centroid(vertices), polygon_area(vertices), polygon_polar_moment(vertices)


@jit
def compute_inertia(vertices: jnp.ndarray, density: Union[jnp.ndarray, float]):
    """Computes inertia of a set of blocks.

    Args:
        vertices (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2).
        density (Union[jnp.ndarray, float]): either a scalar or an array of shape (n_blocks, ) defining the mass density.

    Returns:
        jnp.ndarray: array of shape (n_blocks, 3) collecting the translational and rotational inertia of the blocks.
    """

    _, areas, area_moments = polygons_geometric_properties(vertices)

    translational_inertia = density * areas
    rotational_inertia = density * area_moments

    return jnp.column_stack((translational_inertia, translational_inertia, rotational_inertia))


def DOFsInfo(n_blocks: int, constrained_block_DOF_pairs: jnp.ndarray):
    """Computes arrays defining the free, constrained, and all DOFs.

    Args:
        n_blocks (int): Number of blocks in the geometry (i.e. geometry.n_blocks)
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: arrays defining the free, constrained, and all DOFs.
    """

    constrained_DOF_ids = jnp.array([block_id * 3 + DOF_id for block_id, DOF_id in constrained_block_DOF_pairs])
    all_DOF_ids = jnp.arange(n_blocks * 3)
    free_DOF_ids = jnp.array([dof for dof in all_DOF_ids if dof not in constrained_DOF_ids])

    return free_DOF_ids, constrained_DOF_ids, all_DOF_ids


def compute_edge_unit_vectors(current_block_nodes: jnp.ndarray, node_id: int):
    """Computes unit vectors from bond node to the two closest nodes of the same block.

    Args:
        current_block_coordinates (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the position of all the blocks' vertices.
        node_id (int): global node index.

    Returns:
        Tuple[jnp.array, jnp.array]: void and block angles.
    """

    # _, n_sides, _ = current_block_nodes.shape

    # node = current_block_nodes[node_id // n_sides, node_id % n_sides]
    node = current_block_nodes[node_id]

    # unit_vector_1 = current_block_nodes[node_id // n_sides, (node_id+1) % n_sides] - node
    unit_vector_1 = current_block_nodes[node_id + 1] - node
    unit_vector_1 = unit_vector_1/jnp.linalg.norm(unit_vector_1)

    unit_vector_2 = current_block_nodes[node_id - 1] - node
    unit_vector_2 = unit_vector_2/jnp.linalg.norm(unit_vector_2)

    return unit_vector_1, unit_vector_2


def compute_edge_lengths(centroid_node_vectors: jnp.ndarray):
    """Computes edge lengths of the blocks.

    Args:
        centroid_node_vectors (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the position of all the blocks' vertices relative to the centroids.

    Returns:
        jnp.ndarray: array of shape (n_blocks, n_nodes_per_block) collecting the edge lengths of the blocks.
    """

    return jnp.linalg.norm(
        jnp.roll(centroid_node_vectors, 1, axis=1) - centroid_node_vectors,
        axis=2
    )


def angle_between_unit_vectors(u1, u2):
    """Computes the signed angle between two unit vectors using arctan2.

    Args:
        u1 (jnp.ndarray): array of shape (2, ) defining the first unit vector.
        u2 (jnp.ndarray): array of shape (2, ) defining the second unit vector.

    Returns:
        float: Signed angle measured from u1 to u2 (positive counter-clockwise). Result is in the range [-pi, pi].
    """
    return jnp.arctan2(u1[0] * u2[1] - u1[1] * u2[0], u1[0] * u2[0] + u1[1] * u2[1])


def compute_edge_angles(current_block_nodes: jnp.ndarray, nodes: Tuple[int, int]):
    """Computes the two block and two void angles.

    Args:
        current_block_coordinates (jnp.ndarray): array of shape (all_number_of_nodes, 2) defining the position of all the blocks' vertices.
        nodes (Tuple[int, int]): tuple of node indices connected by a bond.

    Returns:
        Tuple[float, float, float, float]: void and block angles.
    """

    block_1_node_1, block_1_node_2 = compute_edge_unit_vectors(current_block_nodes, nodes[0])
    block_2_node_1, block_2_node_2 = compute_edge_unit_vectors(current_block_nodes, nodes[1])

    void_angle_1 = angle_between_unit_vectors(block_2_node_2, block_1_node_1)
    void_angle_2 = angle_between_unit_vectors(block_1_node_2, block_2_node_1)
    block_angle_1 = angle_between_unit_vectors(block_1_node_1, block_1_node_2)
    block_angle_2 = angle_between_unit_vectors(block_2_node_1, block_2_node_2)

    return void_angle_1, void_angle_2, block_angle_1, block_angle_2


def compute_xy_limits(points: jnp.ndarray):
    """Computes the the pair xlim, ylim for the given set of points.

    Args:
        points (jnp.ndarray): array of shape (n, 2)

    Returns:
        jnp.ndarray: array of xlim, ylim
    """

    return jnp.array([points.min(axis=0), points.max(axis=0)]).T


# Geometry classes
class Geometry:
    """
    Template class for defining geometric data.
    """

    n_blocks: int
    n_nodes: int
    block_centroids: Callable
    centroid_node_vectors: Callable
    bond_connectivity: Callable
    reference_bond_vectors: Callable

    def compute_geometry(self):
        """Any geometric class must implement the definition of the following data structures:
        - `block_centroids`: (ndarray): array of shape (n_blocks, 2) defining the centroid of each block.
        - `centroid_node_vectors` (ndarray): array of shape (n_blocks, n_nodes_per_block, 2) defining the vectors connecting the centroid of the block to each node.
        - `bond_connectivity` (ndarray): array of shape (n_bonds, 2) defining the pair of nodes connected by bonds i.e. each row is of the form [node1, node2].
        - `reference_bond_vectors` (ndarray): array of shape (n_bonds, 2) defining the reference configuration of the bonds.

        Raises:
            NotImplementedError: `compute_geometry` must define `centroid_node_vectors`, `bond_connectivity`, and `reference_bond_vectors`.
        """
        raise NotImplementedError("Child classes should implement this method.")

    def get_reference_geometry(self, *args):
        """
        Computes reference configuration of all the nodes.
        """

        try:
            centroid_node_vectors = self.centroid_node_vectors(*args)
        except AttributeError as err:
            self.compute_geometry()
            centroid_node_vectors = self.centroid_node_vectors(*args)

        centroids = self.block_centroids(*args)

        return vmap(lambda block_nodes, centroid: block_nodes + centroid, in_axes=(0, 0))(centroid_node_vectors, centroids)

    def get_xy_limits(self, *args):
        """
        Computes reference coonfiguration xy limits.
        """

        vertices = self.get_reference_geometry(*args).reshape((self.n_nodes, 2))
        return compute_xy_limits(vertices)

    def get_parametrization(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Returns the set of functions parameterizing the geometry.

        Returns:
            Tuple[Callable, Callable, Callable, Callable]: parameterizing functions: block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors.
        """

        self.compute_geometry()

        return self.block_centroids, self.centroid_node_vectors, self.bond_connectivity, self.reference_bond_vectors


class LatticeGeometry(Geometry):
    """
    docstring
    """

    def __init__(self, n1_cells: int, n2_cells: int, n_bpc: int, direct_basis: jnp.ndarray = jnp.eye(2)):
        """Lattice geometry (not necessarily periodic) composed of unit cells arranged in a parallelepiped array.

        Args:
            n1_cells (int): Number of cells along the x direction.
            n2_cells (int): Number of cells along the y direction.
            n_bpc (int): Number of blocks per cell.
            direct_basis (jnp.ndarray, optional): Direct basis of the tesselation. Defaults to jnp.eye(2).
        """

        self.n1_cells = n1_cells
        self.n2_cells = n2_cells
        self.n_bpc = n_bpc
        self.n_cells = self.n1_cells * self.n2_cells
        self.n_blocks = self.n_cells * self.n_bpc
        self.direct_basis = direct_basis


class RotatedSquareGeometry(LatticeGeometry):
    """
    Rotated square geometry.
    """

    def __init__(self, n1_cells: int, n2_cells: int, spacing: float = 1., bond_length: float = 0.1):
        """
        Creates a rotated square lattice geometry.
        """

        super().__init__(n1_cells=n1_cells, n2_cells=n2_cells, n_bpc=4, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = 2 * self.n1_cells
        self.n2_blocks = 2 * self.n2_cells
        self.n_npb = 4
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between `angle` and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def _centroid_node_vectors(angle, n1: int, n2: int):
            v0 = (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) * \
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)])
            return vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

        def centroid_node_vectors(angle):
            """
            Computes the vectors connecting the centroid of the block to each node.
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))

            return vmap(_centroid_node_vectors, in_axes=(None, 0, 0))(angle, n1s, n2s)

        def block_centroids(angle):
            """
            Computes blocks' centroid.
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2] for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2] for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            horizontal_bonds = jnp.full(((self.n1_blocks - 1) * self.n2_blocks, 2),
                                        self.bond_length*jnp.array([1., 0.]))
            vertical_bonds = jnp.full(((self.n2_blocks - 1) * self.n1_blocks, 2),
                                      self.bond_length*jnp.array([0., 1.]))

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, initial_angle):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(initial_angle)


class KagomePeriodicGeometry(LatticeGeometry):
    """
    Kagome periodic geometry.
    """
    # [block_numeration]([cell_numeration])
    #
    #                    2(5)
    #                  /     \
    #                 /       \
    #   2(2) --- 1(1) 0(3) --- 1(4)
    #    \       /
    #     \     /
    #      0(0)

    def __init__(self, n1_cells: int, n2_cells: int, direct_basis=jnp.array([[1., 0.], [jnp.cos(jnp.pi / 3), jnp.sin(jnp.pi / 3)]]), bond_length: float = 0.1):
        """
        Creates a kagome lattice geometry.
        """

        super().__init__(n1_cells=n1_cells, n2_cells=n2_cells, n_bpc=2, direct_basis=direct_basis)
        self.bond_length = bond_length
        self.n_npb = 3
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between `shifts` and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        # Reference vectors for the bonds at the vertices of the triangles
        reference_vector_internal_bond = self.bond_length * jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)])
        reference_vector_boundary_bond_1 = self.bond_length * jnp.array([0., -1.])
        reference_vector_boundary_bond_2 = self.bond_length * jnp.array([-jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)])

        def reference_node_vectors(shifts: jnp.ndarray = jnp.zeros((3, 2))):
            # regular kagome
            a1, a2 = self.direct_basis
            block_1 = jnp.array([a1 / 2, a1 / 2 + a2 / 2, a2 / 2]) - \
                0.5*jnp.array([reference_vector_boundary_bond_1,
                              reference_vector_internal_bond,
                              reference_vector_boundary_bond_2])  # make space for bonds' length
            block_1 -= polygon_centroid(block_1)
            block_2 = vmap(lambda v: jnp.dot(rotation_matrix(-jnp.pi / 3), v))(block_1)
            # apply shifts
            block_1 += shifts
            block_2 += shifts[jnp.array([1, 2, 0])]
            # return a single cell
            return jnp.array([block_1, block_2])

        def centroid_node_vectors(shifts: jnp.ndarray = jnp.zeros((3, 2))):
            """
            Computes the vectors connecting the centroid of the block to each node.
            """

            # Compute the shifts wrt to the regular kagome
            reference_vectors = reference_node_vectors(shifts)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            cell = vmap(lambda block_nodes, shift: block_nodes - shift,
                        in_axes=(0, 0))(reference_vectors, centroid_shifts)
            return jnp.tile(cell, (self.n_cells, 1, 1))

        def block_centroids(shifts: jnp.ndarray = jnp.zeros((3, 2))):
            """
            Computes blocks' centroid.
            """

            # Compute centroids of the regular kagome.
            a1, a2 = self.direct_basis
            block_1 = polygon_centroid(jnp.array([a1 / 2, a1 / 2 + a2 / 2, a2 / 2]))
            block_2 = polygon_centroid(jnp.array([a1 / 2 + a2 / 2, a1 + a2 / 2, a1 / 2 + a2]))
            # Compute the shifts wrt to the regular kagome
            reference_vectors = reference_node_vectors(shifts)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells))
            n1s, n2s = n1s.reshape((self.n_cells,)), n2s.reshape((self.n_cells,))
            return jnp.concatenate(
                vmap(
                    lambda n1, n2: jnp.array([block_1, block_2]) + centroid_shifts + n1 * a1 + n2 * a2, in_axes=(0, 0)
                )(n1s, n2s)
            )

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)

        def translate_internal_bond(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + (n2 * self.n1_cells + n1) * n_npc

        def translate_boundary_bond1(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + jnp.array([((n2 + 1) * self.n1_cells + n1) * n_npc, (n2 * self.n1_cells + n1) * n_npc])

        def translate_boundary_bond2(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + jnp.array([(n2 * self.n1_cells + n1 + 1) * n_npc, (n2 * self.n1_cells + n1) * n_npc])

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            internal_connectivity = jnp.array([[1, 3]])
            boundary_connectivity1 = jnp.array([[0, 5]])
            boundary_connectivity2 = jnp.array([[2, 4]])
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells))
            n1s, n2s = n1s.reshape((self.n_cells,)), n2s.reshape((self.n_cells,))
            internal_bonds = jnp.concatenate(
                vmap(translate_internal_bond, in_axes=(None, 0, 0))(internal_connectivity, n1s, n2s)
            )
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells - 1))
            n1s = n1s.reshape((self.n1_cells * (self.n2_cells - 1),))
            n2s = n2s.reshape((self.n1_cells * (self.n2_cells - 1),))
            boundary_bonds1 = jnp.concatenate(
                vmap(translate_boundary_bond1, in_axes=(None, 0, 0))(boundary_connectivity1, n1s, n2s)
            )
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells - 1), jnp.arange(self.n2_cells))
            n1s = n1s.reshape(((self.n1_cells - 1) * self.n2_cells,))
            n2s = n2s.reshape(((self.n1_cells - 1) * self.n2_cells,))
            boundary_bonds2 = jnp.concatenate(
                vmap(translate_boundary_bond2, in_axes=(None, 0, 0))(boundary_connectivity2, n1s, n2s)
            )

            return jnp.concatenate([internal_bonds, boundary_bonds1, boundary_bonds2])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            internal_bonds = jnp.full(
                (self.n_cells, 2),
                reference_vector_internal_bond
            )
            boundary_bonds_1 = jnp.full(
                (self.n1_cells * (self.n2_cells - 1), 2),
                reference_vector_boundary_bond_1
            )
            boundary_bonds_2 = jnp.full(
                ((self.n1_cells - 1) * self.n2_cells, 2),
                reference_vector_boundary_bond_2
            )

            return jnp.concatenate([internal_bonds, boundary_bonds_1, boundary_bonds_2])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, shifts: jnp.ndarray = jnp.zeros((3, 2))):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(shifts)


class KagomeGeometry(LatticeGeometry):
    """
    Non-periodic Kagome geometry.
    """
    # [block_numeration]([cell_numeration])
    #
    #                    2(5)
    #                  /     \
    #                 /       \
    #   2(2) --- 1(1) 0(3) --- 1(4)
    #    \       /
    #     \     /
    #      0(0)

    def __init__(self, n1_cells: int, n2_cells: int, direct_basis=jnp.array([[1., 0.], [jnp.cos(jnp.pi / 3), jnp.sin(jnp.pi / 3)]]), bond_length: float = 0.1):
        """
        Creates a kagome lattice geometry.
        """

        super().__init__(n1_cells=n1_cells, n2_cells=n2_cells, n_bpc=2, direct_basis=direct_basis)
        self.bond_length = bond_length
        self.n_npb = 3
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between `shifts` and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        # Reference vectors for the bonds at the vertices of the triangles
        reference_vector_internal_bond = self.bond_length * jnp.array([jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)])
        reference_vector_boundary_bond_1 = self.bond_length * jnp.array([0., -1.])
        reference_vector_boundary_bond_2 = self.bond_length * jnp.array([-jnp.cos(jnp.pi / 6), jnp.sin(jnp.pi / 6)])

        def _reference_node_vectors_cell_blocks(shift_1_1, shift_1_2, shift_2_1, shift_2_2, shift_3):
            """
            Computes the reference node vectors for the a single cell (2 blocks).
            Reference node vectors are the vectors from the bottom left corner of the cell to the nodes.
            Each shift here is a point in 2d space.
            shift_1_1: shift of the node (2)
            shift_1_2: shift of the node (4)
            shift_2_1: shift of the node (0)
            shift_2_2: shift of the node (5)
            shift_3: shift of the node (1)==(3)
            """
            a1, a2 = self.direct_basis
            block_1 = jnp.array([a1 / 2, a1 / 2 + a2 / 2, a2 / 2]) - \
                0.5*jnp.array([reference_vector_boundary_bond_1,
                              reference_vector_internal_bond,
                              reference_vector_boundary_bond_2])  # make space for bonds' length
            block_2 = jnp.array([a1 / 2 + a2 / 2, a1 + a2 / 2, a1 / 2 + a2]) + \
                0.5*jnp.array([reference_vector_internal_bond,
                               reference_vector_boundary_bond_2,
                               reference_vector_boundary_bond_1])  # make space for bonds' length
            # apply shifts
            block_1 += jnp.array([shift_2_1, shift_3, shift_1_1])
            block_2 += jnp.array([shift_3, shift_1_2, shift_2_2])
            # return a single cell
            return jnp.array([block_1, block_2])

        _reference_node_vectors_cell_blocks_mapped = vmap(
            vmap(_reference_node_vectors_cell_blocks, in_axes=(0, 0, 0, 0, 0)), in_axes=(0, 0, 0, 0, 0))

        def reference_node_vectors(
                shifts_1: jnp.ndarray = jnp.zeros((self.n1_cells+1, self.n2_cells, 2)),
                shifts_2: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells+1, 2)),
                shifts_3: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells, 2)),):

            reference_vectors = _reference_node_vectors_cell_blocks_mapped(
                shifts_1[:-1, :, :], shifts_1[1:, :, :], shifts_2[:, :-1, :], shifts_2[:, 1:, :], shifts_3)  # [n1_cells, n2_cells, bpc=2, npb=3, 2]
            # Transpose the first two axis to make sure that the reshaping reflect the row-wise numeration of the blocks.
            reference_vectors = jnp.transpose(
                reference_vectors,
                (1, 0, 2, 3, 4))  # [n2_cells, n1_cells, npb=3, bpc=2, 2]
            return reference_vectors.reshape((self.n_blocks, self.n_npb, 2))

        def centroid_node_vectors(
                shifts_1: jnp.ndarray = jnp.zeros((self.n1_cells+1, self.n2_cells, 2)),
                shifts_2: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells+1, 2)),
                shifts_3: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells, 2)),):
            """
            Computes the vectors connecting the centroid of the block to each node.
            """

            # Compute the shifts wrt to the regular kagome
            reference_vectors = reference_node_vectors(shifts_1, shifts_2, shifts_3)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        def reference_points():
            """
            Computes reference points of the blocks (i.e. positions on regular grid).
            """
            a1, a2 = self.direct_basis
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells))
            n1s, n2s = n1s.reshape((self.n_cells,)), n2s.reshape((self.n_cells,))
            cell_points = vmap(lambda n1, n2: n1 * a1 + n2 * a2, in_axes=(0, 0))(n1s, n2s)  # [n_cells, 2]
            return jnp.repeat(cell_points, self.n_bpc, axis=0)  # [n_blocks, 2]

        def block_centroids(
                shifts_1: jnp.ndarray = jnp.zeros((self.n1_cells+1, self.n2_cells, 2)),
                shifts_2: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells+1, 2)),
                shifts_3: jnp.ndarray = jnp.zeros((self.n1_cells, self.n2_cells, 2)),):
            """
            Computes blocks' centroid.
            """

            # Compute the shifts wrt the regular kagome
            reference_vectors = reference_node_vectors(shifts_1, shifts_2, shifts_3)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute blocks' centroid
            return reference_points() + centroid_shifts

        self.centroid_node_vectors = centroid_node_vectors
        self.block_centroids = block_centroids

        def translate_internal_bond(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + (n2 * self.n1_cells + n1) * n_npc

        def translate_boundary_bond1(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + jnp.array([((n2 + 1) * self.n1_cells + n1) * n_npc, (n2 * self.n1_cells + n1) * n_npc])

        def translate_boundary_bond2(node_pairs: jnp.ndarray, n1: int, n2: int):
            n_npc = self.n_npb * self.n_bpc
            return node_pairs + jnp.array([(n2 * self.n1_cells + n1 + 1) * n_npc, (n2 * self.n1_cells + n1) * n_npc])

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            internal_connectivity = jnp.array([[1, 3]])
            boundary_connectivity1 = jnp.array([[0, 5]])
            boundary_connectivity2 = jnp.array([[2, 4]])
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells))
            n1s, n2s = n1s.reshape((self.n_cells,)), n2s.reshape((self.n_cells,))
            internal_bonds = jnp.concatenate(
                vmap(translate_internal_bond, in_axes=(None, 0, 0))(internal_connectivity, n1s, n2s)
            )
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells), jnp.arange(self.n2_cells - 1))
            n1s = n1s.reshape((self.n1_cells * (self.n2_cells - 1),))
            n2s = n2s.reshape((self.n1_cells * (self.n2_cells - 1),))
            boundary_bonds1 = jnp.concatenate(
                vmap(translate_boundary_bond1, in_axes=(None, 0, 0))(boundary_connectivity1, n1s, n2s)
            )
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_cells - 1), jnp.arange(self.n2_cells))
            n1s = n1s.reshape(((self.n1_cells - 1) * self.n2_cells,))
            n2s = n2s.reshape(((self.n1_cells - 1) * self.n2_cells,))
            boundary_bonds2 = jnp.concatenate(
                vmap(translate_boundary_bond2, in_axes=(None, 0, 0))(boundary_connectivity2, n1s, n2s)
            )

            return jnp.concatenate([internal_bonds, boundary_bonds1, boundary_bonds2])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            internal_bonds = jnp.full(
                (self.n_cells, 2),
                reference_vector_internal_bond
            )
            boundary_bonds_1 = jnp.full(
                (self.n1_cells * (self.n2_cells - 1), 2),
                reference_vector_boundary_bond_1
            )
            boundary_bonds_2 = jnp.full(
                ((self.n1_cells - 1) * self.n2_cells, 2),
                reference_vector_boundary_bond_2
            )

            return jnp.concatenate([internal_bonds, boundary_bonds_1, boundary_bonds_2])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(
            self,
            shifts_1: jnp.ndarray,
            shifts_2: jnp.ndarray,
            shifts_3: jnp.ndarray,):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(shifts_1, shifts_2, shifts_3)


class QuadGeometry(LatticeGeometry):
    """
    Aperiodic lattice made of quadrangles with finite-length bonds.
    """

    def __init__(self, n1_blocks: int, n2_blocks: int, spacing: float = 1.0, bond_length: float = 0.1):
        """
        Creates a non-periodic lattice made of quadrangles with finite-length bonds.
        """

        super().__init__(n1_cells=n1_blocks, n2_cells=n2_blocks, n_bpc=1, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = self.n1_cells
        self.n2_blocks = self.n2_cells
        self.n_npb = 4
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between (`horizontal_shift`, `vertical_shift`) and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def reference_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the reference point (square grid) of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            v0 = (self.spacing - self.bond_length) / 2 * jnp.array([1., 0.])
            v0s = vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

            def _reference_node_vectors_block(n1_block, n2_block):
                return v0s + jnp.array([
                    horizontal_shift[n1_block+1, n2_block],
                    vertical_shift[n1_block, n2_block+1],
                    horizontal_shift[n1_block, n2_block],
                    vertical_shift[n1_block, n2_block],
                ])

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))

            return vmap(_reference_node_vectors_block, in_axes=(0, 0))(n1s, n2s)

        def centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the centroid of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            # Compute the shifts wrt to the square grid
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        def reference_points():
            """
            Computes reference points of the blocks (i.e. positions on the square grid).
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        def block_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """
            Computes blocks' centroid.
            """

            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute blocks' centroid
            return reference_points() + centroid_shifts

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2] for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2] for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            horizontal_bonds = jnp.full(((self.n1_blocks - 1) * self.n2_blocks, 2),
                                        self.bond_length*jnp.array([1., 0.]))
            vertical_bonds = jnp.full(((self.n2_blocks - 1) * self.n1_blocks, 2),
                                      self.bond_length*jnp.array([0., 1.]))
            

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(horizontal_shift, vertical_shift)

    def get_design_from_rotated_square(self, angle):
        """Get horizontal and vertical shifts corresponding to a rotated square geometry with the given angle.

        Args:
            angle (float): Angle of the rotated square geometry.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple of horizontal and vertical shifts.
        """

        horizontal_shifts = jnp.array([[
            (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
            jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
            jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            for n2 in range(self.n2_blocks)] for n1 in range(self.n1_blocks+1)])
        vertical_shifts = jnp.array([[
            jnp.dot(
                rotation_matrix(jnp.pi/2),
                (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
                jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            )
            for n2 in range(self.n2_blocks+1)] for n1 in range(self.n1_blocks)])

        return horizontal_shifts, vertical_shifts


class QuadGeometry_Input_general(LatticeGeometry):
    """
    Aperiodic lattice made of quadrangles with finite-length bonds.
    """

    def __init__(self, n1_blocks: int, n2_blocks: int, spacing: float = 1.0, bond_length: float = 0.1, n_source: int = 2):
        """
        Creates a non-periodic lattice made of quadrangles with finite-length bonds.
        """

        super().__init__(n1_cells=n1_blocks, n2_cells=n2_blocks, n_bpc=1, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = self.n1_cells
        self.n2_blocks = self.n2_cells
        self.n_npb = 4
        self.n_nodes = self.n_npb * self.n_blocks
        if n1_blocks < n_source*2+(n_source-1):
            raise ValueError("n1_blocks must be greater than 2*n_source")
        self.n_source = n_source

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def compute_geometry(self):
        """
        Implements mappings between (`horizontal_shift`, `vertical_shift`) and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def reference_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the reference point (square grid) of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            v0 = (self.spacing - self.bond_length) / 2 * jnp.array([1., 0.])
            v0s = vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

            def _reference_node_vectors_block(n1_block, n2_block):
                return v0s + jnp.array([
                    horizontal_shift[n1_block+1, n2_block],
                    vertical_shift[n1_block, n2_block+1],
                    horizontal_shift[n1_block, n2_block],
                    vertical_shift[n1_block, n2_block],
                ])

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))

            return vmap(_reference_node_vectors_block, in_axes=(0, 0))(n1s, n2s)

        def centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the centroid of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            # Compute the shifts wrt to the square grid
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        def reference_points():
            """
            Computes reference points of the blocks (i.e. positions on the square grid).
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        def block_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """
            Computes blocks' centroid.
            """

            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute blocks' centroid
            return reference_points() + centroid_shifts

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)

        def bond_connectivity():
            """
            Computes bonds' connectivity.
            """

            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2] for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2] for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])

            top_row = (self.n1_blocks-1)*(self.n2_blocks-1) # 16
            hori_idx = []
            for i in range(self.n_source-1):
                hori_idx.append( top_row + 3 * i + 1 )
                hori_idx.append( top_row + 3 * i + 2 )

            top_row = self.n1_blocks*(self.n2_blocks-2) # 15
            vert_idx = []
            for i in range(self.n_source-1):
                vert_idx.append( top_row + 3 * i + 2 )
            
            horizontal_bonds = jnp.delete(horizontal_bonds, jnp.array(hori_idx), axis=0)
            vertical_bonds = jnp.delete(vertical_bonds, jnp.array(vert_idx), axis=0)

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            horizontal_bonds = jnp.full(((self.n1_blocks - 1) * self.n2_blocks, 2),
                                        self.bond_length*jnp.array([1., 0.]))
            vertical_bonds = jnp.full(((self.n2_blocks - 1) * self.n1_blocks, 2),
                                      self.bond_length*jnp.array([0., 1.]))
            
            top_row = (self.n1_blocks-1)*(self.n2_blocks-1) # 16
            hori_idx = []
            for i in range(1,self.n_source):
                hori_idx.append(top_row + 2 * i)
                hori_idx.append(top_row + 2 * i + 1)

            top_row = self.n1_blocks*(self.n2_blocks-2) # 15
            vert_idx = []
            for i in range(1,self.n_source):
                vert_idx.append(top_row + 2 * i)
            
            horizontal_bonds = jnp.delete(horizontal_bonds, jnp.array(hori_idx), axis=0)
            vertical_bonds = jnp.delete(vertical_bonds, jnp.array(vert_idx), axis=0)

            return jnp.concatenate([horizontal_bonds, vertical_bonds])

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(horizontal_shift, vertical_shift)

    def get_design_from_rotated_square(self, angle):
        """Get horizontal and vertical shifts corresponding to a rotated square geometry with the given angle.

        Args:
            angle (float): Angle of the rotated square geometry.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple of horizontal and vertical shifts.
        """

        horizontal_shifts = jnp.array([[
            (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
            jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
            jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            for n2 in range(self.n2_blocks)] for n1 in range(self.n1_blocks+1)])
        vertical_shifts = jnp.array([[
            jnp.dot(
                rotation_matrix(jnp.pi/2),
                (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
                jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            )
            for n2 in range(self.n2_blocks+1)] for n1 in range(self.n1_blocks)])

        return horizontal_shifts, vertical_shifts
    

# ---------------------- revised QuadGeometry_InputSource ----------------------
class QuadGeometry_InputSource(LatticeGeometry):
    """
    Aperiodic lattice made of quadrangles with finite-length bonds + plates.
    Returns blockwise tuples for plotting compatibility.
    """

    def __init__(self, n1_blocks: int, n2_blocks: int, spacing: float = 1.0,
                 bond_length: float = 0.1, n_source: int = 2):
        super().__init__(n1_cells=n1_blocks, n2_cells=n2_blocks, n_bpc=1, direct_basis=spacing * jnp.eye(2))
        self.spacing     = spacing
        self.bond_length = bond_length
        self.n1_blocks   = self.n1_cells
        self.n2_blocks   = self.n2_cells
        self.n_quads     = self.n_blocks  # quadrilateral “blocks” from LatticeGeometry
        self.n_npb_quad  = 4

        if n1_blocks < n_source*2+(n_source-1):
            raise ValueError("n1_blocks must be greater than 2*n_source")
        self.n_source = n_source

        # --- Plates: two top (7 vertices each), one bottom (13 vertices) ---
        self.n_plate_blocks    = n_source + 1
        # self.plate_node_counts = [7, 7, 13]
        
        self.plate_node_counts = [7]*n_source + [2*self.n1_blocks+3] # bottom plate 

        # --- Totals used for flattening, bonds, and reference geometry ---
        self.block_node_counts_list = [self.n_npb_quad]*self.n_quads + self.plate_node_counts
        self.n_blocks       = self.n_quads + self.n_plate_blocks
        self.n_nodes        = int(sum(self.block_node_counts_list))
        self.block_node_counts = jnp.array(self.block_node_counts_list, dtype=int)


        # plate geometry parameters
        self.top_plate_thickness    = 7/15*self.spacing
        self.bottom_plate_thickness = 7/15*self.spacing
        self.plate_side_pad         = 0.0 # 0.5*self.spacing

        # placeholders for callables
        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    # ------------------------------ internals ------------------------------

    def _plate_vertices_from_anchors(self, anchors_xy: jnp.ndarray, thickness: float,
                                     side_pad: float, above: bool, cols_ascending: jnp.ndarray):
        """
        Build polygon above/below a set of k anchors placed along a straight edge.
        Returns (verts, anchor_poly_indices) with len(verts)=2k+3, indices map anchors
        (in ascending x-order) to their vertex indices inside verts.
        """
        k = anchors_xy.shape[0]
        # x_left,  x_right = anchors_xy[0, 0], anchors_xy[-1, 0]
        x_left,  x_right = (cols_ascending[0]-0.5)*(self.spacing), (cols_ascending[-1]+0.5)* (self.spacing)


        if above: # top plate            

            y_top, y_bot = (self.n2_blocks)*(self.spacing)+thickness, (self.n2_blocks)*(self.spacing)

            UL = jnp.array([x_left  - side_pad, y_top])
            UR = jnp.array([x_right + side_pad, y_top])
            br_outer = jnp.array([x_right + side_pad, y_bot])
            bl_outer = jnp.array([x_left  - side_pad, y_bot])
            # bottom chain (right->left): BR, anchor_k, mid, anchor_{k-1}, ..., mid, anchor_1, BL

            chain = [UL, bl_outer]
            for i in range(0, k, 1):
                chain.append(jnp.array([anchors_xy[i, 0], anchors_xy[i, 1]+self.bond_length]))
                if i < k-1:
                    xm = 0.5*(anchors_xy[i, 0] + anchors_xy[i+1, 0])
                    chain.append(jnp.array([xm, y_bot]))
            chain.extend([br_outer, UR])
            verts = jnp.array(chain)

        else: # bottom plate
            y_top, y_bot = -(self.spacing), -(self.spacing)-thickness

            bl_outer = jnp.array([x_left  - side_pad, y_top])
            br_outer = jnp.array([x_right + side_pad, y_top])
            UR = jnp.array([x_right + side_pad, y_bot])
            UL = jnp.array([x_left  - side_pad, y_bot])


            # bottom chain (right->left): BR, anchor_k, mid, anchor_{k-1}, ..., mid, anchor_1, BL
            chain = [UR, br_outer]
            for i in range(k-1, -1, -1):
                chain.append(jnp.array([anchors_xy[i, 0], anchors_xy[i, 1]-self.bond_length]))
                if i > 0:
                    xm = 0.5*(anchors_xy[i, 0] + anchors_xy[i-1, 0])
                    chain.append(jnp.array([xm, y_top]))
            chain.extend([bl_outer, UL])
            # verts = jnp.vstack([UL, UR] + chain)      # (2k+3, 2)
            verts = jnp.array(chain)
        # anchor indices (ascending anchor order) = [2k+1, 2k-1, ..., 3]
        anchor_poly_indices = jnp.array([2*k + 1 - 2*j for j in range(k)], dtype=int)
        return verts, anchor_poly_indices

    # --------------------------- geometry mappings --------------------------
    def compute_geometry(self):
        """
        Defines:
          - centroid_node_vectors(hshift, vshift) -> tuple(block_i: (n_i,2))
          - block_centroids(hshift, vshift)       -> tuple(block_i: (2,))
          - bond_connectivity()                   -> (n_bonds, 2)
          - reference_bond_vectors(hshift, vshift)-> (n_bonds, 2)
        """


        # ---- quadrilateral helpers (same as before) ----
        def reference_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            v0 = (self.spacing - self.bond_length) / 2 * jnp.array([1., 0.])
            v0s = vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(
                jnp.linspace(0., 3 * jnp.pi / 2, 4)
            )


            def _ref_block(n1_block, n2_block):
                return v0s + jnp.array([
                    horizontal_shift[n1_block+1, n2_block],
                    vertical_shift[n1_block,   n2_block+1],
                    horizontal_shift[n1_block, n2_block],
                    vertical_shift[n1_block,   n2_block],
                ])


            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_quads,)), n2s.reshape((self.n_quads,))
            return vmap(_ref_block, in_axes=(0, 0))(n1s, n2s)  # (n_quads,4,2)


        def quad_reference_points():
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_quads,)), n2s.reshape((self.n_quads,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)


        def quad_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            ref_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            shifts = vmap(polygon_centroid)(ref_vectors)
            return quad_reference_points() + shifts  # (n_quads,2)


        def quad_centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            ref_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            shifts = vmap(polygon_centroid)(ref_vectors)
            return vmap(lambda nodes, c: nodes - c, in_axes=(0, 0))(ref_vectors, shifts)  # (n_quads,4,2)


        # ---- plate builders (from anchors on top/bottom rows) ----
        def _top_plate_from_columns(centroids_q, rel_q, cols_ascending):
            top_row = self.n2_blocks - 1
            anchors = []
            for c in cols_ascending:
                b = top_row * self.n1_blocks + c
                anchors.append(centroids_q[b] + rel_q[b, 1, :])  # top node (local idx 1)
            anchors = jnp.vstack(anchors)  # (2,2)
            verts, anchor_idx = self._plate_vertices_from_anchors(
                anchors, self.top_plate_thickness, self.plate_side_pad, above=True, cols_ascending=cols_ascending
            )
            c = polygon_centroid(verts)
            rel = verts - c
            return c, rel, anchor_idx


        def _bottom_plate_from_columns(centroids_q, rel_q, cols_ascending):
            bottom_row = 0
            anchors = []
            for c in cols_ascending:
                b = bottom_row * self.n1_blocks + c
                anchors.append(centroids_q[b] + rel_q[b, 3, :])  # bottom node (local idx 3)
            anchors = jnp.vstack(anchors)  # (5,2)
            verts, anchor_idx = self._plate_vertices_from_anchors(
                anchors, self.bottom_plate_thickness, self.plate_side_pad, above=False, cols_ascending=cols_ascending
            )
            c = polygon_centroid(verts)
            rel = verts - c
            return c, rel, anchor_idx


        # ---- returned callables (TU P L E S !) ----
        def block_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            cq  = quad_centroids(horizontal_shift, vertical_shift)               # (n_quads,2)
            rel = quad_centroid_node_vectors(horizontal_shift, vertical_shift)    # (n_quads,4,2)

            cq_list = []
            for i in range(self.n_source):
                c_top, _, _ = _top_plate_from_columns(cq, rel, cols_ascending=[i*3, i*3+1])
                cq_list.append(c_top)
            c_bot,  _, _ = _bottom_plate_from_columns(cq, rel, cols_ascending=list(range(0, self.n1_blocks)))

            cq_list.append(c_bot)
            cq_list = tuple(cq_list)

            # convert to tuple of (2,)
            quad_list = tuple(cq[i] for i in range(self.n_quads))
            return quad_list + cq_list


        def centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            cq  = quad_centroids(horizontal_shift, vertical_shift)
            rel = quad_centroid_node_vectors(horizontal_shift, vertical_shift)    # (n_quads,4,2)

            rel_list = []
            for i in range(self.n_source):
                _, rel_top, _ = _top_plate_from_columns(cq, rel, cols_ascending=[i*3, i*3+1])
                rel_list.append(rel_top)
        
            _,  rel_bot,  _ = _bottom_plate_from_columns(cq, rel, cols_ascending=list(range(0, self.n1_blocks)))
            rel_list.append(rel_bot)
            rel_list = tuple(rel_list)

            # convert to tuple of (n_i,2)
            quad_list = tuple(rel[i] for i in range(self.n_quads))
            return quad_list + rel_list


        self.block_centroids = block_centroids
        self.centroid_node_vectors = centroid_node_vectors

        # ---- connectivity (unchanged, but consistent with concatenation order) ----
        def bond_connectivity():
            n1B, n2B = self.n1_blocks, self.n2_blocks
            # lattice bonds
            horizontal_bonds = jnp.array([
                [n1B * n2 * 4 + n1 * 4 + 0, n1B * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(n2B) for n1 in range(n1B - 1)
            ], dtype=int)
            vertical_bonds = jnp.array([
                [n1B * n2 * 4 + n1 * 4 + 1, n1B * (n2 + 1) * 4 + n1 * 4 + 3]
                for n2 in range(n2B - 1) for n1 in range(n1B)
            ], dtype=int)


            bonds = [horizontal_bonds, vertical_bonds]

            # offsets for plates in the flattened (quad -> topL -> topR -> bottom) order
            # offset_topL = 4 * self.n_quads
            # offset_topR = offset_topL + 7
            # offset_bot  = offset_topR + 7

            def qnode(n1, n2, local):
                b = n2 * self.n1_blocks + n1
                return 4*b + local

            # top-left (cols 0,1) anchor polygon indices [5, 3]
            offset = 4 * self.n_quads
            for i in range(self.n_source):
                for col, pidx in zip([i*3, i*3+1], [2, 4]):
                    bonds.append(jnp.array([[qnode(col, self.n2_blocks-1, 1), offset + pidx]], dtype=int))
                offset += 7  # move to next top plate offset # if we want to generalize it then we can use self.plate_node_counts

            # bottom (cols 0..4) anchor polygon indices [11,9,7,5,3]
            for col, pidx in zip(list(range(self.n1_blocks)), list(range(2*self.n1_blocks,1,-2))):
                bonds.append(jnp.array([[qnode(col, 0, 3), offset + pidx]], dtype=int))

            return jnp.vstack(bonds)

        self.bond_connectivity = bond_connectivity

        # ---- reference bond vectors computed from tuple outputs ----
        def reference_bond_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            cnv = self.centroid_node_vectors(horizontal_shift, vertical_shift)  # tuple of (n_i,2)
            bcc = self.block_centroids(horizontal_shift, vertical_shift)       # tuple of (2,)

            rel   = jnp.vstack(list(cnv))                         # (n_nodes,2)
            cents = jnp.vstack(list(bcc))                         # (n_blocks_total,2)
            cents_rep = jnp.repeat(cents, self.block_node_counts, axis=0)
            nodes_xy  = cents_rep + rel

            conn = self.bond_connectivity()
            return nodes_xy[conn[:, 1], :] - nodes_xy[conn[:, 0], :]


        self.reference_bond_vectors = reference_bond_vectors


    # ------------------------ ragged-friendly override ------------------------
    def get_reference_geometry(self, horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
        """
        Flattened node coordinates (n_nodes_total, 2) built from tuple outputs.
        Keeps geometry.get_xy_limits(...) working without touching plotting code.
        """
        try:
            cnv = self.centroid_node_vectors(horizontal_shift, vertical_shift)  # tuple
            bcc = self.block_centroids(horizontal_shift, vertical_shift)       # tuple
        except AttributeError:
            self.compute_geometry()
            cnv = self.centroid_node_vectors(horizontal_shift, vertical_shift)
            bcc = self.block_centroids(horizontal_shift, vertical_shift)


        rel   = jnp.vstack(list(cnv))                  # (n_nodes_total,2)
        cents = jnp.vstack(list(bcc))                  # (n_blocks_total,2)
        cents_rep = jnp.repeat(cents, self.block_node_counts, axis=0)
        return cents_rep + rel
    
    def get_design_from_rotated_square(self, angle):
        """Get horizontal and vertical shifts corresponding to a rotated square geometry with the given angle.


        Args:
            angle (float): Angle of the rotated square geometry.


        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple of horizontal and vertical shifts.
        """


        horizontal_shifts = jnp.array([[
            (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
            jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
            jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            for n2 in range(self.n2_blocks)] for n1 in range(self.n1_blocks+1)])
        vertical_shifts = jnp.array([[
            jnp.dot(
                rotation_matrix(jnp.pi/2),
                (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
                jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            )
            for n2 in range(self.n2_blocks+1)] for n1 in range(self.n1_blocks)])


        return horizontal_shifts, vertical_shifts


class QuadGeometry_Circle(LatticeGeometry):
    """
    QuadGeometry to cut a circle in the center of the lattice.
    """

    def __init__(self, n1_blocks: int, n2_blocks: int, spacing: float = 1.0, bond_length: float = 0.1):
        """
        Creates a non-periodic lattice made of quadrangles with finite-length bonds.
        """

        super().__init__(n1_cells=n1_blocks, n2_cells=n2_blocks, n_bpc=1, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = self.n1_cells
        self.n2_blocks = self.n2_cells
        self.n_npb = 4
        self.n_nodes = self.n_npb * self.n_blocks

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable


    def compute_geometry(self):
        """
        Implements mappings between (`horizontal_shift`, `vertical_shift`) and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def reference_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the reference point (square grid) of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            v0 = (self.spacing - self.bond_length) / 2 * jnp.array([1., 0.])
            v0s = vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

            def _reference_node_vectors_block(n1_block, n2_block):
                return v0s + jnp.array([
                    horizontal_shift[n1_block+1, n2_block],
                    vertical_shift[n1_block, n2_block+1],
                    horizontal_shift[n1_block, n2_block],
                    vertical_shift[n1_block, n2_block],
                ])

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))

            return vmap(_reference_node_vectors_block, in_axes=(0, 0))(n1s, n2s)

        def centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the centroid of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            # Compute the shifts wrt to the square grid
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        def reference_points():
            """
            Computes reference points of the blocks (i.e. positions on the square grid).
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        def block_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """
            Computes blocks' centroid.
            """

            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute blocks' centroid
            return reference_points() + centroid_shifts

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)

        # --- Compute block centroids for reference configuration ---
        h_shift = jnp.zeros((self.n1_blocks + 1, self.n2_blocks, 2))
        v_shift = jnp.zeros((self.n1_blocks, self.n2_blocks + 1, 2))
        centroids = self.block_centroids(h_shift, v_shift)

        # --- Circle definition ---
        center = jnp.mean(centroids, axis=0)
        default_radius = 0.5 * self.n1_blocks * self.spacing * 0.9  # 90% of grid half-width
        radius = getattr(self, "circle_radius", default_radius)
        tol = self.spacing * 0.5  # boundary tolerance: about half a block spacing

        # --- Classify blocks ---
        dist = jnp.linalg.norm(centroids - center, axis=1)
        inside_mask = dist < (radius - tol)
        boundary_mask = jnp.logical_and(dist >= (radius - tol), dist <= (radius + tol))
        outside_mask = dist > (radius + tol)

        blocks_inside = jnp.where(inside_mask)[0]
        blocks_boundary = jnp.where(boundary_mask)[0]
        blocks_outside = jnp.where(outside_mask)[0]

        # --- Save results ---
        self.blocks_inside = blocks_inside
        self.blocks_boundary = blocks_boundary
        self.blocks_outside = blocks_outside
        
        def bond_connectivity():
            """
            Computes bonds' connectivity for a circle-shaped structure.
            Bonds between (inside ∪ boundary) and outside blocks are removed.
            """

            # --- Build basic connectivity ---
            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
                for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])
            bonds = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)

            # --- Map node indices to block indices ---
            def bond_to_block(bond):
                return bond // 4  # 4 nodes per block

            bond_blocks = vmap(bond_to_block)(bonds)
            b0 = bond_blocks[:, 0]
            b1 = bond_blocks[:, 1]

            # --- Allowed blocks: inside ∪ boundary ---
            allowed_blocks = jnp.concatenate([self.blocks_inside, self.blocks_boundary])

            b0_allowed = jnp.isin(b0, allowed_blocks)
            b1_allowed = jnp.isin(b1, allowed_blocks)

            # Keep only bonds where both blocks are inside or boundary
            keep_mask = jnp.logical_and(b0_allowed, b1_allowed)
            filtered_bonds = bonds[keep_mask]

            return filtered_bonds

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
                for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])

            horizontal_bonds_vectors = jnp.full(((self.n1_blocks - 1) * self.n2_blocks, 2),
                                        self.bond_length*jnp.array([1., 0.]))
            vertical_bonds_vectors = jnp.full(((self.n2_blocks - 1) * self.n1_blocks, 2),
                                    self.bond_length*jnp.array([0., 1.]))
            reference_vectors = jnp.concatenate([horizontal_bonds_vectors, vertical_bonds_vectors])
            bonds = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)
            # --- Map node indices to block indices ---
            def bond_to_block(bond):
                return bond // 4  # 4 nodes per block

            bond_blocks = vmap(bond_to_block)(bonds)
            b0 = bond_blocks[:, 0]
            b1 = bond_blocks[:, 1]

            # --- Allowed blocks: inside ∪ boundary ---
            allowed_blocks = jnp.concatenate([self.blocks_inside, self.blocks_boundary])

            b0_allowed = jnp.isin(b0, allowed_blocks)
            b1_allowed = jnp.isin(b1, allowed_blocks)

            # Keep only bonds where both blocks are inside or boundary
            keep_mask = jnp.logical_and(b0_allowed, b1_allowed)
            filtered_reference_vectors = reference_vectors[keep_mask]

            return filtered_reference_vectors

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(horizontal_shift, vertical_shift)

    def get_design_from_rotated_square(self, angle):
        """Get horizontal and vertical shifts corresponding to a rotated square geometry with the given angle.

        Args:
            angle (float): Angle of the rotated square geometry.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple of horizontal and vertical shifts.
        """

        horizontal_shifts = jnp.array([[
            (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
            jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
            jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            for n2 in range(self.n2_blocks)] for n1 in range(self.n1_blocks+1)])
        vertical_shifts = jnp.array([[
            jnp.dot(
                rotation_matrix(jnp.pi/2),
                (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
                jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            )
            for n2 in range(self.n2_blocks+1)] for n1 in range(self.n1_blocks)])

        return horizontal_shifts, vertical_shifts
    

class QuadGeometry_Circle_Input(LatticeGeometry):
    """
    Circular structure and remove bonds between boundary blocks to make separate input sources.
    """

    def __init__(self, n1_blocks: int, n2_blocks: int, spacing: float = 1.0, bond_length: float = 0.1):
        """
        Creates a non-periodic lattice made of quadrangles with finite-length bonds.
        """

        super().__init__(n1_cells=n1_blocks, n2_cells=n2_blocks, n_bpc=1, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = self.n1_cells
        self.n2_blocks = self.n2_cells
        self.n_npb = 4
        self.n_nodes = self.n_npb * self.n_blocks
        self.n_piece = 6

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable


    def compute_geometry(self):
        """
        Implements mappings between (`horizontal_shift`, `vertical_shift`) and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def reference_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the reference point (square grid) of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            v0 = (self.spacing - self.bond_length) / 2 * jnp.array([1., 0.])
            v0s = vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

            def _reference_node_vectors_block(n1_block, n2_block):
                return v0s + jnp.array([
                    horizontal_shift[n1_block+1, n2_block],
                    vertical_shift[n1_block, n2_block+1],
                    horizontal_shift[n1_block, n2_block],
                    vertical_shift[n1_block, n2_block],
                ])

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))

            return vmap(_reference_node_vectors_block, in_axes=(0, 0))(n1s, n2s)

        def centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the centroid of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            # Compute the shifts wrt to the square grid
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute node positions relative to centroids
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        def reference_points():
            """
            Computes reference points of the blocks (i.e. positions on the square grid).
            """

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_blocks,)), n2s.reshape((self.n_blocks,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        def block_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """
            Computes blocks' centroid.
            """

            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            # Compute blocks' centroid
            return reference_points() + centroid_shifts

        self.centroid_node_vectors = jit(centroid_node_vectors)
        self.block_centroids = jit(block_centroids)


        horizontal_bonds = jnp.array([
            [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
            for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
        ])
        vertical_bonds = jnp.array([
            [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
            for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
        ])
        _bond_connectivity = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)

        # bond connecting between boundary blocks / n_piece will be variable later
        angle_cut = 360 / self.n_piece

        # --- Compute block centroids for reference configuration ---
        h_shift = jnp.zeros((self.n1_blocks + 1, self.n2_blocks, 2))
        v_shift = jnp.zeros((self.n1_blocks, self.n2_blocks + 1, 2))
        centroids = block_centroids(h_shift, v_shift)
        # --- Circle definition ---
        center = jnp.mean(centroids, axis=0)
        radius = 0.5 * self.n1_blocks * self.spacing * 0.8  # 80% of grid half-width
        tol = self.spacing * 0.5  # boundary tolerance: about half a block spacing

        dist_vec = centroids - center
        # --- Classify blocks ---
        dist = jnp.linalg.norm(dist_vec, axis=1)
        inside_mask = dist < (radius - tol)
        blocks_inside = jnp.where(inside_mask)[0]
        _bond_connectivity_idx = _bond_connectivity // 4 # block indices that are connected
        connection = jnp.isin(_bond_connectivity_idx, blocks_inside)
        check_connection = jnp.sum( connection, axis=1)

        boundary = _bond_connectivity_idx[check_connection==1] # blocks that have connection with blocks inside
        connection_boundary = connection[check_connection==1]
        blocks_boundary = jnp.unique( boundary[~connection_boundary] )

        dist_vec_bound = dist_vec[blocks_boundary]
        angle_bound = jnp.atan2( dist_vec_bound[:,1], dist_vec_bound[:,0] ) * 180./jnp.pi
        bond_boundary_and_outside = _bond_connectivity_idx[check_connection == 0]

        boundary_separation = []
        for count in range(self.n_piece):

            ang_max = - angle_cut * count + 180.
            ang_min = - angle_cut * (count+1) + 180.
            
            mask_angle = jnp.logical_and( angle_bound > ang_min, angle_bound <= ang_max )
            bound_blocks = blocks_boundary[mask_angle] # boundary blocks corresponding the angle interval
            # check bonds that are connected to boundary blocks
            bond_partial = bond_boundary_and_outside[jnp.sum( jnp.isin(bond_boundary_and_outside, bound_blocks), axis=1) > 0]
            
            bridge_block = []
            for i in range( bound_blocks.shape[0] ):

                check = jnp.isin(bond_partial, bound_blocks[i])
                mask = jnp.ones(bound_blocks.shape[0], dtype=bool).at[i].set(False)
                blocks_connected_to_boundary = bond_partial[jnp.sum(check,axis=1)==1]
                check_two_blocks = jnp.isin( blocks_connected_to_boundary, bound_blocks[mask])
                connection_temp = jnp.sum( check_two_blocks ) # find connect
                
                if connection_temp == 0: # if boundary block is isolated
                    # include index of block that connects to the isolated boundary block
                    indices = jnp.where(blocks_connected_to_boundary.flatten() == bound_blocks[i])[0]
                    adjacent_blocks = jnp.delete(blocks_connected_to_boundary.flatten(), indices[0])
                    for search_idx in adjacent_blocks:
                        check = jnp.isin(bond_partial, search_idx) # figure out whether search index is in connections related to boundary blocks
                        blocks_connected_to_boundary = bond_partial[jnp.sum(check,axis=1)==1]
                        check_two_blocks = jnp.isin( blocks_connected_to_boundary, bound_blocks[mask])

                        if jnp.sum( check_two_blocks ) > 0:
                            bridge_block.append( search_idx )

            if len(bridge_block) == 0:
                boundary_separation.append( bound_blocks )
            else:
                boundary_separation.append( jnp.concatenate( ( bound_blocks, jnp.array(bridge_block) ) ) )

        blocks_boundary = jnp.concatenate( boundary_separation )
        blocks_outside = jnp.arange( self.n1_blocks * self.n2_blocks )
        blocks_outside = jnp.delete( blocks_outside, jnp.concatenate( (blocks_inside, blocks_boundary) ) )

        self.blocks_inside = blocks_inside
        self.blocks_boundary = blocks_boundary
        self.blocks_outside = blocks_outside
        self.boundary_separation = boundary_separation
        
        def bond_connectivity():
            """
            Computes bonds' connectivity for a circle-shaped structure.
            Bonds between (inside ∪ boundary) and outside blocks are removed.
            """

            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
                for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])
            bonds = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)

            # --- Map node indices to block indices ---
            def bond_to_block(bond):
                return bond // 4  # 4 nodes per block

            bond_blocks = vmap(bond_to_block)(bonds)
            b0 = bond_blocks[:, 0]
            b1 = bond_blocks[:, 1]
            
            # --- Allowed blocks: inside ∪ boundary ---
            allowed_blocks = jnp.concatenate([self.blocks_inside, self.blocks_boundary])

            b0_allowed = jnp.isin(b0, allowed_blocks)
            b1_allowed = jnp.isin(b1, allowed_blocks)

            # Keep only bonds where both blocks are inside or boundary
            keep_mask = jnp.logical_and(b0_allowed, b1_allowed)
            filtered_bonds = bonds[keep_mask]

            # disconnect the bonds between different input source
            for count in range(self.n_piece):
                filtered_b = vmap(bond_to_block)(filtered_bonds)
                input_source_1 = jnp.sum( jnp.isin( filtered_b, self.boundary_separation[count-1] ), axis=1 ) # (N,2)
                input_source_2 = jnp.sum( jnp.isin( filtered_b, self.boundary_separation[count] ), axis=1 )
                
                mask = jnp.logical_and(input_source_1, input_source_2)
                filtered_bonds = filtered_bonds[~mask].copy()

            return filtered_bonds

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors():
            """
            Computes the reference configuration of the bonds.
            """

            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
                for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])

            horizontal_bonds_vectors = jnp.full(((self.n1_blocks - 1) * self.n2_blocks, 2),
                                        self.bond_length*jnp.array([1., 0.]))
            vertical_bonds_vectors = jnp.full(((self.n2_blocks - 1) * self.n1_blocks, 2),
                                    self.bond_length*jnp.array([0., 1.]))
            reference_vectors = jnp.concatenate([horizontal_bonds_vectors, vertical_bonds_vectors])

            bonds = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)
            # --- Map node indices to block indices ---
            def bond_to_block(bond):
                return bond // 4  # 4 nodes per block

            bond_blocks = vmap(bond_to_block)(bonds)
            b0 = bond_blocks[:, 0]
            b1 = bond_blocks[:, 1]

            # --- Allowed blocks: inside ∪ boundary ---
            allowed_blocks = jnp.concatenate([self.blocks_inside, self.blocks_boundary])

            b0_allowed = jnp.isin(b0, allowed_blocks)
            b1_allowed = jnp.isin(b1, allowed_blocks)

            # Keep only bonds where both blocks are inside or boundary
            keep_mask = jnp.logical_and(b0_allowed, b1_allowed)
            filtered_bonds = bonds[keep_mask]
            filtered_reference_vectors = reference_vectors[keep_mask]

            # disconnect the bonds between different input source
            filtered_b = vmap(bond_to_block)(filtered_bonds)
            for count in range(self.n_piece):
                
                input_source_1 = jnp.sum( jnp.isin( filtered_b, self.boundary_separation[count-1] ), axis=1 ) # (N,2)
                input_source_2 = jnp.sum( jnp.isin( filtered_b, self.boundary_separation[count] ), axis=1 )
                
                if count == 0:
                    mask = jnp.logical_and(input_source_1, input_source_2)
                else:
                    mask += jnp.logical_and(input_source_1, input_source_2)

            filtered_reference_vectors = filtered_reference_vectors[~mask].copy()
            
            return filtered_reference_vectors

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(horizontal_shift, vertical_shift)

    def get_design_from_rotated_square(self, angle):
        """Get horizontal and vertical shifts corresponding to a rotated square geometry with the given angle.

        Args:
            angle (float): Angle of the rotated square geometry.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple of horizontal and vertical shifts.
        """

        horizontal_shifts = jnp.array([[
            (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
            jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
            jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            for n2 in range(self.n2_blocks)] for n1 in range(self.n1_blocks+1)])
        vertical_shifts = jnp.array([[
            jnp.dot(
                rotation_matrix(jnp.pi/2),
                (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
                jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            )
            for n2 in range(self.n2_blocks+1)] for n1 in range(self.n1_blocks)])

        return horizontal_shifts, vertical_shifts
    

class QuadGeometry_Circle_InputSource_vis(LatticeGeometry):
    """
    Aperiodic lattice made of quadrangles with finite-length bonds.
    """

    def __init__(self, n1_blocks: int, n2_blocks: int, spacing: float = 1.0, bond_length: float = 0.1, n_piece=8, n_outer=4):
        """
        Creates a non-periodic lattice made of quadrangles with finite-length bonds.
        """

        super().__init__(n1_cells=n1_blocks, n2_cells=n2_blocks, n_bpc=1, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = self.n1_cells
        self.n2_blocks = self.n2_cells
        self.n_npb = 4
        self.n_quads = self.n_blocks  # quadrilateral blocks from LatticeGeometry
        self.n_piece = n_piece
        self.n_quads_org = self.n_blocks
        self.n_outer = n_outer

        # Shell blocks: one per interval (n_piece total)
        # Node counts will be determined dynamically based on connection vertices
        self.n_shell_blocks = n_piece
        self.shell_node_counts = []  # Will be set in compute_geometry

        # Total blocks: quads (only inside blocks) + shell blocks
        # Note: boundary quads are replaced by shell blocks
        self.block_node_counts_list = []  # Will be set in compute_geometry
        # self.n_blocks_total = 0  # Will be set in compute_geometry
        self.n_nodes_total = 0  # Will be set in compute_geometry
        self.block_node_counts = None  # Will be set in compute_geometry

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def _shell_vertices_from_connections(self, connection_vertices: jnp.ndarray, interval_vertices: jnp.ndarray, order: int):
        """
        Build shell polygon with zigzag inner edge and smooth outer edge.
        Connection vertices are where inner blocks connect to the shell.
        Interval vertices are added between connection vertices for zigzag inner edge.
        Outer edge is smooth with up to 3 additional vertices for circular shape.

        Args:
            connection_vertices: (n_conn, 2) vertices on inner blocks for connection
            interval_vertices: (n_interval, 2) vertices between connection vertices (zigzag inner edge)

        Returns:
            verts: (n_conn + n_interval + n_outer, 2) shell vertices ordered counter-clockwise
            connection_indices: (n_conn,) indices of connection vertices in verts
        """
        n_conn = connection_vertices.shape[0]
        n_interval = interval_vertices.shape[0] # n_interval = n_conn-1

        if n_conn == 0:
                # Fallback: create minimal shell
            return jnp.zeros((3, 2)), jnp.array([], dtype=int)

        # Order connection vertices by angle around center
        # center = jnp.mean(connection_vertices, axis=0)
        center = self._center
        conn_dist_vec = connection_vertices - center
        conn_angles = jnp.atan2(conn_dist_vec[:, 1], conn_dist_vec[:, 0])

        # Add the exception treatment

        # conn_sorted_idx = jnp.argsort(conn_angles)
        # conn_sorted = connection_vertices[conn_sorted_idx]
        conn_sorted = connection_vertices # keep the order
        conn_sorted_idx = jnp.arange(len(connection_vertices))

        # Build inner edge: connection vertices + interval vertices (zigzag pattern)
        # Interleave interval vertices between connection vertices
        inner_edge_verts = []
        conn_indices_inner = []

        # Match interval vertices to connection pairs
        # Interval vertices are created between consecutive connection vertices
        count = 0 # vertices are aligned in counter-clockwise direction
        for i in range(n_conn):
            # Add connection vertex
            inner_edge_verts.append(conn_sorted[i]) # conned_sort
            conn_indices_inner.append(count)
            count += 1
            # Add corresponding interval vertex after this connection vertex
            # Interval vertices are ordered to match connection pairs
            if n_interval > 0 and i < n_interval:
                inner_edge_verts.append(interval_vertices[i])
                count += 1

        inner_edge_verts = jnp.array(inner_edge_verts)

        # Build smooth outer edge with up to 3 additional vertices
        # Compute average radius of inner edge
        inner_radius = 0.5 * self.n1_blocks * self.spacing * 0.8
        outer_radius = inner_radius + 0.6 * self.n1_blocks * self.spacing / 15. # 16.125
        # outer_radius = inner_radius + self.spacing # 16.125

        # Create outer edge vertices (smooth, circular)
        # Use up to 4 vertices to approximate a smooth circular arc
        # n_outer = min(3, max(1, n_conn // 2))  # Up to 3 outer vertices, at least 1 if we have connections
        n_outer = self.n_outer
        outer_edge_verts = []
        angle_cut = 2. * jnp.pi / self.n_piece
        # desired_angle = angle_cut - 5.0 * jnp.pi / 180. # angle of arc
        buffer_angle = 5.0 * jnp.pi / 180.
        desired_angle = angle_cut - buffer_angle


        if n_outer > 0 and n_conn > 0:
            # Compute angles for outer vertices (distributed along the arc)
            # first_angle = conn_angles[conn_sorted_idx[-1]]
            # last_angle = conn_angles[conn_sorted_idx[0]]
            first_angle = - angle_cut * (order+1) + jnp.pi + buffer_angle / 2. + self._bias
            last_angle = - angle_cut * order + jnp.pi - buffer_angle /2. + self._bias
            # Handle wrap-around: if the arc crosses -π/π boundary
            last_angle = jnp.where( last_angle < first_angle, last_angle+2*jnp.pi, last_angle)
            # if last_angle < first_angle:
            #     last_angle += 2 * jnp.pi
            # first_angle = (first_angle+last_angle)/2.0 - desired_angle/2.0
            # last_angle = first_angle + desired_angle

            # Create outer vertices along smooth circular arc
            for i in range(n_outer):
                # Distribute angles evenly between first and last connection
                t = (i) / (n_outer-1)
                angle = first_angle + t * desired_angle
                # angle = last_angle + t * (first_angle - last_angle) # start from last point to first point
                # Use angle directly (already in correct range after wrap-around handling)
                if i == 0 :
                    outer_vert = center + inner_radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
                    outer_edge_verts.append(outer_vert)

                outer_vert = center + outer_radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
                outer_edge_verts.append(outer_vert) # total number of vertices of outer verts is n_outer+2

                if i == n_outer-1:
                    outer_vert = center + inner_radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
                    outer_edge_verts.append(outer_vert)

        outer_edge_verts = jnp.array(outer_edge_verts) if len(outer_edge_verts) > 0 else jnp.zeros((0, 2))

        # Build complete shell: inner edge (zigzag) -> outer edge (smooth)
        # The polygon goes: connection vertices and interval vertices (zigzag inner edge),
        # then outer vertices (smooth outer edge), forming a closed polygon
        shell_verts = jnp.vstack([inner_edge_verts, outer_edge_verts])

        # Map connection indices to final shell vertex indices
        connection_indices = jnp.array(conn_indices_inner, dtype=int)

        return shell_verts, connection_indices

    def compute_geometry(self):
        """
        Implements mappings between (`horizontal_shift`, `vertical_shift`) and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def reference_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the reference point (square grid) of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            v0 = (self.spacing - self.bond_length) / 2 * jnp.array([1., 0.])
            v0s = vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

            def _reference_node_vectors_block(n1_block, n2_block):
                return v0s + jnp.array([
                    horizontal_shift[n1_block+1, n2_block],
                    vertical_shift[n1_block, n2_block+1],
                    horizontal_shift[n1_block, n2_block],
                    vertical_shift[n1_block, n2_block],
                ])

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_quads_org,)), n2s.reshape((self.n_quads_org,))

            return vmap(_reference_node_vectors_block, in_axes=(0, 0))(n1s, n2s)

        def quad_reference_points():
            """Computes reference points of the quad blocks."""
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_quads_org,)), n2s.reshape((self.n_quads_org,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        def quad_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes quad blocks' centroids."""
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            return quad_reference_points() + centroid_shifts

        def quad_centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting quad block centroids to nodes."""
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        # --- Setup bond connectivity for finding boundaries ---
        horizontal_bonds = jnp.array([
            [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
            for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
        ])
        vertical_bonds = jnp.array([
            [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
            for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
        ])
        _bond_connectivity = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)
        self._bond_connectivity = _bond_connectivity

        # bond connecting between boundary blocks / n_piece will be variable later
        angle_cut = 360.0 / self.n_piece

        # --- Compute block centroids for reference configuration ---
        h_shift = jnp.zeros((self.n1_blocks + 1, self.n2_blocks, 2))
        v_shift = jnp.zeros((self.n1_blocks, self.n2_blocks + 1, 2))
        centroids = quad_centroids(h_shift, v_shift)
        quad_rel = quad_centroid_node_vectors(h_shift, v_shift)
        # --- Circle definition ---
        center = jnp.mean(centroids, axis=0)
        radius = 0.5 * self.n1_blocks * self.spacing * 0.8  # 90% of grid half-width
        tol = self.spacing * 0.5  # boundary tolerance: about half a block spacing

        dist_vec = centroids - center
        # --- Classify blocks ---
        dist = jnp.linalg.norm(dist_vec, axis=1)
        inside_mask = dist < (radius - tol)
        blocks_inside = jnp.where(inside_mask)[0]
        _bond_connectivity_idx = _bond_connectivity // 4 # block indices that are connected
        connection = jnp.isin(_bond_connectivity_idx, blocks_inside)
        check_connection = jnp.sum( connection, axis=1)

        boundary = _bond_connectivity_idx[check_connection==1] # blocks that have connection with blocks inside
        connection_boundary = connection[check_connection==1]
        blocks_boundary = jnp.unique( boundary[~connection_boundary] )

        dist_vec_bound = dist_vec[blocks_boundary]
        angle_bound = jnp.atan2( dist_vec_bound[:,1], dist_vec_bound[:,0] ) * 180./jnp.pi
        bond_boundary_and_outside = _bond_connectivity_idx[check_connection == 0]

        boundary_separation = []
        for count in range(self.n_piece):

            ang_max = - angle_cut * count + 180.
            ang_min = - angle_cut * (count+1) + 180.

            mask_angle = jnp.logical_and( angle_bound > ang_min, angle_bound <= ang_max )
            bound_blocks = blocks_boundary[mask_angle] # boundary blocks corresponding the angle interval
            # check bonds that are connected to boundary blocks
            bond_partial = bond_boundary_and_outside[jnp.sum( jnp.isin(bond_boundary_and_outside, bound_blocks), axis=1) > 0]

            bridge_block = []

            if len(bridge_block) == 0:
                boundary_separation.append( bound_blocks )
            else:
                boundary_separation.append( jnp.concatenate( ( bound_blocks, jnp.array(bridge_block) ) ) )

        blocks_boundary = jnp.concatenate( boundary_separation )
        blocks_outside = jnp.arange( self.n1_blocks * self.n2_blocks )
        blocks_outside = jnp.delete( blocks_outside, jnp.concatenate( (blocks_inside, blocks_boundary) ) )
        self._center = center
        self._bias = 0.0
        self.blocks_inside = blocks_inside
        self.blocks_boundary = blocks_boundary
        self.blocks_outside = blocks_outside
        self.boundary_separation = boundary_separation


        # --- Find connection vertices and create shell blocks ---
        # Compute node positions (centroid + relative)
        def get_node_pos(block_idx, node_idx):
            return centroids[block_idx] + quad_rel[block_idx, node_idx]

        # For each interval, find connection vertices and interval vertices
        shell_connection_info = []  # List of (connection_vertices, interval_vertices, connection_node_info, conn_indices)
        shell_node_counts_list = []

        for count in range(self.n_piece):
            bound_blocks = boundary_separation[count]

            # Find bonds between inner blocks and boundary blocks in this interval
            bonds_inner_to_boundary = []
            # Vectorized version using jnp functionality
            # Compute block indices and node indices for each bond
            b0 = _bond_connectivity[:, 0] // 4
            b1 = _bond_connectivity[:, 1] // 4
            node0 = _bond_connectivity[:, 0] % 4
            node1 = _bond_connectivity[:, 1] % 4
            # Determine mask for which bonds connect inner to boundary, either direction
            mask0 = jnp.logical_and(
                jnp.isin(b0, blocks_inside),
                jnp.isin(b1, bound_blocks)
            )
            mask1 = jnp.logical_and(
                jnp.isin(b1, blocks_inside),
                jnp.isin(b0, bound_blocks)
            )
            # Get indices of matches for both cases
            idx0 = jnp.where(mask0)[0]
            idx1 = jnp.where(mask1)[0]

            # For mask0: (b0 in inside, b1 in boundary)
            bonds0 = jnp.stack([b0[idx0], node0[idx0], b1[idx0], node1[idx0]], axis=1)

            # For mask1: (b1 in inside, b0 in boundary) (order reversed)
            bonds1 = jnp.stack([b1[idx1], node1[idx1], b0[idx1], node0[idx1]], axis=1)

            # Combine, convert to list of tuples for expected format
            bonds_inner_to_boundary = [tuple(x) for x in jnp.concatenate([bonds0, bonds1], axis=0)] # axis 0 blocks inside, axis 3 blocks boundary

            # Get connection vertices (on inner blocks)
            connection_vertices = []
            connection_node_info = []  # (block_idx, node_idx) for each connection
            for b_inner, n_inner, b_bound, n_bound in bonds_inner_to_boundary:
                conn_vert = get_node_pos(b_bound, n_bound)
                connection_vertices.append(conn_vert)
                connection_node_info.append((b_bound, n_bound, b_inner, n_inner))

            connection_vertices = jnp.array(connection_vertices) if len(connection_vertices) > 0 else jnp.zeros((0, 2))

            # Generate interval vertices between connection vertices for zigzag inner edge
            # Note: interval_vertices will be created based on sorted connection vertices
            # The _shell_vertices_from_connections method will sort connection_vertices again,
            # so we create interval_vertices here but they'll be properly matched in that method
            interval_vertices = []
            if connection_vertices.shape[0] > 1:
                # Order connection vertices by angle around center (same as in _shell_vertices_from_connections)
                conn_dist_vec = connection_vertices - center
                conn_angles = jnp.atan2(conn_dist_vec[:, 1], conn_dist_vec[:, 0])
                conn_sorted_idx = jnp.argsort(conn_angles, descending=True)
                conn_sorted = connection_vertices[conn_sorted_idx]
                connection_node_info = jnp.array(connection_node_info)[conn_sorted_idx]


                # Create interval vertices between consecutive connection vertices
                # These create the zigzag pattern on the inner edge
                def compute_interval_vert(curr_conn, next_conn):
                    mid_point = 0.5 * (curr_conn + next_conn)
                    direction = next_conn - curr_conn
                    perp_dir = jnp.array([-direction[1], direction[0]])
                    perp_dir = perp_dir / (jnp.linalg.norm(perp_dir) + 1e-8)
                    zigzag_offset = self.spacing * 0.3
                    zigzag_sign = 1
                    return mid_point + zigzag_sign * zigzag_offset * perp_dir

                # We want intervals between consecutive pairs: (conn_sorted[0], conn_sorted[1]), ...
                if conn_sorted.shape[0] > 1:
                    curr_conns = conn_sorted[:-1]
                    next_conns = conn_sorted[1:]
                    interval_vertices = vmap(compute_interval_vert)(curr_conns, next_conns)
                    # interval_vertices = [v for v in interval_vertices]

                # Store interval_vertices in the same sorted order as connection_vertices
                # (They're already in the correct order since we created them based on sorted connections)

            interval_vertices = interval_vertices if len(interval_vertices) > 0 else jnp.zeros((0, 2))

            # Create shell vertices
            if connection_vertices.shape[0] > 0 or interval_vertices.shape[0] > 0:

                if count == 0:
                    conn_dist_vec = conn_sorted - center
                    conn_angles = jnp.atan2(conn_dist_vec[:, 1], conn_dist_vec[:, 0])
                    first_angle = conn_angles[-1]
                    last_angle =  conn_angles[0] # should be larger than first_angle

                    last_angle = jnp.where( last_angle < first_angle, last_angle+2*jnp.pi, last_angle)
                    mid_angle = ( first_angle + last_angle )/2.
                    mid_angle_hard = - 0.5 * angle_cut * jnp.pi / 180. + jnp.pi
                    self._bias =  mid_angle - mid_angle_hard # update bias
                    print( "bias : ", self._bias * 180./jnp.pi)

                shell_verts, conn_indices = self._shell_vertices_from_connections(conn_sorted, interval_vertices, order=count)
                n_shell_verts = shell_verts.shape[0]
            else:
                # Fallback: create minimal shell
                shell_verts = jnp.zeros((3, 2))
                conn_sorted = connection_vertices
                conn_indices = jnp.array([], dtype=int)
                n_shell_verts = 4

            shell_connection_info.append((shell_verts, conn_sorted, interval_vertices, connection_node_info, conn_indices))
            shell_node_counts_list.append(n_shell_verts)

        self.shell_node_counts = shell_node_counts_list
        self.shell_connection_info = shell_connection_info

        # Update block node counts
        self.block_node_counts_list = [4] * len(blocks_inside) + shell_node_counts_list
        self.n_blocks = len(blocks_inside) + self.n_shell_blocks
        self.n_nodes_total = sum(self.block_node_counts_list)
        self.block_node_counts = jnp.array(self.block_node_counts_list, dtype=int)
        self.n_quads = len(blocks_inside)



        # --- Define geometry functions that include shell blocks ---
        def block_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes blocks' centroids including shell blocks. Returns tuple."""

            quad_cents = quad_centroids(horizontal_shift, vertical_shift)
            quad_rel = quad_centroid_node_vectors(horizontal_shift, vertical_shift)

            # Get centroids for inner quads only
            # inner_centroids = [quad_cents[i] for i in blocks_inside]
            inner_centroids = quad_cents[self.blocks_inside]

            def get_node_pos(block_idx, node_idx):
                return quad_cents[block_idx] + quad_rel[block_idx, node_idx]
            # Compute shell block centroids
            shell_centroids = []
            shell_connection_info = []
            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_vertices = conn_info[1]
                interval_vertices = conn_info[2]
                connection_node_info = conn_info[3]
                conn_indices = conn_info[4]

                # # Recompute shell vertices for current configuration
                if connection_vertices.shape[0] > 0 or interval_vertices.shape[0] > 0:
                    b_bound, n_bound = connection_node_info[:,0], connection_node_info[:,1]

                    updated_conn_verts = vmap(get_node_pos, in_axes=(0,0))(b_bound, n_bound)
                    # conn_dist_vec = connection_vertices - self._center
                    # Create interval vertices between consecutive connection vertices
                    # These create the zigzag pattern on the inner edge
                    def compute_interval_vert(curr_conn, next_conn):
                        mid_point = 0.5 * (curr_conn + next_conn)
                        direction = next_conn - curr_conn
                        perp_dir = jnp.array([-direction[1], direction[0]])
                        perp_dir = perp_dir / (jnp.linalg.norm(perp_dir) + 1e-8)
                        zigzag_offset = self.spacing * 0.3
                        zigzag_sign = 1
                        return mid_point + zigzag_sign * zigzag_offset * perp_dir
                    # We want intervals between consecutive pairs: (conn_sorted[0], conn_sorted[1]), ...
                    curr_conns = updated_conn_verts[:-1]
                    next_conns = updated_conn_verts[1:]
                    interval_vertices = vmap(compute_interval_vert)(curr_conns, next_conns) # updated interval_vertices

                    # For interval vertices, we need to recompute from boundary blocks
                    # For now, use the reference positions
                    shell_verts, _ = self._shell_vertices_from_connections(updated_conn_verts, interval_vertices, order=count)
                    shell_centroid = polygon_centroid(shell_verts)
                else:
                    # shell_verts = jnp.zeros((0, 2))
                    # updated_conn_verts = connection_vertices
                    shell_centroid = jnp.array([0., 0.])

                shell_centroids.append(shell_centroid)
            #     shell_connection_info.append((shell_verts, updated_conn_verts, interval_vertices, connection_node_info, conn_indices))
            # self.shell_connection_info = shell_connection_info
            # Return as tuple
            return tuple(inner_centroids) + tuple(shell_centroids)

        def centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting block centroids to nodes. Returns tuple."""
            quad_cents = quad_centroids(horizontal_shift, vertical_shift)
            quad_rel = quad_centroid_node_vectors(horizontal_shift, vertical_shift)

            # Get node vectors for inner quads only
            # inner_node_vectors = [quad_rel[i] for i in blocks_inside]
            inner_node_vectors = quad_rel[self.blocks_inside]

            def get_node_pos(block_idx, node_idx):
                return quad_cents[block_idx] + quad_rel[block_idx, node_idx]

            # Compute shell block node vectors
            shell_node_vectors = []
            shell_connection_info = []
            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_vertices = conn_info[1]
                interval_vertices = conn_info[2]
                connection_node_info = conn_info[3]
                conn_indices = conn_info[4]

                # # Recompute shell vertices for current configuration
                if connection_vertices.shape[0] > 0 or interval_vertices.shape[0] > 0:
                    b_bound, n_bound = connection_node_info[:,0], connection_node_info[:,1]
                    updated_conn_verts = vmap(get_node_pos, in_axes=(0,0))(b_bound, n_bound)
                    # conn_dist_vec = connection_vertices - self._center
                    # Create interval vertices between consecutive connection vertices
                    # These create the zigzag pattern on the inner edge
                    def compute_interval_vert(curr_conn, next_conn):
                        mid_point = 0.5 * (curr_conn + next_conn)
                        direction = next_conn - curr_conn
                        perp_dir = jnp.array([-direction[1], direction[0]])
                        perp_dir = perp_dir / (jnp.linalg.norm(perp_dir) + 1e-8)
                        zigzag_offset = self.spacing * 0.3
                        zigzag_sign = 1
                        return mid_point + zigzag_sign * zigzag_offset * perp_dir
                    # We want intervals between consecutive pairs: (conn_sorted[0], conn_sorted[1]), ...
                    curr_conns = updated_conn_verts[:-1]
                    next_conns = updated_conn_verts[1:]
                    interval_vertices = vmap(compute_interval_vert)(curr_conns, next_conns) # updated interval_vertices

                    # For interval vertices, we need to recompute from boundary blocks
                    # For now, use the reference positions
                    shell_verts, _ = self._shell_vertices_from_connections(updated_conn_verts, interval_vertices, order=count)
                    shell_centroid = polygon_centroid(shell_verts)
                    shell_rel = shell_verts - shell_centroid
                else:
                    # shell_verts = jnp.zeros((0, 2))
                    # updated_conn_verts = connection_vertices
                    shell_rel = jnp.zeros((3, 2))

                shell_node_vectors.append(shell_rel)

            #     shell_connection_info.append((shell_verts, updated_conn_verts, interval_vertices, connection_node_info, conn_indices))
            # self.shell_connection_info = shell_connection_info
            return tuple(inner_node_vectors) + tuple(shell_node_vectors)


        def block_centroids_node(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting block centroids to nodes. Returns tuple."""
            quad_cents = quad_centroids(horizontal_shift, vertical_shift)
            quad_rel = quad_centroid_node_vectors(horizontal_shift, vertical_shift)

            # Get node vectors for inner quads only
            # inner_node_vectors = [quad_rel[i] for i in blocks_inside]
            inner_node_vectors = quad_rel[self.blocks_inside]
            inner_centroids = quad_cents[self.blocks_inside]

            def get_node_pos(block_idx, node_idx):
                return quad_cents[block_idx] + quad_rel[block_idx, node_idx]
            # Compute shell block node vectors
            shell_node_vectors = []
            shell_centroids = []
            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_vertices = conn_info[1]
                interval_vertices = conn_info[2]

                # # Recompute shell vertices for current configuration
                if connection_vertices.shape[0] > 0 or interval_vertices.shape[0] > 0:
                    b_bound, n_bound = connection_node_info[:,0], connection_node_info[:,1]
                    updated_conn_verts = vmap(get_node_pos, in_axes=(0,0))(b_bound, n_bound)
                    # conn_dist_vec = connection_vertices - self._center
                    # Create interval vertices between consecutive connection vertices
                    # These create the zigzag pattern on the inner edge
                    def compute_interval_vert(curr_conn, next_conn):
                        mid_point = 0.5 * (curr_conn + next_conn)
                        direction = next_conn - curr_conn
                        perp_dir = jnp.array([-direction[1], direction[0]])
                        perp_dir = perp_dir / (jnp.linalg.norm(perp_dir) + 1e-8)
                        zigzag_offset = self.spacing * 0.3
                        zigzag_sign = -1
                        return mid_point + zigzag_sign * zigzag_offset * perp_dir
                    # We want intervals between consecutive pairs: (conn_sorted[0], conn_sorted[1]), ...
                    curr_conns = updated_conn_verts[:-1]
                    next_conns = updated_conn_verts[1:]
                    interval_vertices = vmap(compute_interval_vert)(curr_conns, next_conns) # updated interval_vertices

                    # For interval vertices, we need to recompute from boundary blocks
                    # For now, use the reference positions
                    shell_verts, _ = self._shell_vertices_from_connections(updated_conn_verts, interval_vertices, order=count)
                    shell_centroid = polygon_centroid(shell_verts)
                    shell_rel = shell_verts - shell_centroid
                else:
                    shell_rel = jnp.zeros((3, 2))

                shell_node_vectors.append(shell_rel)
                shell_centroids.append(shell_centroid)

            node_vectors = tuple(inner_node_vectors) + tuple(shell_node_vectors)
            centroids = tuple(inner_centroids) + tuple(shell_centroids)
            # Return as tuple
            return centroids, node_vectors

        self.block_centroids_node = jit(block_centroids_node) # integrated function for blocks_centroids, centroid_node_vectors
        self.block_centroids = jit(block_centroids)
        self.centroid_node_vectors = jit(centroid_node_vectors)

        # self.block_centroids_node = block_centroids_node # integrated function for blocks_centroids, centroid_node_vectors
        # self.block_centroids = block_centroids
        # self.centroid_node_vectors = centroid_node_vectors

        def bond_connectivity_org():
            """
            Computes bonds' connectivity for a circle-shaped structure with shell blocks.
            Connects inner blocks to shell blocks instead of boundary blocks.
            """

            # Bonds between inner blocks
            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
                for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])
            bonds = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)

            # --- Map node indices to block indices ---
            def bond_to_block(bond):
                return bond // 4  # 4 nodes per block

            bond_blocks = vmap(bond_to_block)(bonds)
            b0 = bond_blocks[:, 0]
            b1 = bond_blocks[:, 1]

            # --- Keep only bonds between inner blocks ---
            b0_inside = jnp.isin(b0, self.blocks_inside)
            b1_inside = jnp.isin(b1, self.blocks_inside)
            keep_mask = jnp.logical_and(b0_inside, b1_inside)
            filtered_bonds = bonds[keep_mask]

            # --- Add bonds from inner blocks to shell blocks ---
            # Compute node offsets for shell blocks
            offset_quads = 4 * len(self.blocks_inside)
            shell_bonds = []

            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_node_info = conn_info[3]  # (block_idx, node_idx) pairs
                conn_indices = conn_info[4]  # indices in shell block

                shell_offset = offset_quads + sum(self.shell_node_counts[:count])

                for (b_inner, n_inner), shell_node_idx in zip(connection_node_info, conn_indices):
                    inner_node = b_inner * 4 + n_inner
                    shell_node = shell_offset + shell_node_idx
                    shell_bonds.append([inner_node, shell_node])

            if len(shell_bonds) > 0:
                shell_bonds = jnp.array(shell_bonds, dtype=int)
                filtered_bonds = jnp.concatenate([filtered_bonds, shell_bonds], axis=0)

            return filtered_bonds

        self.bond_connectivity_org = bond_connectivity_org

        def bond_connectivity():
            """
            Computes bonds' connectivity for a circle-shaped structure with shell blocks.
            Connects inner blocks to shell blocks instead of boundary blocks.
            Vertex enumeration is based only on blocks_inside.
            """

            # Create mapping from original block index to inner block index
            # blocks_inside[i] -> inner block index i
            # Make lookup array large enough to cover all possible block indices
            total_blocks = self.n1_blocks * self.n2_blocks
            # Create a lookup array: block_to_inner[original_block] = inner_block_idx or -1 if not in blocks_inside
            block_to_inner = -jnp.ones(total_blocks, dtype=int)
            block_to_inner = block_to_inner.at[self.blocks_inside].set(jnp.arange(len(self.blocks_inside)))

            # Helper function to remap node index from full-grid to inner-only enumeration
            def remap_node(original_node_idx):
                """Remap node index from full-grid to inner-only enumeration."""
                original_block = original_node_idx // 4
                node_in_block = original_node_idx % 4
                inner_block_idx = block_to_inner[original_block]
                # If block is not in blocks_inside, return -1 (invalid)
                return jnp.where(inner_block_idx >= 0, inner_block_idx * 4 + node_in_block, -1)

            # Bonds between inner blocks (using full-grid indexing initially)
            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
                for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])
            bonds = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)

            # --- Map node indices to block indices ---
            def bond_to_block(bond):
                return bond // 4  # 4 nodes per block

            bond_blocks = vmap(bond_to_block)(bonds)
            b0 = bond_blocks[:, 0]
            b1 = bond_blocks[:, 1]

            # --- Keep only bonds between inner blocks ---
            b0_inside = jnp.isin(b0, self.blocks_inside)
            b1_inside = jnp.isin(b1, self.blocks_inside)
            keep_mask = jnp.logical_and(b0_inside, b1_inside)
            filtered_bonds = bonds[keep_mask]

            # --- Remap node indices to inner-only enumeration ---
            remap_node_vec = vmap(remap_node)
            filtered_bonds = remap_node_vec(filtered_bonds)

            # --- Add bonds from inner blocks to shell blocks ---
            # Compute node offsets for shell blocks
            offset_quads = 4 * len(self.blocks_inside)
            shell_bonds = []

            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_node_info = conn_info[3]  # (block_idx, node_idx) pairs - block_idx is original block index
                conn_indices = conn_info[4]  # indices in shell block

                shell_offset = offset_quads + sum(self.shell_node_counts[:count])

                for (_, _, b_inner_orig, n_inner), shell_node_idx in zip(connection_node_info, conn_indices):
                    # b_inner_orig is original block index, remap to inner block index
                    inner_block_idx = block_to_inner[b_inner_orig]
                    inner_node = inner_block_idx * 4 + n_inner
                    shell_node = shell_offset + shell_node_idx
                    shell_bonds.append([inner_node, shell_node])

            if len(shell_bonds) > 0:
                shell_bonds = jnp.array(shell_bonds, dtype=int)
                filtered_bonds = jnp.concatenate([filtered_bonds, shell_bonds], axis=0)

            return filtered_bonds

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """
            Computes the reference configuration of the bonds including shell blocks.
            """
            # Get node positions from geometry functions
            cnv = self.centroid_node_vectors(horizontal_shift, vertical_shift)  # tuple
            bcc = self.block_centroids(horizontal_shift, vertical_shift)       # tuple

            # Flatten to arrays
            rel = jnp.vstack(list(cnv))                         # (n_nodes,2)
            cents = jnp.vstack(list(bcc))                       # (n_blocks_total,2)
            cents_rep = jnp.repeat(cents, self.block_node_counts, axis=0)
            nodes_xy = cents_rep + rel                          # (n_nodes_total,2)

            # Get bond connectivity
            conn = self.bond_connectivity()

            # Compute reference vectors
            return nodes_xy[conn[:, 1], :] - nodes_xy[conn[:, 0], :]

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(horizontal_shift, vertical_shift)

    def get_xy_limits(self, vertices: jnp.ndarray):
        """
        Computes reference coonfiguration xy limits.
        """

        return compute_xy_limits(vertices)

    def get_design_from_rotated_square(self, angle):
        """Get horizontal and vertical shifts corresponding to a rotated square geometry with the given angle.

        Args:
            angle (float): Angle of the rotated square geometry.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple of horizontal and vertical shifts.
        """

        horizontal_shifts = jnp.array([[
            (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
            jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
            jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            for n2 in range(self.n2_blocks)] for n1 in range(self.n1_blocks+1)])
        vertical_shifts = jnp.array([[
            jnp.dot(
                rotation_matrix(jnp.pi/2),
                (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
                jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            )
            for n2 in range(self.n2_blocks+1)] for n1 in range(self.n1_blocks)])

        return horizontal_shifts, vertical_shifts

    def get_parametrization(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Returns the set of functions parameterizing the geometry.

        Returns:
            Tuple[Callable, Callable, Callable, Callable]: parameterizing functions: block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors.
        """

        self.compute_geometry()

        return self.block_centroids, self.centroid_node_vectors, self.bond_connectivity, self.reference_bond_vectors


class QuadGeometry_Circle_InputSource(LatticeGeometry):
    """
    Circular structure and remove bonds between boundary blocks to make separate input sources.
    """

    def __init__(self, n1_blocks: int, n2_blocks: int, spacing: float = 1.0, bond_length: float = 0.1, n_piece=8):
        """
        Creates a non-periodic lattice made of quadrangles with finite-length bonds.
        """

        super().__init__(n1_cells=n1_blocks, n2_cells=n2_blocks, n_bpc=1, direct_basis=spacing * jnp.eye(2))
        self.spacing = spacing
        self.bond_length = bond_length
        self.n1_blocks = self.n1_cells
        self.n2_blocks = self.n2_cells
        self.n_npb = 4
        self.n_quads = self.n_blocks  # quadrilateral blocks from LatticeGeometry
        self.n_piece = n_piece
        self.n_quads_org = self.n_blocks

        # Shell blocks: one per interval (n_piece total)
        # Node counts will be determined dynamically based on connection vertices
        self.n_shell_blocks = n_piece
        self.shell_node_counts = []  # Will be set in compute_geometry

        # Total blocks: quads (only inside blocks) + shell blocks
        # Note: boundary quads are replaced by shell blocks
        self.block_node_counts_list = []  # Will be set in compute_geometry
        # self.n_blocks_total = 0  # Will be set in compute_geometry
        self.n_nodes_total = 0  # Will be set in compute_geometry
        self.block_node_counts = None  # Will be set in compute_geometry

        self.block_centroids: Callable
        self.centroid_node_vectors: Callable
        self.bond_connectivity: Callable
        self.reference_bond_vectors: Callable

    def _shell_vertices_from_connections(self, connection_vertices: jnp.ndarray, interval_vertices: jnp.ndarray, order: int):
        """
        Build shell polygon with zigzag inner edge and smooth outer edge.
        Connection vertices are where inner blocks connect to the shell.
        Interval vertices are added between connection vertices for zigzag inner edge.
        Outer edge is smooth with up to 3 additional vertices for circular shape.
        
        Args:
            connection_vertices: (n_conn, 2) vertices on inner blocks for connection
            interval_vertices: (n_interval, 2) vertices between connection vertices (zigzag inner edge)
            
        Returns:
            verts: (n_conn + n_interval + n_outer, 2) shell vertices ordered counter-clockwise
            connection_indices: (n_conn,) indices of connection vertices in verts
        """
        n_conn = connection_vertices.shape[0]
        n_interval = interval_vertices.shape[0] # n_interval = n_conn-1
        
        if n_conn == 0:
                # Fallback: create minimal shell
            return jnp.zeros((3, 2)), jnp.array([], dtype=int)
        
        # Order connection vertices by angle around center
        # center = jnp.mean(connection_vertices, axis=0)
        center = self._center
        conn_dist_vec = connection_vertices - center
        conn_angles = jnp.atan2(conn_dist_vec[:, 1], conn_dist_vec[:, 0])

        # Add the exception treatment

        # conn_sorted_idx = jnp.argsort(conn_angles)
        # conn_sorted = connection_vertices[conn_sorted_idx]
        conn_sorted = connection_vertices # keep the order
        conn_sorted_idx = jnp.arange(len(connection_vertices))
        
        # Build inner edge: connection vertices + interval vertices (zigzag pattern)
        # Interleave interval vertices between connection vertices
        inner_edge_verts = []
        conn_indices_inner = []
        
        # Match interval vertices to connection pairs
        # Interval vertices are created between consecutive connection vertices
        count = 0 # vertices are aligned in counter-clockwise direction
        for i in range(n_conn):
            # Add connection vertex
            inner_edge_verts.append(conn_sorted[i]) # conned_sort
            conn_indices_inner.append(count)
            count += 1
            # Add corresponding interval vertex after this connection vertex
            # Interval vertices are ordered to match connection pairs
            if n_interval > 0 and i < n_interval:
                inner_edge_verts.append(interval_vertices[i])
                count += 1
        
        inner_edge_verts = jnp.array(inner_edge_verts)
        
        # Build smooth outer edge with up to 3 additional vertices
        # Compute average radius of inner edge
        # inner_radii = jnp.linalg.norm(inner_edge_verts - center, axis=1)
        # avg_radius = jnp.mean(inner_radii)
        # TOLERANCE = self.spacing * 0.5
        inner_radius = 0.5 * self.n1_blocks * self.spacing * 0.8
        outer_radius = inner_radius + self.spacing * self.n1_blocks * 1./15 # 16.125
        # outer_radius = inner_radius + self.spacing # 16.125
        
        # Create outer edge vertices (smooth, circular)
        # Use up to 4 vertices to approximate a smooth circular arc
        # n_outer = min(3, max(1, n_conn // 2))  # Up to 3 outer vertices, at least 1 if we have connections
        n_outer = 4
        outer_edge_verts = []
        angle_cut = 2. * jnp.pi / self.n_piece 
        desired_angle = angle_cut - 5.0 * jnp.pi / 180. # angle of arc
        
        if n_outer > 0 and n_conn > 0:
            # Compute angles for outer vertices (distributed along the arc)
            first_angle = conn_angles[conn_sorted_idx[-1]]
            last_angle = conn_angles[conn_sorted_idx[0]]
            # first_angle = - angle_cut * (order+1) + jnp.pi + buffer_angle 
            # last_angle = - angle_cut * order + jnp.pi - buffer_angle
            # Handle wrap-around: if the arc crosses -π/π boundary
            last_angle = jnp.where( last_angle < first_angle, last_angle+2*jnp.pi, last_angle)
            # if last_angle < first_angle:
            #     last_angle += 2 * jnp.pi
            first_angle = (first_angle+last_angle)/2.0 - desired_angle/2.0
            last_angle = first_angle + desired_angle

            # Create outer vertices along smooth circular arc
            for i in range(n_outer):
                # Distribute angles evenly between first and last connection
                t = (i) / (n_outer-1)
                angle = first_angle + t * (last_angle - first_angle)
                # angle = last_angle + t * (first_angle - last_angle) # start from last point to first point
                # Use angle directly (already in correct range after wrap-around handling)
                outer_vert = center + outer_radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
                outer_edge_verts.append(outer_vert)
        
        outer_edge_verts = jnp.array(outer_edge_verts) if len(outer_edge_verts) > 0 else jnp.zeros((0, 2))
        
        # Build complete shell: inner edge (zigzag) -> outer edge (smooth)
        # The polygon goes: connection vertices and interval vertices (zigzag inner edge),
        # then outer vertices (smooth outer edge), forming a closed polygon
        shell_verts = jnp.vstack([inner_edge_verts, outer_edge_verts])
        
        # Map connection indices to final shell vertex indices
        connection_indices = jnp.array(conn_indices_inner, dtype=int)
        
        return shell_verts, connection_indices

    def compute_geometry(self):
        """
        Implements mappings between (`horizontal_shift`, `vertical_shift`) and `centroid_node_vectors`, `bond_connectivity`, `reference_bond_vectors`.
        """

        def reference_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting the reference point (square grid) of the block to each node.

            Args:
                horizontal_shift (jnp.ndarray): array of shape (n1_cells+1, n2_cells, 2) defining the shifts of the horizontally aligned nodes.
                vertical_shift (jnp.ndarray): array of shape (n1_cells, n2_cells+1, 2) defining the shifts of the vertically aligned nodes.
            """

            v0 = (self.spacing - self.bond_length) / 2 * jnp.array([1., 0.])
            v0s = vmap(lambda angle: jnp.dot(rotation_matrix(angle), v0))(jnp.linspace(0., 3 * jnp.pi / 2, 4))

            def _reference_node_vectors_block(n1_block, n2_block):
                return v0s + jnp.array([
                    horizontal_shift[n1_block+1, n2_block],
                    vertical_shift[n1_block, n2_block+1],
                    horizontal_shift[n1_block, n2_block],
                    vertical_shift[n1_block, n2_block],
                ])

            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_quads_org,)), n2s.reshape((self.n_quads_org,))

            return vmap(_reference_node_vectors_block, in_axes=(0, 0))(n1s, n2s)

        def quad_reference_points():
            """Computes reference points of the quad blocks."""
            n1s, n2s = jnp.meshgrid(jnp.arange(self.n1_blocks), jnp.arange(self.n2_blocks))
            n1s, n2s = n1s.reshape((self.n_quads_org,)), n2s.reshape((self.n_quads_org,))
            return vmap(lambda i, j: i * self.direct_basis[0] + j * self.direct_basis[1], in_axes=(0, 0))(n1s, n2s)

        def quad_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes quad blocks' centroids."""
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            return quad_reference_points() + centroid_shifts

        def quad_centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting quad block centroids to nodes."""
            reference_vectors = reference_node_vectors(horizontal_shift, vertical_shift)
            centroid_shifts = vmap(polygon_centroid)(reference_vectors)
            return vmap(lambda block_nodes, shift: block_nodes - shift, in_axes=(0, 0))(reference_vectors, centroid_shifts)

        # --- Setup bond connectivity for finding boundaries ---
        horizontal_bonds = jnp.array([
            [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
            for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
        ])
        vertical_bonds = jnp.array([
            [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
            for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
        ])
        _bond_connectivity = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)
        self._bond_connectivity = _bond_connectivity

        # bond connecting between boundary blocks / n_piece will be variable later
        angle_cut = 360.0 / self.n_piece

        # --- Compute block centroids for reference configuration ---
        h_shift = jnp.zeros((self.n1_blocks + 1, self.n2_blocks, 2))
        v_shift = jnp.zeros((self.n1_blocks, self.n2_blocks + 1, 2))
        centroids = quad_centroids(h_shift, v_shift)
        quad_rel = quad_centroid_node_vectors(h_shift, v_shift)
        # --- Circle definition ---
        center = jnp.mean(centroids, axis=0)
        radius = 0.5 * self.n1_blocks * self.spacing * 0.8  # 90% of grid half-width
        tol = self.spacing * 0.5  # boundary tolerance: about half a block spacing

        dist_vec = centroids - center
        # --- Classify blocks ---
        dist = jnp.linalg.norm(dist_vec, axis=1)
        inside_mask = dist < (radius - tol)
        blocks_inside = jnp.where(inside_mask)[0]
        _bond_connectivity_idx = _bond_connectivity // 4 # block indices that are connected
        connection = jnp.isin(_bond_connectivity_idx, blocks_inside)
        check_connection = jnp.sum( connection, axis=1)

        boundary = _bond_connectivity_idx[check_connection==1] # blocks that have connection with blocks inside
        connection_boundary = connection[check_connection==1]
        blocks_boundary = jnp.unique( boundary[~connection_boundary] )

        dist_vec_bound = dist_vec[blocks_boundary]
        angle_bound = jnp.atan2( dist_vec_bound[:,1], dist_vec_bound[:,0] ) * 180./jnp.pi
        bond_boundary_and_outside = _bond_connectivity_idx[check_connection == 0]

        boundary_separation = []
        for count in range(self.n_piece):

            ang_max = - angle_cut * count + 180.
            ang_min = - angle_cut * (count+1) + 180.
            
            mask_angle = jnp.logical_and( angle_bound > ang_min, angle_bound <= ang_max )
            bound_blocks = blocks_boundary[mask_angle] # boundary blocks corresponding the angle interval
            # check bonds that are connected to boundary blocks
            bond_partial = bond_boundary_and_outside[jnp.sum( jnp.isin(bond_boundary_and_outside, bound_blocks), axis=1) > 0]
            
            bridge_block = []

            if len(bridge_block) == 0:
                boundary_separation.append( bound_blocks )
            else:
                boundary_separation.append( jnp.concatenate( ( bound_blocks, jnp.array(bridge_block) ) ) )

        blocks_boundary = jnp.concatenate( boundary_separation )
        blocks_outside = jnp.arange( self.n1_blocks * self.n2_blocks )
        blocks_outside = jnp.delete( blocks_outside, jnp.concatenate( (blocks_inside, blocks_boundary) ) )
        self._center = center
        self.blocks_inside = blocks_inside
        self.blocks_boundary = blocks_boundary
        self.blocks_outside = blocks_outside
        self.boundary_separation = boundary_separation

        
        # --- Find connection vertices and create shell blocks ---
        # Compute node positions (centroid + relative)
        def get_node_pos(block_idx, node_idx):
            return centroids[block_idx] + quad_rel[block_idx, node_idx]
        
        # For each interval, find connection vertices and interval vertices
        shell_connection_info = []  # List of (connection_vertices, interval_vertices, connection_node_info, conn_indices)
        shell_node_counts_list = []
        
        for count in range(self.n_piece):
            bound_blocks = boundary_separation[count]
            
            # Find bonds between inner blocks and boundary blocks in this interval
            bonds_inner_to_boundary = []
            # Vectorized version using jnp functionality
            # Compute block indices and node indices for each bond
            b0 = _bond_connectivity[:, 0] // 4
            b1 = _bond_connectivity[:, 1] // 4
            node0 = _bond_connectivity[:, 0] % 4
            node1 = _bond_connectivity[:, 1] % 4
            # Determine mask for which bonds connect inner to boundary, either direction
            mask0 = jnp.logical_and(
                jnp.isin(b0, blocks_inside),
                jnp.isin(b1, bound_blocks)
            )
            mask1 = jnp.logical_and(
                jnp.isin(b1, blocks_inside),
                jnp.isin(b0, bound_blocks)
            )
            # Get indices of matches for both cases
            idx0 = jnp.where(mask0)[0]
            idx1 = jnp.where(mask1)[0]

            # For mask0: (b0 in inside, b1 in boundary)
            bonds0 = jnp.stack([b0[idx0], node0[idx0], b1[idx0], node1[idx0]], axis=1)

            # For mask1: (b1 in inside, b0 in boundary) (order reversed)
            bonds1 = jnp.stack([b1[idx1], node1[idx1], b0[idx1], node0[idx1]], axis=1)

            # Combine, convert to list of tuples for expected format
            bonds_inner_to_boundary = [tuple(x) for x in jnp.concatenate([bonds0, bonds1], axis=0)] # axis 0 blocks inside, axis 3 blocks boundary

            # Get connection vertices (on inner blocks)
            connection_vertices = []
            connection_node_info = []  # (block_idx, node_idx) for each connection
            for b_inner, n_inner, b_bound, n_bound in bonds_inner_to_boundary:
                conn_vert = get_node_pos(b_bound, n_bound)
                connection_vertices.append(conn_vert)
                connection_node_info.append((b_bound, n_bound, b_inner, n_inner))
            
            connection_vertices = jnp.array(connection_vertices) if len(connection_vertices) > 0 else jnp.zeros((0, 2))
            
            # Generate interval vertices between connection vertices for zigzag inner edge
            # Note: interval_vertices will be created based on sorted connection vertices
            # The _shell_vertices_from_connections method will sort connection_vertices again,
            # so we create interval_vertices here but they'll be properly matched in that method
            interval_vertices = []
            if connection_vertices.shape[0] > 1:
                # Order connection vertices by angle around center (same as in _shell_vertices_from_connections)
                conn_dist_vec = connection_vertices - center
                conn_angles = jnp.atan2(conn_dist_vec[:, 1], conn_dist_vec[:, 0])
                conn_sorted_idx = jnp.argsort(conn_angles, descending=True)
                conn_sorted = connection_vertices[conn_sorted_idx]
                connection_node_info = jnp.array(connection_node_info)[conn_sorted_idx]
                
                
                # Create interval vertices between consecutive connection vertices
                # These create the zigzag pattern on the inner edge
                def compute_interval_vert(curr_conn, next_conn):
                    mid_point = 0.5 * (curr_conn + next_conn)
                    direction = next_conn - curr_conn
                    perp_dir = jnp.array([-direction[1], direction[0]])
                    perp_dir = perp_dir / (jnp.linalg.norm(perp_dir) + 1e-8)
                    zigzag_offset = self.spacing * 0.3
                    zigzag_sign = 1
                    return mid_point + zigzag_sign * zigzag_offset * perp_dir

                # We want intervals between consecutive pairs: (conn_sorted[0], conn_sorted[1]), ...
                if conn_sorted.shape[0] > 1:
                    curr_conns = conn_sorted[:-1]
                    next_conns = conn_sorted[1:]
                    interval_vertices = vmap(compute_interval_vert)(curr_conns, next_conns)
                    # interval_vertices = [v for v in interval_vertices]
                
                # Store interval_vertices in the same sorted order as connection_vertices
                # (They're already in the correct order since we created them based on sorted connections)
            
            interval_vertices = interval_vertices if len(interval_vertices) > 0 else jnp.zeros((0, 2))
            
            # Create shell vertices
            if connection_vertices.shape[0] > 0 or interval_vertices.shape[0] > 0:
                shell_verts, conn_indices = self._shell_vertices_from_connections(conn_sorted, interval_vertices, order=count)
                n_shell_verts = shell_verts.shape[0]
            else:
                # Fallback: create minimal shell
                shell_verts = jnp.zeros((3, 2))
                conn_sorted = connection_vertices
                conn_indices = jnp.array([], dtype=int)
                n_shell_verts = 4
            
            shell_connection_info.append((shell_verts, conn_sorted, interval_vertices, connection_node_info, conn_indices))
            shell_node_counts_list.append(n_shell_verts)
        
        self.shell_node_counts = shell_node_counts_list
        self.shell_connection_info = shell_connection_info
        
        # Update block node counts
        self.block_node_counts_list = [4] * len(blocks_inside) + shell_node_counts_list
        self.n_blocks = len(blocks_inside) + self.n_shell_blocks
        self.n_nodes_total = sum(self.block_node_counts_list)
        self.block_node_counts = jnp.array(self.block_node_counts_list, dtype=int)
        self.n_quads = len(blocks_inside)
        
        
        # --- Define geometry functions that include shell blocks ---
        def block_centroids(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes blocks' centroids including shell blocks. Returns tuple."""

            quad_cents = quad_centroids(horizontal_shift, vertical_shift)
            quad_rel = quad_centroid_node_vectors(horizontal_shift, vertical_shift)
            
            # Get centroids for inner quads only
            # inner_centroids = [quad_cents[i] for i in blocks_inside]
            inner_centroids = quad_cents[self.blocks_inside]

            def get_node_pos(block_idx, node_idx):
                return quad_cents[block_idx] + quad_rel[block_idx, node_idx]
            # Compute shell block centroids
            shell_centroids = []
            shell_connection_info = []
            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_vertices = conn_info[1]
                interval_vertices = conn_info[2]
                connection_node_info = conn_info[3]
                conn_indices = conn_info[4]
                
                # # Recompute shell vertices for current configuration
                if connection_vertices.shape[0] > 0 or interval_vertices.shape[0] > 0:
                    b_bound, n_bound = connection_node_info[:,0], connection_node_info[:,1]
                    
                    updated_conn_verts = vmap(get_node_pos, in_axes=(0,0))(b_bound, n_bound)
                    # conn_dist_vec = connection_vertices - self._center
                    # Create interval vertices between consecutive connection vertices
                    # These create the zigzag pattern on the inner edge
                    def compute_interval_vert(curr_conn, next_conn):
                        mid_point = 0.5 * (curr_conn + next_conn)
                        direction = next_conn - curr_conn
                        perp_dir = jnp.array([-direction[1], direction[0]])
                        perp_dir = perp_dir / (jnp.linalg.norm(perp_dir) + 1e-8)
                        zigzag_offset = self.spacing * 0.3
                        zigzag_sign = 1
                        return mid_point + zigzag_sign * zigzag_offset * perp_dir
                    # We want intervals between consecutive pairs: (conn_sorted[0], conn_sorted[1]), ...
                    curr_conns = updated_conn_verts[:-1]
                    next_conns = updated_conn_verts[1:]
                    interval_vertices = vmap(compute_interval_vert)(curr_conns, next_conns) # updated interval_vertices

                    # For interval vertices, we need to recompute from boundary blocks
                    # For now, use the reference positions
                    shell_verts, _ = self._shell_vertices_from_connections(updated_conn_verts, interval_vertices, order=count)
                    shell_centroid = polygon_centroid(shell_verts)
                else:
                    # shell_verts = jnp.zeros((0, 2))
                    # updated_conn_verts = connection_vertices
                    shell_centroid = jnp.array([0., 0.])
                
                shell_centroids.append(shell_centroid)
            #     shell_connection_info.append((shell_verts, updated_conn_verts, interval_vertices, connection_node_info, conn_indices))
            # self.shell_connection_info = shell_connection_info
            # Return as tuple
            return tuple(inner_centroids) + tuple(shell_centroids)
        
        def centroid_node_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting block centroids to nodes. Returns tuple."""
            quad_cents = quad_centroids(horizontal_shift, vertical_shift)
            quad_rel = quad_centroid_node_vectors(horizontal_shift, vertical_shift)
            
            # Get node vectors for inner quads only
            # inner_node_vectors = [quad_rel[i] for i in blocks_inside]
            inner_node_vectors = quad_rel[self.blocks_inside]

            def get_node_pos(block_idx, node_idx):
                return quad_cents[block_idx] + quad_rel[block_idx, node_idx]

            # Compute shell block node vectors
            shell_node_vectors = []
            shell_connection_info = []
            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_vertices = conn_info[1]
                interval_vertices = conn_info[2]
                connection_node_info = conn_info[3]
                conn_indices = conn_info[4]
                
                # # Recompute shell vertices for current configuration
                if connection_vertices.shape[0] > 0 or interval_vertices.shape[0] > 0:
                    b_bound, n_bound = connection_node_info[:,0], connection_node_info[:,1]
                    updated_conn_verts = vmap(get_node_pos, in_axes=(0,0))(b_bound, n_bound)
                    # conn_dist_vec = connection_vertices - self._center
                    # Create interval vertices between consecutive connection vertices
                    # These create the zigzag pattern on the inner edge
                    def compute_interval_vert(curr_conn, next_conn):
                        mid_point = 0.5 * (curr_conn + next_conn)
                        direction = next_conn - curr_conn
                        perp_dir = jnp.array([-direction[1], direction[0]])
                        perp_dir = perp_dir / (jnp.linalg.norm(perp_dir) + 1e-8)
                        zigzag_offset = self.spacing * 0.3
                        zigzag_sign = 1
                        return mid_point + zigzag_sign * zigzag_offset * perp_dir
                    # We want intervals between consecutive pairs: (conn_sorted[0], conn_sorted[1]), ...
                    curr_conns = updated_conn_verts[:-1]
                    next_conns = updated_conn_verts[1:]
                    interval_vertices = vmap(compute_interval_vert)(curr_conns, next_conns) # updated interval_vertices

                    # For interval vertices, we need to recompute from boundary blocks
                    # For now, use the reference positions
                    shell_verts, _ = self._shell_vertices_from_connections(updated_conn_verts, interval_vertices, order=count)
                    shell_centroid = polygon_centroid(shell_verts)
                    shell_rel = shell_verts - shell_centroid
                else:
                    # shell_verts = jnp.zeros((0, 2))
                    # updated_conn_verts = connection_vertices
                    shell_rel = jnp.zeros((3, 2))
                
                shell_node_vectors.append(shell_rel)

            #     shell_connection_info.append((shell_verts, updated_conn_verts, interval_vertices, connection_node_info, conn_indices))
            # self.shell_connection_info = shell_connection_info
            return tuple(inner_node_vectors) + tuple(shell_node_vectors)
        

        def block_centroids_node(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """Computes vectors connecting block centroids to nodes. Returns tuple."""
            quad_cents = quad_centroids(horizontal_shift, vertical_shift)
            quad_rel = quad_centroid_node_vectors(horizontal_shift, vertical_shift)
            
            # Get node vectors for inner quads only
            # inner_node_vectors = [quad_rel[i] for i in blocks_inside]
            inner_node_vectors = quad_rel[self.blocks_inside]
            inner_centroids = quad_cents[self.blocks_inside]

            def get_node_pos(block_idx, node_idx):
                return quad_cents[block_idx] + quad_rel[block_idx, node_idx]
            # Compute shell block node vectors
            shell_node_vectors = []
            shell_centroids = []
            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_vertices = conn_info[1]
                interval_vertices = conn_info[2]
                
                # # Recompute shell vertices for current configuration
                if connection_vertices.shape[0] > 0 or interval_vertices.shape[0] > 0:
                    b_bound, n_bound = connection_node_info[:,0], connection_node_info[:,1]
                    updated_conn_verts = vmap(get_node_pos, in_axes=(0,0))(b_bound, n_bound)
                    # conn_dist_vec = connection_vertices - self._center
                    # Create interval vertices between consecutive connection vertices
                    # These create the zigzag pattern on the inner edge
                    def compute_interval_vert(curr_conn, next_conn):
                        mid_point = 0.5 * (curr_conn + next_conn)
                        direction = next_conn - curr_conn
                        perp_dir = jnp.array([-direction[1], direction[0]])
                        perp_dir = perp_dir / (jnp.linalg.norm(perp_dir) + 1e-8)
                        zigzag_offset = self.spacing * 0.3
                        zigzag_sign = -1
                        return mid_point + zigzag_sign * zigzag_offset * perp_dir
                    # We want intervals between consecutive pairs: (conn_sorted[0], conn_sorted[1]), ...
                    curr_conns = updated_conn_verts[:-1]
                    next_conns = updated_conn_verts[1:]
                    interval_vertices = vmap(compute_interval_vert)(curr_conns, next_conns) # updated interval_vertices

                    # For interval vertices, we need to recompute from boundary blocks
                    # For now, use the reference positions
                    shell_verts, _ = self._shell_vertices_from_connections(updated_conn_verts, interval_vertices, order=count)
                    shell_centroid = polygon_centroid(shell_verts)
                    shell_rel = shell_verts - shell_centroid
                else:
                    shell_rel = jnp.zeros((3, 2))
                
                shell_node_vectors.append(shell_rel)
                shell_centroids.append(shell_centroid)

            node_vectors = tuple(inner_node_vectors) + tuple(shell_node_vectors)
            centroids = tuple(inner_centroids) + tuple(shell_centroids)
            # Return as tuple
            return centroids, node_vectors

        self.block_centroids_node = jit(block_centroids_node) # integrated function for blocks_centroids, centroid_node_vectors
        self.block_centroids = jit(block_centroids)
        self.centroid_node_vectors = jit(centroid_node_vectors)


        def bond_connectivity_org():
            """
            Computes bonds' connectivity for a circle-shaped structure with shell blocks.
            Connects inner blocks to shell blocks instead of boundary blocks.
            """

            # Bonds between inner blocks
            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
                for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])
            bonds = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)

            # --- Map node indices to block indices ---
            def bond_to_block(bond):
                return bond // 4  # 4 nodes per block

            bond_blocks = vmap(bond_to_block)(bonds)
            b0 = bond_blocks[:, 0]
            b1 = bond_blocks[:, 1]
            
            # --- Keep only bonds between inner blocks ---
            b0_inside = jnp.isin(b0, self.blocks_inside)
            b1_inside = jnp.isin(b1, self.blocks_inside)
            keep_mask = jnp.logical_and(b0_inside, b1_inside)
            filtered_bonds = bonds[keep_mask]
            
            # --- Add bonds from inner blocks to shell blocks ---
            # Compute node offsets for shell blocks
            offset_quads = 4 * len(self.blocks_inside)
            shell_bonds = []
            
            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_node_info = conn_info[3]  # (block_idx, node_idx) pairs
                conn_indices = conn_info[4]  # indices in shell block
                
                shell_offset = offset_quads + sum(self.shell_node_counts[:count])
                
                for (b_inner, n_inner), shell_node_idx in zip(connection_node_info, conn_indices):
                    inner_node = b_inner * 4 + n_inner
                    shell_node = shell_offset + shell_node_idx
                    shell_bonds.append([inner_node, shell_node])
            
            if len(shell_bonds) > 0:
                shell_bonds = jnp.array(shell_bonds, dtype=int)
                filtered_bonds = jnp.concatenate([filtered_bonds, shell_bonds], axis=0)

            return filtered_bonds

        self.bond_connectivity_org = bond_connectivity_org

        def bond_connectivity():
            """
            Computes bonds' connectivity for a circle-shaped structure with shell blocks.
            Connects inner blocks to shell blocks instead of boundary blocks.
            Vertex enumeration is based only on blocks_inside.
            """

            # Create mapping from original block index to inner block index
            # blocks_inside[i] -> inner block index i
            # Make lookup array large enough to cover all possible block indices
            total_blocks = self.n1_blocks * self.n2_blocks
            # Create a lookup array: block_to_inner[original_block] = inner_block_idx or -1 if not in blocks_inside
            block_to_inner = -jnp.ones(total_blocks, dtype=int)
            block_to_inner = block_to_inner.at[self.blocks_inside].set(jnp.arange(len(self.blocks_inside)))

            # Helper function to remap node index from full-grid to inner-only enumeration
            def remap_node(original_node_idx):
                """Remap node index from full-grid to inner-only enumeration."""
                original_block = original_node_idx // 4
                node_in_block = original_node_idx % 4
                inner_block_idx = block_to_inner[original_block]
                # If block is not in blocks_inside, return -1 (invalid)
                return jnp.where(inner_block_idx >= 0, inner_block_idx * 4 + node_in_block, -1)

            # Bonds between inner blocks (using full-grid indexing initially)
            horizontal_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4, self.n1_blocks * n2 * 4 + (n1 + 1) * 4 + 2]
                for n2 in range(self.n2_blocks) for n1 in range(self.n1_blocks - 1)
            ])
            vertical_bonds = jnp.array([
                [self.n1_blocks * n2 * 4 + n1 * 4 + 1, self.n1_blocks * (n2 + 1) * 4 + n1 * 4 + 1 + 2]
                for n2 in range(self.n2_blocks - 1) for n1 in range(self.n1_blocks)
            ])
            bonds = jnp.concatenate([horizontal_bonds, vertical_bonds], axis=0)

            # --- Map node indices to block indices ---
            def bond_to_block(bond):
                return bond // 4  # 4 nodes per block

            bond_blocks = vmap(bond_to_block)(bonds)
            b0 = bond_blocks[:, 0]
            b1 = bond_blocks[:, 1]
            
            # --- Keep only bonds between inner blocks ---
            b0_inside = jnp.isin(b0, self.blocks_inside)
            b1_inside = jnp.isin(b1, self.blocks_inside)
            keep_mask = jnp.logical_and(b0_inside, b1_inside)
            filtered_bonds = bonds[keep_mask]
            
            # --- Remap node indices to inner-only enumeration ---
            remap_node_vec = vmap(remap_node)
            filtered_bonds = remap_node_vec(filtered_bonds)
            
            # --- Add bonds from inner blocks to shell blocks ---
            # Compute node offsets for shell blocks
            offset_quads = 4 * len(self.blocks_inside)
            shell_bonds = []
            
            for count in range(self.n_piece):
                conn_info = self.shell_connection_info[count]
                connection_node_info = conn_info[3]  # (block_idx, node_idx) pairs - block_idx is original block index
                conn_indices = conn_info[4]  # indices in shell block
                
                shell_offset = offset_quads + sum(self.shell_node_counts[:count])
                
                for (_, _, b_inner_orig, n_inner), shell_node_idx in zip(connection_node_info, conn_indices):
                    # b_inner_orig is original block index, remap to inner block index
                    inner_block_idx = block_to_inner[b_inner_orig]
                    inner_node = inner_block_idx * 4 + n_inner
                    shell_node = shell_offset + shell_node_idx
                    shell_bonds.append([inner_node, shell_node])
            
            if len(shell_bonds) > 0:
                shell_bonds = jnp.array(shell_bonds, dtype=int)
                filtered_bonds = jnp.concatenate([filtered_bonds, shell_bonds], axis=0)

            return filtered_bonds

        self.bond_connectivity = bond_connectivity

        def reference_bond_vectors(horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
            """
            Computes the reference configuration of the bonds including shell blocks.
            """
            # Get node positions from geometry functions
            cnv = self.centroid_node_vectors(horizontal_shift, vertical_shift)  # tuple
            bcc = self.block_centroids(horizontal_shift, vertical_shift)       # tuple
            
            # Flatten to arrays
            rel = jnp.vstack(list(cnv))                         # (n_nodes,2)
            cents = jnp.vstack(list(bcc))                       # (n_blocks_total,2)
            cents_rep = jnp.repeat(cents, self.block_node_counts, axis=0)
            nodes_xy = cents_rep + rel                          # (n_nodes_total,2)
            
            # Get bond connectivity
            conn = self.bond_connectivity()
            
            # Compute reference vectors
            return nodes_xy[conn[:, 1], :] - nodes_xy[conn[:, 0], :]

        self.reference_bond_vectors = reference_bond_vectors

    def get_reference_geometry(self, horizontal_shift: jnp.ndarray, vertical_shift: jnp.ndarray):
        """
        Computes reference coonfiguration.
        """
        return super().get_reference_geometry(horizontal_shift, vertical_shift) 

    def get_xy_limits(self, vertices: jnp.ndarray):
        """
        Computes reference coonfiguration xy limits.
        """

        return compute_xy_limits(vertices)

    def get_design_from_rotated_square(self, angle):
        """Get horizontal and vertical shifts corresponding to a rotated square geometry with the given angle.

        Args:
            angle (float): Angle of the rotated square geometry.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple of horizontal and vertical shifts.
        """

        horizontal_shifts = jnp.array([[
            (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
            jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
            jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            for n2 in range(self.n2_blocks)] for n1 in range(self.n1_blocks+1)])
        vertical_shifts = jnp.array([[
            jnp.dot(
                rotation_matrix(jnp.pi/2),
                (self.spacing - self.bond_length) / (2 * jnp.cos((-1)**(n1 + n2) * angle)) *
                jnp.array([jnp.cos((-1)**(n1 + n2) * angle), jnp.sin((-1)**(n1 + n2) * angle)]) -
                jnp.array([1, 0]) * (self.spacing - self.bond_length) / 2
            )
            for n2 in range(self.n2_blocks+1)] for n1 in range(self.n1_blocks)])

        return horizontal_shifts, vertical_shifts

    def get_parametrization(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Returns the set of functions parameterizing the geometry.

        Returns:
            Tuple[Callable, Callable, Callable, Callable]: parameterizing functions: block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors.
        """

        self.compute_geometry()

        return self.block_centroids, self.centroid_node_vectors, self.bond_connectivity, self.reference_bond_vectors
