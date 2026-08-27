import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from matplotlib import cm, colors
from matplotlib.collections import (LineCollection, PatchCollection,
                                    PolyCollection)
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize

from blockymetamaterials.geometry import compute_xy_limits, current_coordinates, rotation_matrix
from blockymetamaterials.utils import EigenmodeData, SolutionData, load_data
from blockymetamaterials.kinematics import compute_current_nodes_for_plot


def orange_blue_cmap():
    """
    Custom colormap
    """
    # top = matplotlib.colormaps['Oranges_r']
    # bottom = matplotlib.colormaps['Blues']
    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    return ListedColormap(newcolors, name='OrangeBlue')


def plot_energy(dat):
    pot_energy = []
    kin_energy = []
    for i in range(dat.fields.shape[0]):
        dx = dat.fields[i, 0, :, 0]
        dy = dat.fields[i, 0, :, 1]

        pot_energy.append(np.sum(dx**2+dy**2))
        vx = dat.fields[i, 1, :, 0]
        vy = dat.fields[i, 1, :, 1]
        kin_energy.append(np.sum(vx**2+vy**2))

    plt.figure(2)
    plt.plot(dat.timepoints, kin_energy, lw=2, label="kinetic")
    plt.plot(dat.timepoints, pot_energy, lw=2, label="potential")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.savefig("out/energy.png", dpi=300, bbox_inches='tight')


def generate_polygons(block_centroids, centroid_node_vectors, block_displacements=None, deformed=False):
    """
    docstring
    """

    if deformed and block_displacements is not None:
        # polygons = [
        #     Polygon((rotation_matrix(DOFs[-1]) @ vertices.T).T + centroid + DOFs[:2])
        #     for vertices, centroid, DOFs in zip(centroid_node_vectors, block_centroids, block_displacements)]
            # get per-block node displacements (tuple of (ni,3))
        current_nodes = compute_current_nodes_for_plot(block_displacements, block_centroids, centroid_node_vectors)
        polygons = [Polygon(vertices) for vertices in current_nodes]
    else:
        polygons = [Polygon(vertices + centroid)
                    for vertices, centroid in zip(centroid_node_vectors, block_centroids)]
        # polygons = [Polygon(vertices + centroid, True)
        #             for vertices, centroid in zip(centroid_node_vectors, block_centroids)]

    return polygons

# add new function to generate polycollection
def generate_polycollection(block_centroids, centroid_node_vectors,
                            block_displacements=None, field_values=None,
                            deformed=False, clim=None, cmap=orange_blue_cmap(),
                            fill=True):
    if deformed and block_displacements is not None:
        # list of (ni,2) arrays
        polys = compute_current_nodes_for_plot(block_displacements,
                                               block_centroids,
                                               centroid_node_vectors)
    else:
        # list of (ni,2) arrays
        polys = [verts + c for verts, c in zip(centroid_node_vectors, block_centroids)]

    coll = PolyCollection(polys, cmap=cmap, alpha=0.95)
    if field_values is not None:
        coll.set_array(field_values)
        vmin, vmax = (field_values.min(), field_values.max()) if clim is None else clim
        coll.set_clim(vmin, vmax)
    coll.set(edgecolor="black", linewidth=0.5)
    
    # --- If fill is False, we want colored edges according to colormap ---
    if not fill:
        # Get colormap + normalized field values → (N, 4) RGBA colors
        if field_values is not None:
            norm = plt.Normalize(vmin, vmax)
            rgba = cmap(norm(field_values))

            # set facecolor transparent
            rgba_face = rgba.copy()
            rgba_face[:, -1] = 0.0  # alpha = 0

            # apply edgecolor from the original colormap (opaque)
            rgba_edge = rgba.copy()
            rgba_edge[:, -1] = 1.0  # alpha = 1

            coll.set_facecolor(rgba_face)
            coll.set_edgecolor(rgba_edge)
        else:
            # If no field_values, default to black edges
            coll.set_facecolor("none")
            coll.set_edgecolor("black")

    else:
        # normal filled mode
        coll.set_edgecolor("black")
    
    return coll

def generate_patch_collection(block_centroids, centroid_node_vectors, block_displacements=None, field_values=None, deformed=False, clim=None, cmap=orange_blue_cmap()):
    """
    docstring
    """

    polygons = generate_polygons(block_centroids, centroid_node_vectors,
                                 block_displacements=block_displacements, deformed=deformed)
    patches = PatchCollection(polygons, cmap=cmap, alpha=1.0)
    # patches = PolyCollection(polygons, cmap=cmap, alpha=0.95)
    if field_values is not None:
        patches.set_array(field_values)
        min_value, max_value = (field_values.min(), field_values.max()) if clim is None else clim
        patches.set_clim(min_value, max_value)
    patches.set(edgecolor="black", linewidth=0.5)

    return patches


def generate_bond_collection(block_centroids, centroid_node_vectors, bond_connectivity, block_displacements=None, deformed=False):
    """
    docstring
    """

    # Generate collection of bonds as lines
    if deformed and block_displacements is not None:
        node_coords = current_coordinates(centroid_node_vectors, block_centroids,
                                           block_displacements[:, -1], block_displacements[:, :2])
    else:
        # block_coords = vmap(lambda centroid, centroid_node_vector: centroid +
        #                     centroid_node_vector, in_axes=(0, 0))(centroid_node_vectors, block_centroids)
        vertices_list = [vertices + centroid for vertices, centroid in zip(centroid_node_vectors, block_centroids)]
        node_coords = np.concatenate(vertices_list, axis=0) 

    return LineCollection(node_coords[bond_connectivity], color="black", linewidth=0.5)


def plot_geometry(block_centroids, centroid_node_vectors, bond_connectivity, block_displacements=None, deformed=False,
                  color="#2980b9", figsize=None, xlim=None, ylim=None, ax=None):
    """
    docstring
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.axis("equal")
    # Generate collection of blocks as polygons
    patches = generate_patch_collection(block_centroids, centroid_node_vectors,
                                        block_displacements=block_displacements, deformed=deformed)
    patches.set(color=color)
    patches.set(edgecolor="black", linewidth=0.5)
    ax.add_collection(patches)
    # Generate collection of bonds as lines
    collection_bonds = generate_bond_collection(
        block_centroids, centroid_node_vectors, bond_connectivity, block_displacements=block_displacements, deformed=deformed)
    ax.add_collection(collection_bonds)

    if deformed and block_displacements is not None:
        points = current_coordinates(centroid_node_vectors, block_centroids,
                                     block_displacements[:, -1], block_displacements[:, :2]).reshape((-1, 2))
    else:
        points = current_coordinates(centroid_node_vectors, block_centroids,
                                     np.zeros((len(centroid_node_vectors),)), np.zeros((len(centroid_node_vectors),2))).reshape((-1, 2))

    _xlim, _ylim = compute_xy_limits(points)
    xlim = _xlim if xlim is None else xlim
    ylim = _ylim if ylim is None else ylim
    ax.set(xlim=xlim, ylim=ylim)

    fig = ax.get_figure()

    return fig, ax



def prepare_solution_figure(data: SolutionData, field, frame_range, figsize, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True, field_values=None):

    if field == "ux":
        field_values = data.fields[:, 0, :, 0]
        _legend_label = r"$u_1$"
    elif field == "uy":
        field_values = data.fields[:, 0, :, 1]
        _legend_label = r"$u_2$"
    elif field == "theta":
        field_values = data.fields[:, 0, :, 2] * 180. / np.pi
        _legend_label = r"$\theta$"
    elif field == "vx":
        field_values = data.fields[:, 1, :, 0]
        _legend_label = r"$\dot{u}_1$"
    elif field == "vy":
        field_values = data.fields[:, 1, :, 1]
        _legend_label = r"$\dot{u}_2$"
    elif field == "omega":
        field_values = data.fields[:, 1, :, 2]
        _legend_label = r"$\dot{theta}$"
    elif field == "omega_abs":
        field_values = np.abs(data.fields[:, 1, :, 2])
        _legend_label = r"$|\dot{theta}|$"
    elif field == "u":
        field_values = (data.fields[:, 0, :, 0]**2 + data.fields[:, 0, :, 1]**2)**0.5
        _legend_label = r"$u$"
    elif field == "v":
        field_values = (data.fields[:, 1, :, 0]**2 + data.fields[:, 1, :, 1]**2)**0.5
        _legend_label = r"$\dot{u}$"
    elif field == "theta_abs":
        field_values = np.abs(data.fields[:, 0, :, 2])
        _legend_label = r"$|\theta|$"
    elif type(field) == str and field_values is not None:
        _legend_label = field
    else:
        raise ValueError

    min_value, max_value = field_values.min(), field_values.max()
    vmin, vmax = vlim if vlim is not None else (min_value, max_value)
    _legend_label = legend_label if legend_label is not None else _legend_label

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.axis("equal")
    axes.tick_params(labelsize=ticksize)
    if not axis:
        axes.axis("off")
    cb = fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        ax=axes,
        pad=0.02,
        aspect=40
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize)
    frames = range(len(data.timepoints)) if frame_range is None else frame_range

    return field_values, min_value, max_value, fig, axes, frames


def prepare_mode_figure(data: EigenmodeData, field, mode_range, figsize, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True):

    if field == "ux":
        field_values = data.fields[:, :, 0]
        _legend_label = r"$u_1$"
    elif field == "uy":
        field_values = data.fields[:, :, 1]
        _legend_label = r"$u_2$"
    elif field == "theta":
        field_values = data.fields[:, :, 2]
        _legend_label = r"$\theta$"
    elif field == "u":
        field_values = (data.fields[:, :, 0]**2 + data.fields[:, :, 1]**2)**0.5
        _legend_label = r"$u$"
    elif field == "theta_abs":
        field_values = np.abs(data.fields[:, :, 2])
        _legend_label = r"$|\theta|$"
    else:
        raise ValueError

    vmin, vmax = vlim if vlim is not None else (None, None)
    _legend_label = legend_label if legend_label is not None else _legend_label

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.axis("equal")
    axes.tick_params(labelsize=ticksize)
    if not axis:
        axes.axis("off")
    cb = fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        ax=axes,
        pad=0.02,
        label=_legend_label,
        aspect=40
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize)
    frames = range(len(data.fields)) if mode_range is None else mode_range

    return field_values, fig, axes, frames


def generate_mode_images(data: EigenmodeData, field, out_dir, deformed=False, mode_range=None, scale_deformation=1, figsize=None, xlim=None, ylim=None, dpi=200, geometry=None, mesh=None, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True):
    """
    mesh=None: if set to True, a mesh connecting the centroids of each block is superimposed on the images
    docstring
    """

    field_values, fig, axes, frames = prepare_mode_figure(
        data, field, mode_range, figsize, cmap=cmap, vlim=vlim, legend_label=legend_label, fontsize=fontsize, ticksize=ticksize, axis=axis
    )
    block_centroids = data.block_centroids
    centroid_node_vectors = data.centroid_node_vectors
    block_displacements = data.fields

    for i in frames:
        # Each frame refer to a mode
        patches = generate_patch_collection(
            block_centroids=block_centroids,
            centroid_node_vectors=centroid_node_vectors,
            block_displacements=block_displacements[i, :, :] * scale_deformation,
            field_values=field_values[i],
            deformed=deformed,
            clim=None  # Normalize colors between min and max
        )
        axes.clear()
        axes.set_title(fr"$\Omega={data.eigenvalues[i]:.4f}$", fontsize=fontsize)
        axes.add_collection(patches)
        axes.set(xlim=xlim, ylim=ylim)

        if mesh == True:
            n1 = geometry.n1_blocks
            n2 = geometry.n2_blocks
            for j in np.arange(geometry.n2_blocks):
                row_block_coordinates = np.array([block_centroids[n1*j:n1*(j+1), 0] + block_displacements[i, n1*j:n1*(j+1), 0]*scale_deformation,
                                                  block_centroids[n1*j:n1*(j+1), 1] + block_displacements[i, n1*j:n1*(j+1), 1]*scale_deformation])
                axes.plot(row_block_coordinates[0, :], row_block_coordinates[1, :], 'k')

            for k in np.arange(geometry.n1_blocks):
                col_block_coordinates = np.array([block_centroids[k:n1*(n2-1)+k+1:n1, 0] + block_displacements[i, k:n1*(n2-1)+k+1:n1, 0]*scale_deformation,
                                                  block_centroids[k:n1*(n2-1)+k+1:n1, 1] + block_displacements[i, k:n1*(n2-1)+k+1:n1, 1]*scale_deformation])
                axes.plot(col_block_coordinates[0, :], col_block_coordinates[1, :], 'k')

        out_path = Path(f"{str(out_dir)}/{i:04d}.pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)  # Make sure parents directories exist
        fig.savefig(str(out_path), dpi=dpi)

    plt.close(fig)


def _compute_shell_arrow_directions(n_piece):
    """Return unit outward-normal directions (n_piece, 2) for each shell piece.

    Piece i covers [180 - 45*(i+1), 180 - 45*i] degrees (clockwise from +x=right side).
    The outward normal points at the midpoint of that arc:
        angle_i = (180 - angle_cut/2 - angle_cut * i) degrees
    """
    angle_cut = 360.0 / n_piece
    normal_angles = np.array([
        (180.0 - angle_cut / 2.0 - angle_cut * i) * np.pi / 180.0
        for i in range(n_piece)
    ])
    return np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)  # (n_piece, 2)


def generate_frames_with_arrows(
    data: SolutionData,
    field,
    out_dir,
    force_data,
    n_piece=8,
    shell_indices=None,
    arrow_scale=None,
    arrow_offset=0.0,
    max_arrow_width=None,
    arrow_color="black",
    field_values=None,
    deformed=False,
    frame_range=None,
    figsize=None,
    xlim=None,
    ylim=None,
    dpi=200,
    cmap=orange_blue_cmap(),
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    grid=False,
    save_formats=("png",),
):
    """Like generate_frames but overlays force arrows on the shell pieces.

    Arrow size (length, shaft width, head width, head length) all scale
    proportionally with instantaneous force magnitude. The head tip is fixed
    at shell_centroid + arrow_offset outward; the tail moves in/out with force.
    At maximum force the tail is at shell_centroid + max_force*arrow_scale outward.
    The frame limits are automatically expanded to include the farthest tail.

    Parameters
    ----------
    force_data : array-like, shape (n_timepoints, n_piece)
        Scalar force magnitude for each shell piece at each simulation timestep.
    n_piece : int
        Number of shell pieces (default 8).
    shell_indices : array-like of int, shape (n_piece,)
        Block indices of the shell pieces in data.block_centroids.
        Defaults to the last n_piece blocks.
    arrow_scale : float or None
        Base length scale for arrows (data units). At maximum force, arrow
        length = max_force_i * arrow_scale. Length scales proportionally at
        other force levels. If None, defaults to 3 * (structure diameter / 10).
    arrow_offset : float
        Extra outward offset for the fixed head tip position (data units).
        Default 0.
    max_arrow_width : float or None
        Shaft width (data units) when force equals its per-shell maximum.
        Width scales linearly with force ratio. If None, defaults to
        0.08 * arrow_scale.
    arrow_color : str
        Matplotlib color for the arrows.
    """
    force_data = np.asarray(force_data, dtype=float)  # (n_timepoints, n_piece)

    fmts = tuple(
        str(ext).lower().lstrip(".")
        for ext in (save_formats if isinstance(save_formats, (tuple, list)) else (save_formats,))
    )

    _field_values, min_value, max_value, fig, axes, frames = prepare_solution_figure(
        data, field, frame_range, figsize, cmap=cmap, vlim=vlim, legend_label=legend_label,
        fontsize=fontsize, ticksize=ticksize, axis=axis, field_values=field_values
    )
    block_centroids_ref = np.asarray(data.block_centroids)
    centroid_node_vectors = data.centroid_node_vectors
    bond_connectivity = data.bond_connectivity
    block_displacements = data.fields[:, 0, :, :]  # (n_timepoints, n_blocks, 3)
    clim = vlim if vlim is not None else (min_value, max_value)

    n_blocks = block_centroids_ref.shape[0]
    if shell_indices is None:
        shell_indices = np.arange(n_blocks - n_piece, n_blocks)
    shell_indices = np.asarray(shell_indices, dtype=int)

    normal_dirs = _compute_shell_arrow_directions(n_piece)  # (n_piece, 2)

    if arrow_scale is None:
        arrow_scale = 3.0 * np.linalg.norm(np.ptp(block_centroids_ref, axis=0)) / 10.0

    if max_arrow_width is None:
        max_arrow_width = 0.20 * arrow_scale

    # Global maximum force across all shells and timesteps → absolute arrow scale
    global_max_force = max(float(np.max(np.abs(force_data))), 1e-12)
    global_max_length = global_max_force * arrow_scale

    shell_centroids_ref = block_centroids_ref[shell_indices]  # (n_piece, 2)
    # Fixed head tip positions (arrow points inward toward here)
    shell_heads = shell_centroids_ref + arrow_offset * normal_dirs  # (n_piece, 2)

    # head shape multiplier: constant since both length and width scale with force
    hl = 0.5 * global_max_length / max_arrow_width

    # Expand axis limits to include farthest possible tail positions (at global max force)
    pad = 0.5 * arrow_scale
    if xlim is not None and ylim is not None:
        max_tails = shell_heads + global_max_length * normal_dirs
        xlim_eff = (min(xlim[0], max_tails[:, 0].min()) - pad,
                    max(xlim[1], max_tails[:, 0].max()) + pad)
        ylim_eff = (min(ylim[0], max_tails[:, 1].min()) - pad,
                    max(ylim[1], max_tails[:, 1].max()) + pad)
    else:
        xlim_eff = xlim
        ylim_eff = ylim

    fig.canvas.draw()

    locked_pos = axes.get_position()
    fig.set_constrained_layout(False)
    fig.set_tight_layout(False)

    axes.set_autoscale_on(False)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(xlim_eff)
    axes.set_ylim(ylim_eff)

    for i in frames:
        axes.clear()
        axes.set_aspect("equal")

        patches = generate_patch_collection(
            block_centroids=data.block_centroids,
            centroid_node_vectors=centroid_node_vectors,
            block_displacements=block_displacements[i, :, :],
            field_values=_field_values[i],
            deformed=deformed,
            clim=clim,
            cmap=cmap,
        )
        axes.add_collection(patches)

        collection_bonds = generate_bond_collection(
            data.block_centroids, centroid_node_vectors, bond_connectivity,
            block_displacements=block_displacements[i], deformed=deformed
        )
        axes.add_collection(collection_bonds)

        # Force magnitudes at this frame (clamp to available timesteps)
        t_idx = min(i, force_data.shape[0] - 1)
        forces = force_data[t_idx]  # (n_piece,)

        # Draw one quiver per shell; all dimensions ∝ absolute force
        for ip in range(n_piece):
            f_abs = abs(float(forces[ip]))
            if f_abs < 0.005 * global_max_force:
                continue
            length = f_abs * arrow_scale
            # Tail moves outward proportionally; head tip stays fixed at shell_heads[ip]
            tail_x = float(shell_heads[ip, 0]) + length * float(normal_dirs[ip, 0])
            tail_y = float(shell_heads[ip, 1]) + length * float(normal_dirs[ip, 1])
            dx = -length * float(normal_dirs[ip, 0])
            dy = -length * float(normal_dirs[ip, 1])
            w = (f_abs / global_max_force) * max_arrow_width
            axes.quiver(
                tail_x, tail_y, dx, dy,
                angles="xy", scale_units="xy", scale=1, units="xy",
                width=w, headwidth=hl, headlength=hl, headaxislength=hl,
                color=arrow_color, zorder=5,
            )

        axes.set(xlim=xlim_eff, ylim=ylim_eff)
        if not grid:
            axes.grid(False)
        if not axis:
            axes.axis("off")

        base_dir = Path(out_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fmts:
            out_path = base_dir / f"{i:04d}.{fmt}"
            fig.savefig(str(out_path), dpi=dpi, format=fmt)

    plt.close(fig)


def generate_animation_with_arrows(
    data: SolutionData,
    field,
    out_filename,
    force_data,
    n_piece=8,
    shell_indices=None,
    arrow_scale=None,
    arrow_offset=0.0,
    max_arrow_width=None,
    arrow_color="black",
    field_values=None,
    deformed=False,
    frame_range=None,
    figsize=None,
    xlim=None,
    ylim=None,
    dpi=200,
    cmap=None,
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    grid=True,
    fill_polygons=True,
    transparent_background=False,
    *,
    sync="step",
    fps=None,
    playback_speed=1.0,
    interpolate=True,
    time_range=None,
    annotate_time=True,
    time_unit="s",
):
    """Like generate_animation but overlays force arrows on the shell pieces.

    Arrow size (length, shaft width, head width, head length) all scale
    proportionally with instantaneous force magnitude. The head tip is fixed
    at shell_centroid + arrow_offset outward; the tail moves in/out with force.
    At maximum force the tail is at shell_centroid + max_force*arrow_scale outward.
    The frame limits are automatically expanded to include the farthest tail.

    Parameters
    ----------
    force_data : array-like, shape (n_timepoints, n_piece)
        Scalar force magnitude for each shell piece at each simulation timestep.
    n_piece : int
        Number of shell pieces (default 8).
    shell_indices : array-like of int, shape (n_piece,)
        Block indices of the shell pieces in data.block_centroids.
        Defaults to the last n_piece blocks.
    arrow_scale : float or None
        Base length scale for arrows (data units). At maximum force, arrow
        length = max_force_i * arrow_scale. Length scales proportionally at
        other force levels. If None, defaults to 3 * (structure diameter / 10).
    arrow_offset : float
        Extra outward offset for the fixed head tip position (data units).
        Default 0.
    max_arrow_width : float or None
        Shaft width (data units) when force equals its per-shell maximum.
        Width scales linearly with force ratio. If None, defaults to
        0.08 * arrow_scale.
    arrow_color : str
        Matplotlib color for the arrows.
    """
    force_data = np.asarray(force_data, dtype=float)  # (n_timepoints, n_piece)

    if cmap is None:
        cmap = orange_blue_cmap()

    _field_values, min_value, max_value, fig, axes, frames0 = prepare_solution_figure(
        data, field, frame_range, figsize, cmap=cmap, vlim=vlim,
        legend_label=legend_label, fontsize=fontsize, ticksize=ticksize,
        axis=axis, field_values=field_values
    )
    if vlim is None:
        clim = (float(min_value), float(max_value))
    else:
        clim = tuple(vlim)

    # Move colorbar created by prepare_solution_figure to the top
    # _cbar_axes = [ax for ax in fig.axes if ax is not axes]
    # if _cbar_axes:
    #     _cb_label = _cbar_axes[0].get_ylabel()
    #     _cbar_axes[0].remove()
    #     _cb = fig.colorbar(
    #         cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=clim[0], vmax=clim[1])),
    #         ax=axes,
    #         location="top",
    #         pad=0.05,
    #         aspect=40,
    #     )
    #     _cb.ax.tick_params(labelsize=ticksize)
    #     _cb.ax.set_xlabel(_cb_label, fontsize=fontsize)

    if transparent_background:
        fig.patch.set_alpha(0.0)
        axes.patch.set_alpha(0.0)

    axes.grid(grid)

    out_path = Path(f"{out_filename}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vertices  = data.centroid_node_vectors
    centroids = data.block_centroids
    counts    = np.array([len(v) for v in vertices])
    bond_conn = data.bond_connectivity

    n_blocks = np.asarray(centroids).shape[0]
    if shell_indices is None:
        shell_indices = np.arange(n_blocks - n_piece, n_blocks)
    shell_indices = np.asarray(shell_indices, dtype=int)

    normal_dirs = _compute_shell_arrow_directions(n_piece)  # (n_piece, 2)

    if arrow_scale is None:
        pts = np.asarray(centroids)
        arrow_scale = 3.0 * np.linalg.norm(np.ptp(pts, axis=0)) / 10.0

    if max_arrow_width is None:
        max_arrow_width = 0.20 * arrow_scale

    # Global maximum force across all shells and timesteps → absolute arrow scale
    global_max_force = max(float(np.max(np.abs(force_data))), 1e-12)
    global_max_length = global_max_force * arrow_scale

    shell_centroids_ref = np.asarray(centroids)[shell_indices]  # (n_piece, 2)
    # Fixed head tip positions (arrow points inward toward here)
    shell_heads = shell_centroids_ref + arrow_offset * normal_dirs  # (n_piece, 2)

    # head shape multiplier: constant since both length and width scale with force
    hl = 0.5 * global_max_length / max_arrow_width

    # Expand axis limits to include farthest possible tail positions (at global max force)
    pad = 0.5 * arrow_scale
    if xlim is not None and ylim is not None:
        max_tails = shell_heads + global_max_length * normal_dirs
        xlim_eff = (min(xlim[0], max_tails[:, 0].min()) - pad,
                    max(xlim[1], max_tails[:, 0].max()) + pad)
        ylim_eff = (min(ylim[0], max_tails[:, 1].min()) - pad,
                    max(ylim[1], max_tails[:, 1].max()) + pad)
    else:
        xlim_eff = None
        ylim_eff = None

    norm = Normalize(vmin=clim[0], vmax=clim[1])

    collection_blocks = generate_polycollection(
        centroids, vertices,
        field_values=_field_values[0],
        block_displacements=None,
        deformed=deformed,
        clim=clim,
        cmap=cmap,
        fill=fill_polygons
    )
    collection_blocks.set_norm(norm)
    axes.add_collection(collection_blocks)

    if bond_conn is not None:
        collection_bonds = generate_bond_collection(
            centroids, vertices, bond_conn,
            block_displacements=None,
            deformed=deformed
        )
        axes.add_collection(collection_bonds)
    else:
        collection_bonds = None

    if xlim_eff is not None:
        axes.set(xlim=xlim_eff, ylim=ylim_eff)
    else:
        axes.autoscale(enable=True)
        axes.autoscale_view()

    if annotate_time:
        time_text = axes.text(
            0.02, 0.98, "", transform=axes.transAxes,
            va="top", ha="left", fontsize=fontsize
        )
    else:
        time_text = None

    # Mutable list to track per-frame quiver artists (created/destroyed each frame)
    _quiver_artists = []

    def _clear_arrows():
        for q in _quiver_artists:
            q.remove()
        _quiver_artists.clear()

    def _draw_arrows(forces):
        _clear_arrows()
        for ip in range(n_piece):
            f_abs = abs(float(forces[ip]))
            if f_abs < 0.005 * global_max_force:
                continue
            length = f_abs * arrow_scale
            # Tail moves outward proportionally; head tip stays fixed at shell_heads[ip]
            tail_x = float(shell_heads[ip, 0]) + length * float(normal_dirs[ip, 0])
            tail_y = float(shell_heads[ip, 1]) + length * float(normal_dirs[ip, 1])
            dx = -length * float(normal_dirs[ip, 0])
            dy = -length * float(normal_dirs[ip, 1])
            w = (f_abs / global_max_force) * max_arrow_width
            q = axes.quiver(
                tail_x, tail_y, dx, dy,
                angles="xy", scale_units="xy", scale=1, units="xy",
                width=w, headwidth=hl, headlength=hl, headaxislength=hl,
                color=arrow_color, zorder=5,
            )
            _quiver_artists.append(q)

    # Build frame-time mapping
    t = getattr(data, "timepoints", None)
    if t is None:
        t = np.arange(data.fields.shape[0], dtype=float)
    frame_times, idx, w, fps_eff, (lo, hi) = _build_time_mapping(
        t, sync, fps, playback_speed, interpolate, time_range
    )

    def _get_forces_at(k):
        """Return interpolated force array (n_piece,) for animation frame k."""
        if sync == "step" and frame_range is not None:
            i = frames0[k]
            return force_data[min(i, force_data.shape[0] - 1)]
        elif w is None:
            i = idx[k]
            return force_data[min(i, force_data.shape[0] - 1)]
        else:
            j = idx[k]
            tau = w[k]
            j1 = min(j + 1, force_data.shape[0] - 1)
            return (1.0 - tau) * force_data[j] + tau * force_data[j1]

    def get_state_at(k):
        if sync == "step" and frame_range is not None:
            i = frames0[k]
            time_val = t[i]
            DOFs = data.fields[i, 0, :, :]
            vals = _field_values[i]
        elif w is None:
            i = idx[k]
            time_val = frame_times[k]
            DOFs = data.fields[i, 0, :, :]
            vals = _field_values[i]
        else:
            j = idx[k]
            tau = w[k]
            time_val = frame_times[k]
            DOFs0 = data.fields[j,   0, :, :]
            DOFs1 = data.fields[j+1, 0, :, :]
            DOFs  = _interp(DOFs0, DOFs1, tau)
            vals0 = _field_values[j]
            vals1 = _field_values[j+1]
            vals  = _interp(vals0, vals1, tau)

        block_coords = current_coordinates(vertices, centroids, DOFs[:, -1], DOFs[:, :2])
        return block_coords, vals, time_val

    # Prepare first frame
    bc0, vals0, t0 = get_state_at(0)
    polys0 = split_polygons(bc0, counts)
    collection_blocks.set_verts(polys0)
    collection_blocks.set_array(vals0)
    if collection_bonds is not None:
        collection_bonds.set_segments(bc0[bond_conn])
    if time_text is not None:
        time_text.set_text(f"t = {t0:.2g} {time_unit}")
    _draw_arrows(_get_forces_at(0))

    # --- Freeze axes layout so it doesn't shift between frames ---
    # blit=False causes constrained_layout to re-run every frame. When quiver
    # artists of varying shaft widths are added/removed, the layout engine's
    # bbox computation changes slightly each frame, shifting the axes in pixel
    # space. Because axis("equal") ties data scale to pixel dimensions, even a
    # 1-2 px shift makes the structure appear to change size. Fix: draw once to
    # lock in the layout, then disable constrained_layout and pin the position.
    fig.canvas.draw()

    locked_pos = axes.get_position()
    fig.set_constrained_layout(False)
    fig.set_tight_layout(False)

    axes.set_position(locked_pos)
    axes.set_autoscale_on(False)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(xlim_eff)
    axes.set_ylim(ylim_eff)


    def _animate(k):
        bc, vals, tt = get_state_at(k)
        forces = _get_forces_at(k)
        polys = split_polygons(bc, counts)
        collection_blocks.set_verts(polys)
        collection_blocks.set_array(vals)
        if collection_bonds is not None:
            collection_bonds.set_segments(bc[bond_conn])
        if time_text is not None:
            time_text.set_text(f"t = {tt:.2g} {time_unit}")
        if xlim_eff is not None:
            axes.set(xlim=xlim_eff, ylim=ylim_eff)
        axes.set_xlim(xlim_eff)
        axes.set_ylim(ylim_eff)
        axes.set_aspect("equal", adjustable="box")
        axes.set_position(locked_pos)
        _draw_arrows(forces)
        # blit=False: return value not used for blitting but required by FuncAnimation
        return []

    frames_count = len(frame_times) if sync == "time" else (frames0 if frame_range is not None else np.arange(lo, hi+1)).__len__()
    anim = animation.FuncAnimation(
        fig, _animate,
        frames=frames_count, blit=False
    )

    writer_kwargs = dict(fps=fps_eff, dpi=dpi)
    writer_extra = dict(writer="ffmpeg", extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), **writer_extra, **writer_kwargs)
    plt.close(fig)


def generate_frames(data: SolutionData, field, out_dir, field_values=None, deformed=False, frame_range=None, figsize=None, xlim=None, ylim=None, dpi=200, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True, grid=False, save_formats=("png",)):
    """
    save_formats: tuple of file extensions (no dot), e.g. ("png",) or ("png", "eps").
    """

    fmts = tuple(
        str(ext).lower().lstrip(".")
        for ext in (save_formats if isinstance(save_formats, (tuple, list)) else (save_formats,))
    )

    _field_values, min_value, max_value, fig, axes, frames = prepare_solution_figure(
        data, field, frame_range, figsize, cmap=cmap, vlim=vlim, legend_label=legend_label, fontsize=fontsize, ticksize=ticksize, axis=axis, field_values=field_values
    )
    block_centroids = data.block_centroids
    centroid_node_vectors = data.centroid_node_vectors
    bond_connectivity = data.bond_connectivity
    block_displacements = data.fields[:, 0, :, :]
    clim = vlim if vlim is not None else (min_value, max_value)

    for i in frames:
        # Delete old patches
        axes.clear()
        # Restore equal aspect after clear() — axes.clear() resets aspect to "auto",
        # causing the first frame to render at a different size than subsequent frames
        # when constrained_layout=True caches the pre-loop axis("equal") state.
        axes.set_aspect("equal")
        # Draw new patches
        patches = generate_patch_collection(
            block_centroids=block_centroids,
            centroid_node_vectors=centroid_node_vectors,
            block_displacements=block_displacements[i, :, :],
            field_values=_field_values[i],
            deformed=deformed,
            clim=clim,
            cmap=cmap,
        )
        axes.add_collection(patches)
        # Generate collection of bonds as lines
        collection_bonds = generate_bond_collection(
            block_centroids, centroid_node_vectors, bond_connectivity, block_displacements=block_displacements[i], deformed=deformed)
        axes.add_collection(collection_bonds)

        axes.set(xlim=xlim, ylim=ylim)
        if not grid:
            axes.grid(False)
        if not axis:
            axes.axis("off")

        base_dir = Path(out_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fmts:
            out_path = base_dir / f"{i:04d}.{fmt}"
            fig.savefig(str(out_path), dpi=dpi, format=fmt)

    plt.close(fig)

# Add new function to calculate the number of vertices per polygon
def split_polygons(block_coords_concat: np.ndarray, counts: np.ndarray):
    """
    block_coords_concat: (N_total, 2)
    counts: (n_blocks,) number of vertices per polygon
    returns: list of (n_i, 2) arrays
    """
    counts = np.asarray(counts, dtype=int)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array of per-polygon vertex counts")
    cuts = np.cumsum(counts)[:-1]
    return np.split(block_coords_concat, cuts)

def _build_time_mapping(t, sync, target_fps, playback_speed, interpolate, time_range):
    t = np.asarray(t).astype(float)
    assert t.ndim == 1 and len(t) >= 2, "timepoints must be 1D with length >=2"

    # apply time_range trimming (by time, not index)
    if time_range is not None:
        t0, t1 = time_range
        mask = (t >= t0) & (t <= t1)
        # keep at least endpoints if mask is too strict
        if not mask.any():
            raise ValueError("time_range excludes all data")
        idx = np.where(mask)[0]
        lo, hi = idx[0], idx[-1]
        t = t[lo:hi+1]
        lohi = (lo, hi)
    else:
        lohi = (0, len(t)-1)

    if sync == "step":
        # 1 sim step -> 1 frame
        frame_times = t.copy()
        frame_indices = np.arange(len(t), dtype=int)
        weights = None  # not used
        fps = target_fps if target_fps is not None else 20  # any fps is okay; duration = len(t)/fps
    else:
        # Real-time playback using timestamps
        if target_fps is None:
            target_fps = 30
        dt = 1.0 / (target_fps * max(playback_speed, 1e-12))
        frame_times = np.arange(t[0], t[-1] + 0.5*dt, dt)

        # nearest or linear mapping from frame_times to sim indices
        if not interpolate:
            # nearest neighbor
            j = np.searchsorted(t, frame_times, side="left")
            j = np.clip(j, 0, len(t)-1)
            # choose nearer of j and j-1
            j_minus = np.clip(j-1, 0, len(t)-1)
            choose_minus = (j > 0) & (np.abs(frame_times - t[j_minus]) <= np.abs(frame_times - t[j]))
            frame_indices = np.where(choose_minus, j_minus, j).astype(int)
            weights = None
        else:
            # linear interpolation weights
            j = np.searchsorted(t, frame_times, side="right") - 1
            j = np.clip(j, 0, len(t)-2)
            tau = (frame_times - t[j]) / np.maximum(t[j+1] - t[j], 1e-12)
            frame_indices = j
            weights = tau.astype(float)

        fps = target_fps

    return frame_times, frame_indices, weights, fps, lohi

def _interp(a0, a1, w):
    # linear interpolation that works for arrays
    return (1.0 - w) * a0 + w * a1

# general version
"""
def generate_animation(
    data: SolutionData,
    field,
    out_filename,
    field_values=None,
    deformed=False,
    frame_range=None,               # kept for backward compat (indices)
    figsize=None,
    xlim=None,
    ylim=None,
    dpi=200,
    cmap=None,
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    grid=True,
    block_idx=None,
    *,
    # NEW: time-aware controls
    sync="step",                    # 'step' or 'time'
    fps=None,                # used if sync='time' (default 30 if None)
    playback_speed=1.0,             # 0.5 half-speed, 2.0 double-speed
    interpolate=True,               # interpolate DOFs/field when sync='time'
    time_range=None,                # (t0, t1) in data.timepoints space
    annotate_time=True,             # draw time text
    time_unit="s",                  # time label
):

    if block_idx is None:
        block_idx = np.arange( data.fields.shape[2] )
        
    if cmap is None:
        cmap = orange_blue_cmap()

    # --- Figure & axes from your existing prepare_solution_figure ---
    _field_values, min_value, max_value, fig, axes, frames0 = prepare_solution_figure(
        data, field, frame_range, figsize, cmap=cmap, vlim=vlim,
        legend_label=legend_label, fontsize=fontsize, ticksize=ticksize,
        axis=axis, field_values=field_values
    )
    
    # consistent color normalization across entire movie
    if vlim is None:
        clim = (float(min_value), float(max_value))
    else:
        clim = tuple(vlim)

    axes.grid(grid)

    out_path = Path(f"{out_filename}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Geometry ---
    vertices  = data.centroid_node_vectors          # list/tuple of (ni,2)
    centroids = data.block_centroids                # (nb,2)
    counts    = np.array([len(v) for v in vertices[block_idx]])
    
    bond_conn = data.bond_connectivity

    # Colormap normalization shared by artists
    norm = Normalize(vmin=clim[0], vmax=clim[1])
    
    # Build initial polygons (undeformed at step 0)
    collection_blocks = generate_polycollection(
        centroids[block_idx], vertices[block_idx],
        field_values=_field_values[0,block_idx],
        block_displacements=None,
        deformed=deformed,
        clim=clim,
        cmap=cmap
    )
    collection_blocks.set_norm(norm)
    axes.add_collection(collection_blocks)

    if bond_conn is not None:
        collection_bonds = generate_bond_collection(
            centroids, vertices, bond_conn,
            block_displacements=None,
            deformed=deformed
        )
        axes.add_collection(collection_bonds)
    else:
        collection_bonds = None

    # Axis limits (stable over time)
    # axes.set_aspect('auto')
    if xlim is not None and ylim is not None:
        axes.set(xlim=xlim, ylim=ylim)
    else:
        # If not provided, compute once from undeformed geometry
        axes.autoscale(enable=True)
        axes.autoscale_view()

    # Optional time annotation
    if annotate_time:
        time_text = axes.text(
            0.02, 0.98, "", transform=axes.transAxes,
            va="top", ha="left", fontsize=fontsize
        )
    else:
        time_text = None

    # --- Build frame mapping ---
    t = getattr(data, "timepoints", None)
    if t is None:
        # fallback: treat indices like time
        t = np.arange(data.fields.shape[0], dtype=float)
    frame_times, idx, w, fps_eff, (lo, hi) = _build_time_mapping(
        t, sync, fps, playback_speed, interpolate, time_range
    )

    # Keep references into data arrays
    # shape: fields[time, ?, block, 3] with last dim [dx, dy, rot]
    # your code used data.fields[:, 0, :, :]
    def get_state_at(k, tau=None):
        if sync == "step" and frame_range is not None:
            # if user passed old frame_range (indices), use frames0
            i = frames0[k]
            time_val = t[i]
            DOFs = data.fields[i, 0, :, :]
            vals = _field_values[i]
        elif w is None:
            # nearest
            i = idx[k]
            time_val = frame_times[k]
            DOFs = data.fields[i, 0, :, :]
            vals = _field_values[i]
        else:
            # linear interpolation between j=idx[k] and j+1
            j = idx[k]
            tau = w[k]
            time_val = frame_times[k]
            DOFs0 = data.fields[j,   0, :, :]
            DOFs1 = data.fields[j+1, 0, :, :]
            DOFs  = _interp(DOFs0, DOFs1, tau)
            vals0 = _field_values[j]
            vals1 = _field_values[j+1]
            vals  = _interp(vals0, vals1, tau)

        block_coords_all = current_coordinates(vertices, centroids, DOFs[:, -1], DOFs[:, :2])
        block_coords = current_coordinates(vertices[block_idx], centroids[block_idx], DOFs[block_idx, -1], DOFs[block_idx, :2])
        return block_coords_all, block_coords, vals, time_val

    # Prepare first frame
    bc0_all, bc0, vals0, t0 = get_state_at(0)
    polys0 = split_polygons(bc0, counts)
    collection_blocks.set_verts(polys0)
    collection_blocks.set_array(vals0)
    if collection_bonds is not None:
        collection_bonds.set_segments(bc0_all[bond_conn])
    if time_text is not None:
        time_text.set_text(f"t = {t0:.6g} {time_unit}")

    # blitting init
    def _init():
        collection_blocks.set_array(vals0)
        collection_blocks.set_verts(polys0)
        artists = [collection_blocks]
        if collection_bonds is not None:
            collection_bonds.set_segments(bc0_all[bond_conn])
            artists.append(collection_bonds)
        if time_text is not None:
            time_text.set_text(f"t = {t0:.6g} {time_unit}")
            artists.append(time_text)
        return tuple(artists)

    def _animate(k):
        bc_all, bc, vals, tt = get_state_at(k)
        polys = split_polygons(bc, counts)
        collection_blocks.set_verts(polys)
        collection_blocks.set_array(vals)
        if collection_bonds is not None:
            collection_bonds.set_segments(bc_all[bond_conn])
        if time_text is not None:
            time_text.set_text(f"t = {tt:.6g} {time_unit}")
        # keep limits stable
        if xlim is not None and ylim is not None:
            axes.set(xlim=xlim, ylim=ylim)
        return (collection_blocks, collection_bonds, time_text) if collection_bonds and time_text else \
               (collection_blocks, collection_bonds) if collection_bonds else \
               (collection_blocks,)

    # Build & save animation
    frames_count = len(frame_times) if sync == "time" else (frames0 if frame_range is not None else np.arange(lo, hi+1)).__len__()
    anim = animation.FuncAnimation(
        fig, _animate, init_func=_init,
        frames=frames_count, blit=True
    )

    # Note: save() uses fps here; interval is irrelevant for file output
    writer_kwargs = dict(fps=fps_eff, dpi=dpi)
    # Optional: control quality/compat for media players
    writer_extra = dict(writer="ffmpeg", extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), **writer_extra, **writer_kwargs)
    plt.close(fig)
"""

def generate_animation(
    data: SolutionData,
    field,
    out_filename,
    field_values=None,
    deformed=False,
    frame_range=None,               # kept for backward compat (indices)
    figsize=None,
    xlim=None,
    ylim=None,
    dpi=200,
    cmap=None,
    vlim=None,
    legend_label=None,
    fontsize=14,
    ticksize=14,
    axis=True,
    grid=True,
    fill_polygons=True,             # if False, show only edges (no fill)
    transparent_background=False,    # if True, make background transparent
    *,
    # NEW: time-aware controls
    sync="step",                    # 'step' or 'time'
    fps=None,                # used if sync='time' (default 30 if None)
    playback_speed=1.0,             # 0.5 half-speed, 2.0 double-speed
    interpolate=True,               # interpolate DOFs/field when sync='time'
    time_range=None,                # (t0, t1) in data.timepoints space
    annotate_time=True,             # draw time text
    time_unit="s",                  # time label
):
    """
    Make an animation that is consistent with simulation timestamps.

    sync='step': 1 simulation step -> 1 frame (simple).
    sync='time': use data.timepoints to resample to real-time at target_fps; supports linear interpolation.
    
    Parameters:
    -----------
    fill_polygons : bool, default=True
        If True, polygons are filled with colors based on field_values.
        If False, only polygon edges are shown (no fill).
    transparent_background : bool, default=False
        If True, the figure and axes backgrounds are made transparent.
        Note: MP4 format may not fully support transparency; consider using other formats if needed.
    """

    if cmap is None:
        cmap = orange_blue_cmap()

    # --- Figure & axes from your existing prepare_solution_figure ---
    _field_values, min_value, max_value, fig, axes, frames0 = prepare_solution_figure(
        data, field, frame_range, figsize, cmap=cmap, vlim=vlim,
        legend_label=legend_label, fontsize=fontsize, ticksize=ticksize,
        axis=axis, field_values=field_values
    )
    # consistent color normalization across entire movie
    if vlim is None:
        clim = (float(min_value), float(max_value))
    else:
        clim = tuple(vlim)

    # Set transparent background if requested
    if transparent_background:
        fig.patch.set_alpha(0.0)
        axes.patch.set_alpha(0.0)

    axes.grid(grid)

    out_path = Path(f"{out_filename}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Geometry ---
    vertices  = data.centroid_node_vectors          # list/tuple of (ni,2)
    centroids = data.block_centroids                # (nb,2)
    counts    = np.array([len(v) for v in vertices])
    bond_conn = data.bond_connectivity

    # Colormap normalization shared by artists
    norm = Normalize(vmin=clim[0], vmax=clim[1])

    # Build initial polygons (undeformed at step 0)
    collection_blocks = generate_polycollection(
        centroids, vertices,
        field_values=_field_values[0],
        block_displacements=None,
        deformed=deformed,
        clim=clim,
        cmap=cmap,
        fill=fill_polygons
    )
    collection_blocks.set_norm(norm)
    axes.add_collection(collection_blocks)

    if bond_conn is not None:
        collection_bonds = generate_bond_collection(
            centroids, vertices, bond_conn,
            block_displacements=None,
            deformed=deformed
        )
        axes.add_collection(collection_bonds)
    else:
        collection_bonds = None

    # Axis limits (stable over time)
    # axes.set_aspect('auto')
    if xlim is not None and ylim is not None:
        axes.set(xlim=xlim, ylim=ylim)
    else:
        # If not provided, compute once from undeformed geometry
        axes.autoscale(enable=True)
        axes.autoscale_view()

    # Optional time annotation
    if annotate_time:
        time_text = axes.text(
            0.02, 0.98, "", transform=axes.transAxes,
            va="top", ha="left", fontsize=fontsize
        )
    else:
        time_text = None

    # --- Build frame mapping ---
    t = getattr(data, "timepoints", None)
    if t is None:
        # fallback: treat indices like time
        t = np.arange(data.fields.shape[0], dtype=float)
    frame_times, idx, w, fps_eff, (lo, hi) = _build_time_mapping(
        t, sync, fps, playback_speed, interpolate, time_range
    )

    # Keep references into data arrays
    # shape: fields[time, ?, block, 3] with last dim [dx, dy, rot]
    # your code used data.fields[:, 0, :, :]
    def get_state_at(k, tau=None):
        """Return (block_coords, values_for_colormap, time_value) for frame k."""
        if sync == "step" and frame_range is not None:
            # if user passed old frame_range (indices), use frames0
            i = frames0[k]
            time_val = t[i]
            DOFs = data.fields[i, 0, :, :]
            vals = _field_values[i]
        elif w is None:
            # nearest
            i = idx[k]
            time_val = frame_times[k]
            DOFs = data.fields[i, 0, :, :]
            vals = _field_values[i]
        else:
            # linear interpolation between j=idx[k] and j+1
            j = idx[k]
            tau = w[k]
            time_val = frame_times[k]
            DOFs0 = data.fields[j,   0, :, :]
            DOFs1 = data.fields[j+1, 0, :, :]
            DOFs  = _interp(DOFs0, DOFs1, tau)
            vals0 = _field_values[j]
            vals1 = _field_values[j+1]
            vals  = _interp(vals0, vals1, tau)

        block_coords = current_coordinates(vertices, centroids, DOFs[:, -1], DOFs[:, :2])
        return block_coords, vals, time_val

    # Prepare first frame
    bc0, vals0, t0 = get_state_at(0)
    polys0 = split_polygons(bc0, counts)
    collection_blocks.set_verts(polys0)
    collection_blocks.set_array(vals0)
    if collection_bonds is not None:
        collection_bonds.set_segments(bc0[bond_conn])
    if time_text is not None:
        time_text.set_text(f"t = {t0:.6g} {time_unit}")

    # blitting init
    def _init():
        collection_blocks.set_array(vals0)
        collection_blocks.set_verts(polys0)
        artists = [collection_blocks]
        if collection_bonds is not None:
            collection_bonds.set_segments(bc0[bond_conn])
            artists.append(collection_bonds)
        if time_text is not None:
            time_text.set_text(f"t = {t0:.2g} {time_unit}")
            artists.append(time_text)
        return tuple(artists)

    def _animate(k):
        bc, vals, tt = get_state_at(k)
        polys = split_polygons(bc, counts)
        collection_blocks.set_verts(polys)
        collection_blocks.set_array(vals)
        if collection_bonds is not None:
            collection_bonds.set_segments(bc[bond_conn])
        if time_text is not None:
            time_text.set_text(f"t = {tt:.2g} {time_unit}")
        # keep limits stable
        if xlim is not None and ylim is not None:
            axes.set(xlim=xlim, ylim=ylim)
        return (collection_blocks, collection_bonds, time_text) if collection_bonds and time_text else \
               (collection_blocks, collection_bonds) if collection_bonds else \
               (collection_blocks,)

    # Build & save animation
    frames_count = len(frame_times) if sync == "time" else (frames0 if frame_range is not None else np.arange(lo, hi+1)).__len__()
    anim = animation.FuncAnimation(
        fig, _animate, init_func=_init,
        frames=frames_count, blit=True
    )

    # Note: save() uses fps here; interval is irrelevant for file output
    writer_kwargs = dict(fps=fps_eff, dpi=dpi)
    # Optional: control quality/compat for media players
    writer_extra = dict(writer="ffmpeg", extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), **writer_extra, **writer_kwargs)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(prog="BlockyMetamaterials plotting script")
    parser.add_argument("-i", "--data-file", help='Destination to pkl data file', required=True)
    parser.add_argument("-o", "--out", help="Output path.", required=True)
    parser.add_argument("-f", "--field", help="Field to plot.", type=str, default="v")
    parser.add_argument("-d", "--deformed", help='Plot on deformed configuration.',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fps", help="Frame rate of the animation.", type=int, default=20)
    parser.add_argument("--dpi", help="DPI.", type=int, default=200)
    parser.add_argument("--figsize", help="Figure size.", type=float, nargs=2, default=(16, 9))
    parser.add_argument("-a", "--animation", help='Produce animation or frames.',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tex", help='Use TeX fonts.', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fontsize", help='Font size.', type=int, default=20)

    parser.add_argument("-e", help='Plot Energy', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clear", help='Clear output', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-n", help='number of processes to use', type=int, default=1)

    args = parser.parse_args()

    if args.tex:
        plt.style.use(["science"])  # enable latex fonts
    plt.rc('font', size=args.fontsize)  # font size

    data = load_data(args.data_file)

    if args.animation:
        # Generate animation
        generate_animation(data=data, field=args.field, out_filename=args.out,
                           deformed=args.deformed, fps=args.fps, dpi=args.dpi, figsize=args.figsize)
    else:
        # Generate frames
        if args.n > 1:
            # In parallel
            print("Generating images in parallel.\nThere is a large overhead and may be slow.")
            global generate_frames_parallel  # Needed for multiprocessing

            def generate_frames_parallel(i):
                return generate_frames(data=data, field=args.field, out_dir=args.out,
                                       deformed=args.deformed, figsize=args.figsize, frame_range=[i])
            with Pool(args.n) as pool:
                pool.map(generate_frames_parallel, range(len(data.timepoints)))
        else:
            # Sequentially
            generate_frames(data=data, field=args.field, out_dir=args.out, deformed=args.deformed, figsize=args.figsize)

    if args.e:
        plot_energy(data)


        
def plot_multiple_geometries(geometry_data, figsize=(10, 10), colors=None, xlim=None, ylim=None):
    """
    Plots multiple geometries in the same figure.

    Parameters:
    - geometry_data: List of dictionaries, each containing parameters for a single geometry.
                     Each dictionary should include:
                     - block_centroids
                     - centroid_node_vectors
                     - bond_connectivity
                     - block_displacements (optional)
                     - deformed (optional)
    - figsize: Tuple for the figure size.
    - colors: List of colors for each geometry.
    - xlim, ylim: (Optional) Axis limits for the plot.

    Returns:
    - fig, ax: Matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.axis("equal")
    
    # Default colors if none are provided
    if colors is None:
        colors = ["#2980b9", "#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f"]
    
    for i, geom in enumerate(geometry_data):
        color = colors[i % len(colors)]
        
        block_centroids = geom["block_centroids"]
        centroid_node_vectors = geom["centroid_node_vectors"]
        bond_connectivity = geom["bond_connectivity"]
        block_displacements = geom.get("block_displacements", None)
        deformed = geom.get("deformed", False)

        # Generate collection of blocks as polygons
        if deformed and block_displacements is not None:
            patches = generate_patch_collection(block_centroids, centroid_node_vectors,
                                            block_displacements=block_displacements, deformed=deformed)
        else:
            patches = generate_patch_collection(block_centroids, centroid_node_vectors,
                                            block_displacements=block_displacements, deformed=deformed)
        patches.set(color=color, edgecolor="black", linewidth=0.5)
        ax.add_collection(patches)
        
        # Generate collection of bonds as lines
        collection_bonds = generate_bond_collection(
            block_centroids, centroid_node_vectors, bond_connectivity, 
            block_displacements=block_displacements, deformed=deformed)
        ax.add_collection(collection_bonds)
    
    # Compute xlim and ylim if not provided
    if xlim is None or ylim is None:
        all_points = []
        for geom in geometry_data:
            block_centroids = geom["block_centroids"]
            centroid_node_vectors = geom["centroid_node_vectors"]
            block_displacements = geom.get("block_displacements", None)
            deformed = geom.get("deformed", False)
            
            if deformed and block_displacements is not None:
                points = current_coordinates(centroid_node_vectors, block_centroids,
                                             block_displacements[:, -1], block_displacements[:, :2]).reshape((-1, 2))
            else:
                points = (block_centroids[:, None, :] + centroid_node_vectors).reshape((-1, 2))
            all_points.append(points)
        
        all_points = np.vstack(all_points)
        _xlim, _ylim = compute_xy_limits(all_points)
        xlim = _xlim if xlim is None else xlim
        ylim = _ylim if ylim is None else ylim
    
    ax.set(xlim=xlim, ylim=ylim)
    return fig, ax



if __name__ == "__main__":
    main()
