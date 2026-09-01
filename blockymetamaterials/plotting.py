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

    # return LineCollection(node_coords[bond_connectivity], color="black", linewidth=0.5)
    return LineCollection(node_coords[bond_connectivity], color="black", linewidth=1.0)


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
    # patches.set(edgecolor="black", linewidth=0.5)
    patches.set(edgecolor="#333333", linewidth=0.25)
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
        label=_legend_label,
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


def generate_frames(data: SolutionData, field, out_dir, field_values=None, deformed=False, frame_range=None, figsize=None, xlim=None, ylim=None, dpi=200, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True, grid=False):
    """
    docstring
    """

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

        out_path = Path(f"{str(out_dir)}/{i:04d}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)  # Make sure parents directories exist
        fig.savefig(str(out_path), dpi=dpi)

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
            time_text.set_text(f"t = {t0:.6g} {time_unit}")
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
