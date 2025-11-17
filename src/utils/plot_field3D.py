import torch
import numpy as np
from matplotlib import pyplot as plt

import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

# plt.rcParams.update({
#     'font.size': 16,               # Default font size
#     'font.sans-serif': ['Arial'],  # Specific font for sans-serif
#     'axes.titlesize': 24,             # Font size for axes titles
#     'axes.labelsize': 20,             # Font size for x and y labels
#     'xtick.labelsize': 20,            # Font size for x-axis tick labels
#     'ytick.labelsize': 20,            # Font size for y-axis tick labels
#     'legend.fontsize': 18,            # Font size for legend
#     'figure.titlesize': 28,
# })

import plotly.graph_objects as go

def plot_3slices_plotly(data, fname="slices3d",
                        opacity=0.45, colorscale="Greys", show_colorbar=True):
    """
    data: 3D numpy array with shape (sx, sy, sz)
    Saves an interactive HTML and a static PNG (requires `pip install -U kaleido`).
    """
    sx, sy, sz = data.shape
    x = np.arange(sx)
    y = np.arange(sy)
    z = np.arange(sz)

    # Mid-plane indices
    ix = sx // 2
    iy = sy // 2
    iz = sz // 2

    # Slices (match your original orientation: XY @ z=mid, YZ @ x=mid, ZX @ y=mid)
    xy_slice = data[:, :, iz]
    yz_slice = data[ix, :, :]
    zx_slice = data[:, iy, :].T  # consistent with your function where zx_slice = data[:, int(sy/2), :].T

    # Build plane coordinates
    # XY plane at constant z = iz
    X_xy, Y_xy = np.meshgrid(x, y, indexing="ij")  # shape (sx, sy)
    Z_xy = np.full_like(X_xy, fill_value=iz, dtype=float)

    # YZ plane at constant x = ix
    Y_yz, Z_yz = np.meshgrid(y, z, indexing="ij")  # shape (sy, sz)
    X_yz = np.full_like(Y_yz, fill_value=ix, dtype=float)

    # ZX plane at constant y = iy
    Z_zx, X_zx = np.meshgrid(z, x, indexing="ij")  # shape (sz, sx)
    Y_zx = np.full_like(Z_zx, fill_value=iy, dtype=float)

    # Create figure
    fig = go.Figure()

    # XY surface
    fig.add_surface(
        x=X_xy, y=Y_xy, z=Z_xy,
        surfacecolor=xy_slice,
        colorscale=colorscale,
        showscale=show_colorbar,
        opacity=opacity,
        cmin=np.min(data), cmax=np.max(data),
    )

    # YZ surface
    fig.add_surface(
        x=X_yz, y=Y_yz, z=Z_yz,
        surfacecolor=yz_slice,
        colorscale=colorscale,
        showscale=False,           # avoid multiple colorbars
        opacity=opacity,
        cmin=np.min(data), cmax=np.max(data),
    )

    # ZX surface
    fig.add_surface(
        x=X_zx, y=Y_zx, z=Z_zx,
        surfacecolor=zx_slice,
        colorscale=colorscale,
        showscale=False,
        opacity=opacity,
        cmin=np.min(data), cmax=np.max(data),
    )

    # Scene & layout (true aspect, hide axes for a clean look)
    fig.update_scenes(
        aspectmode="data",
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # Save interactive HTML
    print(f"Saving interactive HTML to {fname}.html")
    fig.write_html(f"{fname}.html", include_plotlyjs="cdn")

    # Save static image (requires: pip install -U kaleido)
    try:
        fig.write_image(f"{fname}.png", scale=2, format="png")  # 2x for higher DPI
    except ValueError as e:
        print("Static image export needs 'kaleido'. Install with:\n  pip install -U kaleido")
        print("Error:", e)

    return fig

def plot_3slices(data, fname=None, stride = 1, my_cmap = plt.cm.binary, cm_zero_center=True, title=None, ticks=True, colorbar=True):
    # using3D()
    sx, sy, sz = data.shape
    xy_slice = data[:, :, int(sz/2)]
    yz_slice = data[int(sx/2), :, :]
    zx_slice = data[:, int(sy/2), :].T

    x = list(range(sx))
    y = list(range(sy))
    z = list(range(sz))

    fig = plt.figure(figsize=(12,4))
    ax = plt.subplot(131, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x1 = np.array([0*i + j for j in x for i in y]).reshape((sx,sy))
    y1 = np.array([i + 0*j for j in x for i in y]).reshape((sx,sy))
    z1 = sz/2*np.ones((len(x), len(y)))
    if cm_zero_center:
        vm = max(np.max(xy_slice)+1e-6, -np.min(xy_slice)-1e-6)
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(xy_slice), vmax=np.max(xy_slice))

    # plt.figure()
    # plt.imshow(xy_slice)
    # plt.savefig("debug.png")
    # plt.close()

    surf = ax.plot_surface(x1.T, y1.T, z1.T, rstride=stride, cstride=stride, facecolors=my_cmap(norm(xy_slice.T)))
    ax.set_zlim((0,sz))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array(xy_slice)
    if colorbar:
        cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # transparent background
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.grid(False)

    ax = plt.subplot(132, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x2 = sx/2*np.ones((len(y), len(z)))
    y2 = np.array([0*i + j for j in y for i in z]).reshape((sy,sz))
    z2 = np.array([i + 0*j for j in y for i in z]).reshape((sy,sz))
    if cm_zero_center:
        vm = max(np.max(yz_slice)+1e-6, -np.min(yz_slice)-1e-6)
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(yz_slice), vmax=np.max(yz_slice))
    ax.plot_surface(x2, y2, z2, rstride=stride,cstride=stride, facecolors=my_cmap(norm(yz_slice)))
    ax.set_xlim((0,sx))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array(yz_slice)
    if colorbar:
        cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # transparent background
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.grid(False)

    ax = plt.subplot(133, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x3 = np.array([i + 0*j for j in z for i in x]).reshape((sz,sx))
    y3 = sy/2*np.ones((len(z), len(x)))
    z3 = np.array([0*i + j for j in z for i in x]).reshape((sz,sx))
    if cm_zero_center:
        vm = max(np.max(zx_slice)+1e-6, -np.min(zx_slice)-1e-6)
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(zx_slice), vmax=np.max(zx_slice))
    ax.plot_surface(x3, y3, z3, rstride=stride,cstride=stride, facecolors=my_cmap(norm(zx_slice)))
    ax.set_ylim((0,sy))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array(zx_slice)
    if colorbar:
        cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # transparent background
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.grid(False)

    plt.tight_layout()
    # plt.subplots_adjust(wspace=-0.5)

    if title:
        plt.title(title)

    if fname:
        plt.savefig(fname, dpi=100, transparent=True)
        plt.close()
    else:
        return fig

def plot_3slices_together(data, fname=None, stride = 1, my_cmap = plt.cm.binary, cm_zero_center=True, title=None, ticks=True, colorbar=True):
    # using3D()
    sx, sy, sz = data.shape
    xy_slice = data[:, :, int(sz/2)]
    yz_slice = data[int(sx/2), :, :]
    zx_slice = data[:, int(sy/2), :].T

    x = list(range(sx))
    y = list(range(sy))
    z = list(range(sz))

    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    x3 = np.array([i + 0*j for j in z for i in x]).reshape((sz,sx))
    y3 = sy/2*np.ones((len(z), len(x)))
    z3 = np.array([0*i + j for j in z for i in x]).reshape((sz,sx))
    if cm_zero_center:
        vm = max(np.max(zx_slice), -np.min(zx_slice))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(zx_slice), vmax=np.max(zx_slice))
    
    colors_plane = my_cmap(norm(zx_slice))
    colors_plane[...,3] = 0.4
    ax.plot_surface(x3, y3, z3, rstride=stride,cstride=stride, facecolors=colors_plane)
    ax.set_ylim((0,sy))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array(zx_slice)
    if colorbar:
        cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # transparent background
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

    x1 = np.array([0*i + j for j in x for i in y]).reshape((sx,sy))
    y1 = np.array([i + 0*j for j in x for i in y]).reshape((sx,sy))
    z1 = sz/2*np.ones((len(x), len(y)))
    if cm_zero_center:
        vm = max(np.max(xy_slice), -np.min(xy_slice))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(xy_slice), vmax=np.max(xy_slice))

    colors_plane = my_cmap(norm(xy_slice.T))
    colors_plane[...,3] = 0.4
    surf = ax.plot_surface(x1.T, y1.T, z1.T, rstride=stride, cstride=stride, facecolors=colors_plane)
    ax.set_zlim((0,sz))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array(xy_slice)
    if colorbar:
        cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # transparent background
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

    x2 = sx/2*np.ones((len(y), len(z)))
    y2 = np.array([0*i + j for j in y for i in z]).reshape((sy,sz))
    z2 = np.array([i + 0*j for j in y for i in z]).reshape((sy,sz))
    if cm_zero_center:
        vm = max(np.max(yz_slice), -np.min(yz_slice))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(yz_slice), vmax=np.max(yz_slice))
    
    colors_plane = my_cmap(norm(yz_slice))
    colors_plane[...,3] = 0.4
    ax.plot_surface(x2, y2, z2, rstride=stride,cstride=stride, facecolors=colors_plane)
    ax.set_xlim((0,sx))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array(yz_slice)
    if colorbar:
        cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # transparent background
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

    plt.tight_layout()
    if title:
        plt.title(title)

    if fname:
        plt.savefig(fname, dpi=100, transparent=True)
        plt.close()
    else:
        return fig

def alpha_show_two_extremes(ratio, max_alpha=1.0):
    # ratio is a value between -1 and 1,
    # if close to 0, set low alpha,
    # if close to -1 or 1, set high alpha
    return np.abs(ratio)**3 * max_alpha

def plot_contours(data, fname, stride = 1, my_cmap = plt.cm.binary, cm_zero_center=True, title=None, num_contours=20, contour_alpha_fn=alpha_show_two_extremes):
    """Plot 3D contours of volumetric data.
    
    Args:
        data: 3D numpy array of shape (sx, sy, sz)
        fname: Output filename
        stride: Stride for sampling points
        my_cmap: Matplotlib colormap
        cm_zero_center: Whether to center colormap at zero
        title: Optional plot title
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get data dimensions
    sx, sy, sz = data.shape
    
    # Create coordinate grids
    x, y, z = np.mgrid[0:sx:stride, 0:sy:stride, 0:sz:stride]
    
    # Set normalization
    if cm_zero_center:
        vm = max(np.max(data), -np.min(data))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    
    # Plot multiple isosurfaces, omitting levels that's close to zero
    levels = []
    data_mean = np.mean(np.abs(data))
    data_min = np.percentile(data, 1e-2)
    data_max = np.percentile(data, 99.99)
    print(f"1e-2 percentile data_min: {data_min}, 99.99 percentile data_max: {data_max}, np.min(data): {np.min(data)}, np.max(data): {np.max(data)}")
    assert data_min < 0 and data_max > 0
    negative_levels = round(num_contours * np.abs(data_min) / (np.abs(data_max) + np.abs(data_min)))
    positive_levels = round(num_contours * np.abs(data_max) / (np.abs(data_max) + np.abs(data_min)))
    
    levels = np.concatenate(
                (np.linspace(data_min, 0.2*data_min, negative_levels),
                 np.linspace(0.2*data_max, data_max, positive_levels))
             )
    
    for level in levels:
        verts, faces, _, _ = measure.marching_cubes(data[::stride,::stride,::stride], level)
        
        # Scale vertices back to original coordinates
        verts = verts * stride
        
        # Create mesh and plot
        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor(my_cmap(norm(level)))

        normalized_level = np.sign(level) * (level/data_min if level < 0 else level/data_max)
        mesh.set_alpha(contour_alpha_fn(normalized_level))  # Set contour_transparency
        ax.add_collection3d(mesh)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y') 
    ax.set_zlabel('z')
    
    # Set axis limits
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.set_zlim(0, sz)
    
    # Add colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    
    if title:
        plt.title(title)
        
    plt.savefig(fname, dpi=100, transparent=True)
    plt.close()


def plot_full_farfield(data, fname, plot_batch_idx=0):
    """Plot 3D full farfield of volumetric data.
    
    Args:
        data: 3D numpy array of shape (sx, sy, sz)
        fname: Output filename
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    theta, phi, value, target_theta, target_phi = data
    theta = theta.detach().cpu().numpy()
    phi = phi.detach().cpu().numpy()
    value = value[plot_batch_idx].detach().cpu().numpy()

    max_dir_index = np.argmax(value.flatten())
    max_dir_theta = theta.flatten()[max_dir_index]
    max_dir_phi = phi.flatten()[max_dir_index]

    # Convert to cartesian coordinates
    r = np.abs(value)  # Use magnitude of data as radius
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Create the scatter plot
    surf = ax.scatter(x, y, z,
                     c=value,  # Color points by value
                     cmap=plt.cm.viridis,
                     s=10)  # Point size

    # Plot coordinate axes
    origin = np.array([0, 0, 0])
    axis_length = np.max(np.abs([x, y, z])) * 1.2  # Make axes slightly longer than data
    
    # X axis in red
    ax.quiver(origin[0], origin[1], origin[2], 
              axis_length, 0, 0, 
              color='red', alpha=0.5, lw=2)
    # Y axis in green  
    ax.quiver(origin[0], origin[1], origin[2],
              0, axis_length, 0,
              color='green', alpha=0.5, lw=2)
    # Z axis in blue
    ax.quiver(origin[0], origin[1], origin[2],
              0, 0, axis_length,
              color='blue', alpha=0.5, lw=2)

    # Plot vector in direction of (theta, phi)
    # Use middle values of theta/phi arrays
    theta_val = target_theta
    phi_val = target_phi
    
    # Convert spherical to cartesian coordinates
    dir_x = np.sin(theta_val) * np.cos(phi_val)
    dir_y = np.sin(theta_val) * np.sin(phi_val)
    dir_z = np.cos(theta_val)
    
    # Plot direction vector in yellow
    target_dir_length = np.max(r)
    ax.quiver(origin[0], origin[1], origin[2],
              dir_x * target_dir_length, dir_y * target_dir_length, dir_z * target_dir_length,
              color='yellow', alpha=0.8, lw=2)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-target_dir_length, target_dir_length)
    ax.set_ylim(-target_dir_length, target_dir_length)
    ax.set_zlim(-target_dir_length, target_dir_length)
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])

    ax.set_title(f"target theta, phi: {theta_val*180/np.pi:.1f}, {phi_val*180/np.pi:.1f}\nmax output theta, phi: {max_dir_theta*180/np.pi:.1f}, {max_dir_phi*180/np.pi:.1f}")
    
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()

def plot_poynting_radial_scatter(u0, E, H, fname=None, plot_batch_idx=0, normalize=True, point_size=8):
    """
    Scatter plot on the sphere for the radial component of the Poynting vector.
    Radius ∝ |Re(S_r)| where S = 0.5 * Re(E × H*), r̂ = u0.

    Args:
        u0 : torch.Tensor, shape (3, Nt, Np) or (1,3,Nt,Np). Unit vectors on the sphere.
        E  : torch.Tensor, shape (B,3,Nt,Np) or (3,Nt,Np). Complex far-field E.
        H  : torch.Tensor, shape (B,3,Nt,Np) or (3,Nt,Np). Complex far-field H.
        fname : str. Output image path.
        plot_batch_idx : int. Which batch to plot if E/H have batch dim.
        normalize : bool. If True, radius is normalized to [0,1].
        point_size : int. Scatter point size.
    """
    # ---- unify shapes ----
    if u0.dim() == 3: # (bs,3,N_points_on_sphere)
        u0 = u0[0]
    if E.dim() == 3: # (bs,3,N_points_on_sphere)
        E = E[0]
    if H.dim() == 3: # (bs,3,N_points_on_sphere)
        H = H[0]   

    # S = 0.5 * Re(E × H*)
    S = 0.5 * torch.real(torch.linalg.cross(E, torch.conj(H), dim=0))  # (bs,3,N_points_on_sphere), real

    # radial component S_r = S · r̂
    Sr = torch.sum(S * u0, dim=0)      # (3,N_points_on_sphere), real
    val = Sr # Re(S_r)
    assert (val>0).all(), "val should be positive"

    r = val

    # Cartesian coords: r * r̂
    x = r * u0[0]
    y = r * u0[1]
    z = r * u0[2]

    # to numpy
    x, y, z, c = (t.detach().cpu().numpy() for t in (x, y, z, val))

    # plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=c, s=point_size, cmap='viridis')

    R = 1.05 * np.max(np.abs([x, y, z]))
    R = 1.0 if not np.isfinite(R) or R == 0 else R
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=18, pad=0.02)
    cb.set_label(r'$|\,\mathrm{Re}(S_r)\,|$')
    ax.set_title(r'Radial Poynting Component: $|\,\mathrm{Re}(S_r)\,|$ (radius ∝ value)')

    if fname:
        plt.tight_layout()
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        return fig

def plot_Sr_subplot(u0, Sr, ax, point_size=8):
    """
    Scatter plot on the sphere for the radial component of the Poynting vector.
    Radius ∝ |Re(S_r)| where S = 0.5 * Re(E × H*), r̂ = u0.

    Args:
        u0 : torch.Tensor, shape (3, Nt, Np) or (1,3,Nt,Np). Unit vectors on the sphere.
        Sr : torch.Tensor, shape (Nt,Np). Radial component of the Poynting vector.
        ax : matplotlib.axes.Axes. Subplot to plot on.
        normalize : bool. If True, radius is normalized to [0,1].
        point_size : int. Scatter point size.
    """
    # ---- unify shapes ----
    assert len(u0.shape) == 2
    assert len(Sr.shape) == 1

    val = Sr 
    # radius
    r = val

    x = r * u0[0]
    y = r * u0[1]
    z = r * u0[2]
    # to numpy
    x, y, z, c = (t for t in (x, y, z, val))

    # plot
    sc = ax.scatter(x, y, z, c=c, s=point_size, cmap='viridis')

    R = 1.05 * np.max(np.abs([x, y, z]))
    R = 1.0 if not np.isfinite(R) or R == 0 else R
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    # plt.colorbar(shrink=0.5, aspect=5)
    # cb.set_label(r'$|\,\mathrm{Re}(S_r)\,|$')
    # ax.set_title(r'Radial Poynting Component: $|\,\mathrm{Re}(S_r)\,|$ (radius ∝ value)')
    
    return sc

# def plot_poynting_radial_scatter_old(u0, E, H, fname=None, plot_batch_idx=0, normalize=True, point_size=8):
#     """
#     Scatter plot on the sphere for the radial component of the Poynting vector.
#     Radius ∝ |Re(S_r)| where S = 0.5 * Re(E × H*), r̂ = u0.

#     Args:
#         u0 : torch.Tensor, shape (3, Nt, Np) or (1,3,Nt,Np). Unit vectors on the sphere.
#         E  : torch.Tensor, shape (B,3,Nt,Np) or (3,Nt,Np). Complex far-field E.
#         H  : torch.Tensor, shape (B,3,Nt,Np) or (3,Nt,Np). Complex far-field H.
#         fname : str. Output image path.
#         plot_batch_idx : int. Which batch to plot if E/H have batch dim.
#         normalize : bool. If True, radius is normalized to [0,1].
#         point_size : int. Scatter point size.
#     """
#     # ---- unify shapes ----
#     if u0.dim() == 4:       # (1,3,Nt,Np)
#         u0 = u0[0]
#     if E.dim() == 3:        # (3,Nt,Np) -> add batch
#         E = E.unsqueeze(0)
#     if H.dim() == 3:
#         H = H.unsqueeze(0)

#     # pick batch and move to CPU numpy
#     Eb = E[plot_batch_idx]               # (3,Nt,Np), complex
#     Hb = H[plot_batch_idx]               # (3,Nt,Np), complex
#     print(f"Eb.shape: {Eb.shape}, Hb.shape: {Hb.shape}")

#     # (Nt,Np,3)
#     E3 = Eb.permute(1, 2, 0)
#     H3 = Hb.permute(1, 2, 0)
#     U3 = u0.permute(1, 2, 0)             # r̂

#     # S = 0.5 * Re(E × H*)
#     S3 = 0.5 * torch.real(torch.linalg.cross(E3, torch.conj(H3), dim=-1))  # (Nt,Np,3), real

#     # radial component S_r = S · r̂
#     Sr = torch.sum(S3 * U3, dim=-1)      # (Nt,Np), real
#     # val = torch.abs(Sr)                  # |Re(S_r)|
#     val = Sr
#     assert (val>0).all(), "val should be positive"

#     # radius
#     if normalize:
#         vmax = torch.clamp(val.max(), min=1e-12)
#         r = val / vmax
#     else:
#         r = val

#     # Cartesian coords: r * r̂
#     x = r * U3[..., 0]
#     y = r * U3[..., 1]
#     z = r * U3[..., 2]

#     # to numpy
#     x, y, z, c = (t.detach().cpu().numpy().reshape(-1) for t in (x, y, z, val))

#     # plot
#     fig = plt.figure(figsize=(9, 9))
#     ax = fig.add_subplot(111, projection='3d')
#     sc = ax.scatter(x, y, z, c=c, s=point_size, cmap='viridis')

#     R = 1.05 * (np.max(r.detach().cpu().numpy()) if normalize else np.max(np.abs([x, y, z])))
#     R = 1.0 if not np.isfinite(R) or R == 0 else R
#     ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
#     ax.set_box_aspect([1, 1, 1])
#     ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
#     cb = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=18, pad=0.02)
#     cb.set_label(r'$|\,\mathrm{Re}(S_r)\,|$')
#     ax.set_title(r'Radial Poynting Component: $|\,\mathrm{Re}(S_r)\,|$ (radius ∝ value)')

#     if fname:
#         plt.tight_layout()
#         plt.savefig(fname, dpi=120, bbox_inches='tight')
#         plt.close()
#     else:
#         return fig

# def plot_Sr_subplot_old(u0, Sr, ax, point_size=8):
#     """
#     Scatter plot on the sphere for the radial component of the Poynting vector.
#     Radius ∝ |Re(S_r)| where S = 0.5 * Re(E × H*), r̂ = u0.

#     Args:
#         u0 : torch.Tensor, shape (3, Nt, Np) or (1,3,Nt,Np). Unit vectors on the sphere.
#         Sr : torch.Tensor, shape (Nt,Np). Radial component of the Poynting vector.
#         ax : matplotlib.axes.Axes. Subplot to plot on.
#         normalize : bool. If True, radius is normalized to [0,1].
#         point_size : int. Scatter point size.
#     """
#     # ---- unify shapes ----
#     assert len(u0.shape) == 3
#     assert len(Sr.shape) == 2
#     U3 = u0.transpose(1, 2, 0) # (Nt,Np,3) 

#     val = np.abs(Sr) 

#     # radius
#     r = val

#     x = r * U3[..., 0]
#     y = r * U3[..., 1]
#     z = r * U3[..., 2]
#     # to numpy
#     x, y, z, c = (t.reshape(-1) for t in (x, y, z, val))

#     # plot
#     sc = ax.scatter(x, y, z, c=c, s=point_size, cmap='viridis')

#     R = 1.05 * np.max(np.abs([x, y, z]))
#     R = 1.0 if not np.isfinite(R) or R == 0 else R
#     ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
#     ax.set_box_aspect([1, 1, 1])
#     ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
#     # plt.colorbar(shrink=0.5, aspect=5)
#     # cb.set_label(r'$|\,\mathrm{Re}(S_r)\,|$')
#     # ax.set_title(r'Radial Poynting Component: $|\,\mathrm{Re}(S_r)\,|$ (radius ∝ value)')
    
#     return 

# plot helper for 2d test cases:
def plot_2d(data, fname=None, stride = 1, my_cmap = plt.cm.binary, cm_zero_center=True, title=None):
    """Plot 2D slices of volumetric data.
    
    Args:
        data: 2D numpy array of shape (sx, sy)
        fname: Output filename
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    # Get data dimensions
    sx, sy = data.shape
    if cm_zero_center:
        vm = max(np.max(data), -np.min(data))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))
        
    # Plot the data
    ax.imshow(data, cmap=my_cmap, norm=norm)
    
    # Add colorbar
    fig.colorbar(ax.imshow(data, cmap=my_cmap, norm=norm), ax=ax, shrink=0.5, aspect=5)
    
    if title:
        plt.title(title)
        
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
