import numpy as np
from matplotlib import pyplot as plt
import torch

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

def plot_3slices(data, fname=None, stride = 1, my_cmap = plt.cm.binary, cm_zero_center=True, title=None):
    # using3D()
    sx, sy, sz = data.shape
    xy_slice = data[:, :, int(sz/2)]
    yz_slice = data[int(sx/2), :, :]
    zx_slice = data[:, int(sy/2), :].T

    x = list(range(sx))
    y = list(range(sy))
    z = list(range(sz))

    fig = plt.figure(figsize=(14,4))
    ax = plt.subplot(131, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x1 = np.array([0*i + j for j in x for i in y]).reshape((sx,sy))
    y1 = np.array([i + 0*j for j in x for i in y]).reshape((sx,sy))
    z1 = sz/2*np.ones((len(x), len(y)))
    if cm_zero_center:
        vm = max(np.max(xy_slice), -np.min(xy_slice))
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
    cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)


    ax = plt.subplot(132, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x2 = sx/2*np.ones((len(y), len(z)))
    y2 = np.array([0*i + j for j in y for i in z]).reshape((sy,sz))
    z2 = np.array([i + 0*j for j in y for i in z]).reshape((sy,sz))
    if cm_zero_center:
        vm = max(np.max(yz_slice), -np.min(yz_slice))
        norm = Normalize(vmin=-vm, vmax=vm)
    else:
        norm = Normalize(vmin=np.min(yz_slice), vmax=np.max(yz_slice))
    ax.plot_surface(x2, y2, z2, rstride=stride,cstride=stride, facecolors=my_cmap(norm(yz_slice)))
    ax.set_xlim((0,sx))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array(yz_slice)
    cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    
    ax = plt.subplot(133, projection="3d")
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
    ax.plot_surface(x3, y3, z3, rstride=stride,cstride=stride, facecolors=my_cmap(norm(zx_slice)))
    ax.set_ylim((0,sy))
    ax.set_aspect('equal')
    mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    mappable.set_array(zx_slice)
    cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

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

def plot_poynting_radial_scatter(u0, E, H, fname, plot_batch_idx=0, normalize=True, point_size=8):
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
    if u0.dim() == 4:       # (1,3,Nt,Np)
        u0 = u0[0]
    if E.dim() == 3:        # (3,Nt,Np) -> add batch
        E = E.unsqueeze(0)
    if H.dim() == 3:
        H = H.unsqueeze(0)

    # pick batch and move to CPU numpy
    Eb = E[plot_batch_idx]               # (3,Nt,Np), complex
    Hb = H[plot_batch_idx]               # (3,Nt,Np), complex

    # (Nt,Np,3)
    E3 = Eb.permute(1, 2, 0)
    H3 = Hb.permute(1, 2, 0)
    U3 = u0.permute(1, 2, 0)             # r̂

    # S = 0.5 * Re(E × H*)
    S3 = 0.5 * torch.real(torch.linalg.cross(E3, torch.conj(H3), dim=-1))  # (Nt,Np,3), real

    # radial component S_r = S · r̂
    Sr = torch.sum(S3 * U3, dim=-1)      # (Nt,Np), real
    val = torch.abs(Sr)                  # |Re(S_r)|

    # radius
    if normalize:
        vmax = torch.clamp(val.max(), min=1e-12)
        r = val / vmax
    else:
        r = val

    # Cartesian coords: r * r̂
    x = r * U3[..., 0]
    y = r * U3[..., 1]
    z = r * U3[..., 2]

    # to numpy
    x, y, z, c = (t.detach().cpu().numpy().reshape(-1) for t in (x, y, z, val))

    # plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=c, s=point_size, cmap='viridis')

    R = 1.05 * (np.max(r.detach().cpu().numpy()) if normalize else np.max(np.abs([x, y, z])))
    R = 1.0 if not np.isfinite(R) or R == 0 else R
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=18, pad=0.02)
    cb.set_label(r'$|\,\mathrm{Re}(S_r)\,|$')
    ax.set_title(r'Radial Poynting Component: $|\,\mathrm{Re}(S_r)\,|$ (radius ∝ value)')
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()