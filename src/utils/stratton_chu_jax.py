import jax.numpy as jnp


C_0 = 299792458.13099605
EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6

def E_to_H(Ex, Ey, Ez, dxes, omega, bloch_vector=None):
    # dxes: List[List[np.ndarray]], dxes[0] is the grid spacing for E and dxes[1] for H, dxes[0][0] is the grid spacing for x
    Hx = E_to_Hx(Ey, Ez, dxes, omega, bloch_vector=bloch_vector)
    Hy = E_to_Hy(Ez, Ex, dxes, omega, bloch_vector=bloch_vector)
    Hz = E_to_Hz(Ex, Ey, dxes, omega, bloch_vector=bloch_vector)
    return (Hx, Hy, Hz)
    

def E_to_Hx(Ey, Ez, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dEzdy = jnp.roll(Ez, -1, axis=1) - Ez # np.roll([1,2,3],-1) = [2,3,1]
        dEydz = jnp.roll(Ey, -1, axis=2) - Ey
    else:
        dEzdy = jnp.concatenate((Ez[:,1:,:], Ez[:,0:1,:]*jnp.exp(-1j*bloch_vector[1]*dL*1e9*Ez.shape[1])), axis=1) - Ez
        dEydz = jnp.concatenate((Ey[:,:,1:], Ey[:,:,0:1]*jnp.exp(-1j*bloch_vector[2]*dL*1e9*Ey.shape[2])), axis=2) - Ey

    Hx = (dEzdy / dxes[0][1][None,:,None] - dEydz / dxes[0][2][None,None,:]) / (-1j*omega)
    return Hx

def E_to_Hy(Ez, Ex, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dExdz = jnp.roll(Ex, -1, axis=2) - Ex
        dEzdx = jnp.roll(Ez, -1, axis=0) - Ez
    else:
        dExdz = jnp.concatenate((Ex[:,:,1:], Ex[:,:,0:1]*jnp.exp(-1j*bloch_vector[2]*dL*1e9*Ex.shape[2])), axis=2) - Ex
        dEzdx = jnp.concatenate((Ez[1:,:,:], Ez[0:1,:,:]*jnp.exp(-1j*bloch_vector[0]*dL*1e9*Ez.shape[0])), axis=0) - Ez

    Hy = (dExdz / dxes[0][2][None,None,:] - dEzdx / dxes[0][0][:,None,None]) / (-1j*omega)
    return Hy

def E_to_Hz(Ex, Ey, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dEydx = jnp.roll(Ey, -1, axis=0) - Ey
        dExdy = jnp.roll(Ex, -1, axis=1) - Ex
    else:
        dEydx = jnp.concatenate((Ey[1:,:,:], Ey[0:1,:,:]*jnp.exp(-1j*bloch_vector[0]*dL*1e9*Ey.shape[0])), axis=0) - Ey
        dExdy = jnp.concatenate((Ex[:,1:,:], Ex[:,0:1,:]*jnp.exp(-1j*bloch_vector[1]*dL*1e9*Ex.shape[1])), axis=1) - Ex

    Hz = (dEydx / dxes[0][0][:,None,None] - dExdy / dxes[0][1][None,:,None]) / (-1j*omega)
    return Hz


def strattonChu3D_GPU(dl, xc, yc, zc, Rx, Ry, Rz, lambda_val, Ex_OBJ, Ey_OBJ, Ez_OBJ, Hx_OBJ, Hy_OBJ, Hz_OBJ,
                      device='cuda', t_theta=0, t_phi=0, bs=1):
    k0 = 2 * jnp.pi / lambda_val
    ds = dl * dl
    r_obs = 10000 * lambda_val

    x_back = int(xc - Rx)
    x_front = int(xc + Rx - 1)
    y_left = int(yc - Ry)
    y_right = int(yc + Ry - 1)
    z_bottom = int(zc - Rz)
    z_top = int(zc + Rz)

    ##################### Prepare the normal vector of the data points on 6 faces #####################
    n_null = jnp.zeros((1, 3), dtype=jnp.complex64)
    n_back = jnp.tile(jnp.array([[-1, 0, 0]]), ((y_right - y_left)*(z_top - z_bottom), 1))
    n_front = jnp.tile(jnp.array([[1, 0, 0]]), ((y_right - y_left)*(z_top - z_bottom), 1))
    n_left = jnp.tile(jnp.array([[0, -1, 0]]), ((x_front - x_back)*(z_top - z_bottom), 1))
    n_right = jnp.tile(jnp.array([[0, 1, 0]]), ((x_front - x_back)*(z_top - z_bottom), 1))
    n_bottom = jnp.tile(jnp.array([[0, 0, -1]]), ((x_front - x_back)*(y_right - y_left), 1))
    n_top = jnp.tile(jnp.array([[0, 0, 1]]), ((x_front - x_back)*(y_right - y_left), 1))

    n = jnp.concatenate([n_null, n_back, n_front, n_left, n_right, n_bottom, n_top], axis=0)
    n = jnp.broadcast_to(n[jnp.newaxis, ...], (bs, *n.shape))  # (bs, N_total, 3)

    ##################### Prepare the coordinates of the data points on 6 faces #####################
    y = jnp.arange(y_left, y_right)
    z = jnp.arange(z_bottom, z_top)
    x = jnp.arange(x_back, x_front)

    Yb, Zb = jnp.meshgrid(y, z, indexing="ij")
    Xb = jnp.full_like(Yb, x_back)
    xyz_back = jnp.stack([(Xb - xc)*dl, (Yb - yc)*dl, (Zb - zc)*dl], axis=-1).reshape(-1, 3)

    Yf, Zf = jnp.meshgrid(y, z, indexing="ij")
    Xf = jnp.full_like(Yf, x_front)
    xyz_front = jnp.stack([(Xf - xc)*dl, (Yf - yc)*dl, (Zf - zc)*dl], axis=-1).reshape(-1, 3)

    Xl, Zl = jnp.meshgrid(x, z, indexing="ij")
    Yl = jnp.full_like(Xl, y_left)
    xyz_left = jnp.stack([(Xl - xc)*dl, (Yl - yc)*dl, (Zl - zc)*dl], axis=-1).reshape(-1, 3)

    Xr, Zr = jnp.meshgrid(x, z, indexing="ij")
    Yr = jnp.full_like(Xr, y_right)
    xyz_right = jnp.stack([(Xr - xc)*dl, (Yr - yc)*dl, (Zr - zc)*dl], axis=-1).reshape(-1, 3)

    Xbo, Ybo = jnp.meshgrid(x, y, indexing="ij")
    Zbo = jnp.full_like(Xbo, z_bottom)
    xyz_bottom = jnp.stack([(Xbo - xc)*dl, (Ybo - yc)*dl, (Zbo - zc)*dl], axis=-1).reshape(-1, 3)

    Xt, Yt = jnp.meshgrid(x, y, indexing="ij")
    Zt = jnp.full_like(Xt, z_top)
    xyz_top = jnp.stack([(Xt - xc)*dl, (Yt - yc)*dl, (Zt - zc)*dl], axis=-1).reshape(-1, 3)

    xyz_null = jnp.zeros((1, 3), dtype=jnp.float32)

    xyz = jnp.concatenate([xyz_null, xyz_back, xyz_front, xyz_left, xyz_right, xyz_bottom, xyz_top], axis=0)  # (N_total, 3)

    ##################### Prepare the fields on the six faces #####################

    E_null = jnp.zeros((bs, 1, 3), dtype=jnp.complex64)
    H_null = jnp.zeros((bs, 1, 3), dtype=jnp.complex64)

    E_back = jnp.stack([Ex_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Ey_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Ez_OBJ[:, x_back, y_left:y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)
    H_back = jnp.stack([Hx_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Hy_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Hz_OBJ[:, x_back, y_left:y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)

    E_front = jnp.stack([Ex_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Ey_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Ez_OBJ[:, x_front, y_left:y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)
    H_front = jnp.stack([Hx_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Hy_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Hz_OBJ[:, x_front, y_left:y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)

    E_left = jnp.stack([Ex_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Ey_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Ez_OBJ[:, x_back:x_front, y_left, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)
    H_left = jnp.stack([Hx_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Hy_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Hz_OBJ[:, x_back:x_front, y_left, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)

    E_right = jnp.stack([Ex_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Ey_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Ez_OBJ[:, x_back:x_front, y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)
    H_right = jnp.stack([Hx_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Hy_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Hz_OBJ[:, x_back:x_front, y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)

    E_bottom = jnp.stack([Ex_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Ey_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Ez_OBJ[:, x_back:x_front, y_left:y_right, z_bottom]], axis=3).reshape(bs, -1, 3)
    H_bottom = jnp.stack([Hx_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Hy_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Hz_OBJ[:, x_back:x_front, y_left:y_right, z_bottom]], axis=3).reshape(bs, -1, 3)

    E_top = jnp.stack([Ex_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Ey_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Ez_OBJ[:, x_back:x_front, y_left:y_right, z_top]], axis=3).reshape(bs, -1, 3)
    H_top = jnp.stack([Hx_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Hy_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Hz_OBJ[:, x_back:x_front, y_left:y_right, z_top]], axis=3).reshape(bs, -1, 3)

    E = jnp.concatenate([E_null, E_back, E_front, E_left, E_right, E_bottom, E_top], axis=1)
    H = jnp.concatenate([H_null, H_back, H_front, H_left, H_right, H_bottom, H_top], axis=1)

    t_theta = jnp.array(t_theta) * jnp.pi/ 180
    t_phi = jnp.array(t_phi) * jnp.pi/ 180
    cos_theta = jnp.cos(t_theta)
    sin_theta = jnp.sin(t_theta)
    cos_phi = jnp.cos(t_phi)
    sin_phi = jnp.sin(t_phi)
    ux_0 = (cos_phi * sin_theta)
    uy_0 = (sin_phi * sin_theta)
    uz_0 = (cos_theta)
    u0 = jnp.stack([ux_0, uy_0, uz_0])

    r_rs = jnp.abs(r_obs * u0 - xyz)
    r_rs = jnp.sqrt(r_rs[:, 0] ** 2 + r_rs[:, 1] ** 2 + r_rs[:, 2] ** 2)  # [15841]
    ux = (r_obs * ux_0 - xyz[:, 0]) / r_rs
    uy = (r_obs * uy_0 - xyz[:, 1]) / r_rs
    uz = (r_obs * uz_0 - xyz[:, 2]) / r_rs
    t_u = jnp.stack([ux, uy, uz], axis=1)
    t_u = jnp.array(t_u, dtype=jnp.complex64)
    t_coe = 1j * k0 * ds * jnp.exp(-1j * k0 * r_rs) / (4 * jnp.pi * r_rs)
    t_coe = jnp.stack([t_coe, t_coe, t_coe], axis=1)
    t_n = n

    t_u = jnp.expand_dims(t_u, axis=0).repeat(bs, axis=0)
    t_coe = jnp.expand_dims(t_coe, axis=0).repeat(bs, axis=0)

    far_E = t_coe * (
            -C_0 * MU_0 * jnp.cross(t_n, H, axis=-1)
            + jnp.cross(jnp.cross(t_n, E, axis=-1), t_u, axis=-1)
            + jnp.sum(t_n * E, axis=-1, keepdims=True) * t_u
    )
    far_H = t_coe * (
            C_0 * EPSILON_0 * jnp.cross(t_n, E, axis=-1)
            + jnp.cross(jnp.cross(t_n, H, axis=-1), t_u, axis=-1)
            + jnp.sum(t_n * H, axis=-1, keepdims=True) * t_u
    )

    tg_E_sum = jnp.sum(far_E, axis=1)
    tg_H_sum = jnp.sum(far_H, axis=1)
    u0 = jnp.broadcast_to(u0[jnp.newaxis, ...], (bs, *u0.shape))

    return u0, tg_E_sum, tg_H_sum


def strattonChu3D_full_sphere_GPU(dl, xc, yc, zc, Rx, Ry, Rz, lambda_val, Ex_OBJ, Ey_OBJ, Ez_OBJ, Hx_OBJ, Hy_OBJ, Hz_OBJ,
                                  device='cuda', bs=1):
    k0 = 2 * jnp.pi / lambda_val
    ds = dl * dl
    r_obs = 10000 * lambda_val
    
    x_back = int(xc - Rx)
    x_front = int(xc + Rx - 1)
    y_left = int(yc - Ry)
    y_right = int(yc + Ry - 1)
    z_bottom = int(zc - Rz)
    z_top = int(zc + Rz)

    ##################### Prepare the normal vector of the data points on 6 faces #####################
    n_null = jnp.zeros((1, 3), dtype=jnp.complex64)
    n_back = jnp.tile(jnp.array([[-1, 0, 0]]), ((y_right - y_left)*(z_top - z_bottom), 1))
    n_front = jnp.tile(jnp.array([[1, 0, 0]]), ((y_right - y_left)*(z_top - z_bottom), 1))
    n_left = jnp.tile(jnp.array([[0, -1, 0]]), ((x_front - x_back)*(z_top - z_bottom), 1))
    n_right = jnp.tile(jnp.array([[0, 1, 0]]), ((x_front - x_back)*(z_top - z_bottom), 1))
    n_bottom = jnp.tile(jnp.array([[0, 0, -1]]), ((x_front - x_back)*(y_right - y_left), 1))
    n_top = jnp.tile(jnp.array([[0, 0, 1]]), ((x_front - x_back)*(y_right - y_left), 1))

    n = jnp.concatenate([n_null, n_back, n_front, n_left, n_right, n_bottom, n_top], axis=0)
    n = jnp.broadcast_to(n[jnp.newaxis, ...], (bs, *n.shape))  # (bs, N_total, 3)

    ##################### Prepare the coordinates of the data points on 6 faces #####################
    y = jnp.arange(y_left, y_right)
    z = jnp.arange(z_bottom, z_top)
    x = jnp.arange(x_back, x_front)

    Yb, Zb = jnp.meshgrid(y, z, indexing="ij")
    Xb = jnp.full_like(Yb, x_back)
    xyz_back = jnp.stack([(Xb - xc)*dl, (Yb - yc)*dl, (Zb - zc)*dl], axis=-1).reshape(-1, 3)

    Yf, Zf = jnp.meshgrid(y, z, indexing="ij")
    Xf = jnp.full_like(Yf, x_front)
    xyz_front = jnp.stack([(Xf - xc)*dl, (Yf - yc)*dl, (Zf - zc)*dl], axis=-1).reshape(-1, 3)

    Xl, Zl = jnp.meshgrid(x, z, indexing="ij")
    Yl = jnp.full_like(Xl, y_left)
    xyz_left = jnp.stack([(Xl - xc)*dl, (Yl - yc)*dl, (Zl - zc)*dl], axis=-1).reshape(-1, 3)

    Xr, Zr = jnp.meshgrid(x, z, indexing="ij")
    Yr = jnp.full_like(Xr, y_right)
    xyz_right = jnp.stack([(Xr - xc)*dl, (Yr - yc)*dl, (Zr - zc)*dl], axis=-1).reshape(-1, 3)

    Xbo, Ybo = jnp.meshgrid(x, y, indexing="ij")
    Zbo = jnp.full_like(Xbo, z_bottom)
    xyz_bottom = jnp.stack([(Xbo - xc)*dl, (Ybo - yc)*dl, (Zbo - zc)*dl], axis=-1).reshape(-1, 3)

    Xt, Yt = jnp.meshgrid(x, y, indexing="ij")
    Zt = jnp.full_like(Xt, z_top)
    xyz_top = jnp.stack([(Xt - xc)*dl, (Yt - yc)*dl, (Zt - zc)*dl], axis=-1).reshape(-1, 3)

    xyz_null = jnp.zeros((1, 3), dtype=jnp.float32)

    xyz = jnp.concatenate([xyz_null, xyz_back, xyz_front, xyz_left, xyz_right, xyz_bottom, xyz_top], axis=0)  # (N_total, 3)
    xyz = jnp.broadcast_to(xyz[jnp.newaxis, ...], (bs, *xyz.shape))  # (bs, N_total, 3)

    ##################### Prepare the fields on the six faces #####################
    E_null = jnp.zeros((bs, 1, 3), dtype=jnp.complex64)
    H_null = jnp.zeros((bs, 1, 3), dtype=jnp.complex64)

    E_back = jnp.stack([Ex_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Ey_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Ez_OBJ[:, x_back, y_left:y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)
    H_back = jnp.stack([Hx_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Hy_OBJ[:, x_back, y_left:y_right, z_bottom:z_top],
                          Hz_OBJ[:, x_back, y_left:y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)

    E_front = jnp.stack([Ex_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Ey_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Ez_OBJ[:, x_front, y_left:y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)
    H_front = jnp.stack([Hx_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Hy_OBJ[:, x_front, y_left:y_right, z_bottom:z_top],
                           Hz_OBJ[:, x_front, y_left:y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)

    E_left = jnp.stack([Ex_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Ey_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Ez_OBJ[:, x_back:x_front, y_left, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)
    H_left = jnp.stack([Hx_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Hy_OBJ[:, x_back:x_front, y_left, z_bottom:z_top],
                          Hz_OBJ[:, x_back:x_front, y_left, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)

    E_right = jnp.stack([Ex_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Ey_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Ez_OBJ[:, x_back:x_front, y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)
    H_right = jnp.stack([Hx_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Hy_OBJ[:, x_back:x_front, y_right, z_bottom:z_top],
                           Hz_OBJ[:, x_back:x_front, y_right, z_bottom:z_top]], axis=3).reshape(bs, -1, 3)

    E_bottom = jnp.stack([Ex_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Ey_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Ez_OBJ[:, x_back:x_front, y_left:y_right, z_bottom]], axis=3).reshape(bs, -1, 3)
    H_bottom = jnp.stack([Hx_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Hy_OBJ[:, x_back:x_front, y_left:y_right, z_bottom],
                            Hz_OBJ[:, x_back:x_front, y_left:y_right, z_bottom]], axis=3).reshape(bs, -1, 3)

    E_top = jnp.stack([Ex_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Ey_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Ez_OBJ[:, x_back:x_front, y_left:y_right, z_top]], axis=3).reshape(bs, -1, 3)
    H_top = jnp.stack([Hx_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Hy_OBJ[:, x_back:x_front, y_left:y_right, z_top],
                         Hz_OBJ[:, x_back:x_front, y_left:y_right, z_top]], axis=3).reshape(bs, -1, 3)

    E = jnp.concatenate([E_null, E_back, E_front, E_left, E_right, E_bottom, E_top], axis=1) # (bs, N_total, 3)
    H = jnp.concatenate([H_null, H_back, H_front, H_left, H_right, H_bottom, H_top], axis=1) # (bs, N_total, 3)

    theta_step = 1
    theta_min, theta_max = theta_step, 90 - theta_step
    theta_range = jnp.linspace(theta_min, theta_max, (theta_max - theta_min) // theta_step + 1)
    phi_step = 4
    phi_min, phi_max = 0, 360 - phi_step
    phi_range = jnp.linspace(phi_min, phi_max, (phi_max - phi_min) // phi_step + 1)
    N_theta, N_phi = len(theta_range), len(phi_range)

    thetas, phis = jnp.meshgrid(theta_range, phi_range, indexing='ij')
    thetas = (thetas * jnp.pi/ 180)
    phis = (phis * jnp.pi/ 180)
    cos_theta = jnp.cos(thetas)
    sin_theta = jnp.sin(thetas)
    cos_phi = jnp.cos(phis)
    sin_phi = jnp.sin(phis)
    ux_0 = cos_phi * sin_theta
    uy_0 = sin_phi * sin_theta
    uz_0 = cos_theta
    u0 = jnp.stack([ux_0, uy_0, uz_0]) # (3, N_theta, N_phi)
    xyz = xyz[0]

    # xyz: (N_total, 3)
    r_rs = jnp.abs(r_obs * u0[None] - xyz[:,:,None,None]) # (N_total, N_theta, N_phi)
    r_rs = jnp.sqrt(r_rs[:, 0] ** 2 + r_rs[:, 1] ** 2 + r_rs[:, 2] ** 2)  # [N_total, N_theta, N_phi]
    ux = (r_obs * ux_0[None] - xyz[:, 0, None, None]) / r_rs
    uy = (r_obs * uy_0[None] - xyz[:, 1, None, None]) / r_rs
    uz = (r_obs * uz_0[None] - xyz[:, 2, None, None]) / r_rs
    t_u = jnp.stack([ux, uy, uz], axis=1).astype(jnp.complex64) # (N_total, 3, N_theta, N_phi)
    t_coe = 1j * k0 * ds * jnp.exp(-1j * k0 * r_rs) / (4 * jnp.pi * r_rs) # (N_total, N_theta, N_phi)
    
    t_n = n[...,None] # (bs, N_total, 3, 1)
    t_u = jnp.expand_dims(t_u, axis=0).repeat(bs, axis=0) # (bs, N_total, 3, N_theta, N_phi)

    t_coe = jnp.stack([t_coe, t_coe, t_coe], axis=1) # (N_total, 3, N_theta, N_phi)
    t_coe = jnp.expand_dims(t_coe, axis=0).repeat(bs, axis=0) # (bs, N_total, 3, N_theta, N_phi)

    #######################  processing theta iteratively to avoid OOM ####################### 
    tg_E_sum = jnp.zeros((bs, 3, N_theta, N_phi), dtype=jnp.complex64)
    tg_H_sum = jnp.zeros((bs, 3, N_theta, N_phi), dtype=jnp.complex64)
    for theta_idx in range(N_theta):
        far_E = t_coe[:,:,:,theta_idx,:] * (
                -C_0 * MU_0 * jnp.cross(t_n, H[...,None], axis=2)
                + jnp.cross(jnp.cross(t_n, E[...,None], axis=2), t_u[:,:,:,theta_idx,:], axis=2)
                + jnp.sum(t_n * E[...,None], axis=2, keepdims=True) * t_u[:,:,:,theta_idx,:]
        )
        far_H = t_coe[:,:,:,theta_idx,:] * (
                C_0 * EPSILON_0 * jnp.cross(t_n, E[...,None], axis=2)
                + jnp.cross(jnp.cross(t_n, H[...,None], axis=2), t_u[:,:,:,theta_idx,:], axis=2)
                + jnp.sum(t_n * H[...,None], axis=2, keepdims=True) * t_u[:,:,:,theta_idx,:]
        )
        far_E_sum = jnp.sum(far_E, axis=1)
        far_H_sum = jnp.sum(far_H, axis=1)
        tg_E_sum = tg_E_sum.at[:,:,theta_idx,:].set(far_E_sum)
        tg_H_sum = tg_H_sum.at[:,:,theta_idx,:].set(far_H_sum)
    #######################  processing all directions at once ####################### 
    # far_E = t_coe * (
    #         -C_0 * MU_0 * torch.cross(t_n, H[...,None,None], dim=2)
    #         + torch.cross(torch.cross(t_n, E[...,None,None], dim=2), t_u, dim=2)
    #         + torch.sum(t_n * E[...,None,None], dim=2, keepdim=True) * t_u
    # )
    # far_H = t_coe * (
    #         C_0 * EPSILON_0 * torch.cross(t_n, E[...,None,None], dim=2)
    #         + torch.cross(torch.cross(t_n, H[...,None,None], dim=2), t_u, dim=2)
    #         + torch.sum(t_n * H[...,None,None], dim=2, keepdim=True) * t_u
    # )
    # tg_E_sum = torch.sum(far_E, dim=1) # (bs, 3, N_theta, N_phi)
    # tg_H_sum = torch.sum(far_H, dim=1) # (bs, 3, N_theta, N_phi)
    ##################################################################################
    u0 = jnp.expand_dims(u0, axis=0).repeat(bs, axis=0) # (bs, 3, N_theta, N_phi)

    return thetas, phis, u0, tg_E_sum, tg_H_sum