import torch, os
import numpy as np

from src.utils.PML_utils import apply_scpml, conditioners

import gin

C_0 = 299792458.13099605
EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6

def c2r(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_real(x).reshape(bs, sx, sy, sz, 6)

def r2c(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_complex(x.reshape(bs, sx, sy, sz, 3, 2))

def src2rhs(src, dL, wavelength):
    # equation: 1/w * (del E + w**2 eps (1-i gamma) E) = i J 
    # multiply 1/w on both sides to normalize to the magnitude of J
    # src: (bs, sx, sy, sz, 6)
    bs, sx, sy, sz, _ = src.shape
    # omega = 2*np.pi/wavelength # narutal unit with C_0 as 1
    src = torch.view_as_complex(src.reshape(bs, sx, sy, sz, -1, 2))
    rhs = 1j*src
    rhs = torch.view_as_real(rhs).reshape(bs, sx, sy, sz, -1)
    return rhs

def src2rhs2d(src, dL, wavelength):
    # equation: 1/w * (del E + w**2 eps (1-i gamma) E) = i J 
    # multiply 1/w on both sides to normalize to the magnitude of J
    # src: (bs, sx, sy, 2)
    bs, sx, sy, _ = src.shape
    # omega = 2*np.pi/wavelength # narutal unit with C_0 as 1
    src = torch.view_as_complex(src.reshape(bs, sx, sy, -1, 2))
    rhs = 1j*src
    rhs = torch.view_as_real(rhs).reshape(bs, sx, sy, -1)
    return rhs

#######################################################################################################
############################## for testing stratton chu, remove later #################################
#######################################################################################################
def E_to_H(Ex, Ey, Ez, dxes, omega, bloch_vector=None):
    # dxes: List[List[np.ndarray]], dxes[0] is the grid spacing for E and dxes[1] for H, dxes[0][0] is the grid spacing for x
    Hx = E_to_Hx(Ey, Ez, dxes, omega, bloch_vector=bloch_vector)
    Hy = E_to_Hy(Ez, Ex, dxes, omega, bloch_vector=bloch_vector)
    Hz = E_to_Hz(Ex, Ey, dxes, omega, bloch_vector=bloch_vector)
    return (Hx, Hy, Hz)
    

def E_to_Hx(Ey, Ez, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dEzdy = torch.roll(Ez, shifts=-1, dims=2) - Ez  # Shift along y-axis
        dEydz = torch.roll(Ey, shifts=-1, dims=3) - Ey  # Shift along z-axis
    else:
        dEzdy = torch.cat((Ez[:, :, 1:, :], Ez[:, :, 0:1, :] * torch.exp(-1j * bloch_vector[1] * dL * 1e9 * Ez.shape[2])),
                          dim=2) - Ez
        dEydz = torch.cat((Ey[:, :, :, 1:], Ey[:, :, :, 0:1] * torch.exp(-1j * bloch_vector[2] * dL * 1e9 * Ey.shape[3])),
                          dim=3) - Ey

    Hx = (dEzdy - dEydz) / (-1j * dxes * omega * MU_0)
    return Hx

def E_to_Hy(Ez, Ex, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dExdz = torch.roll(Ex, shifts=-1, dims=3) - Ex  # Shift along z-axis
        dEzdx = torch.roll(Ez, shifts=-1, dims=1) - Ez  # Shift along x-axis
    else:
        dExdz = torch.cat((Ex[:, :, :, 1:], Ex[:, :, :, 0:1] * torch.exp(-1j * bloch_vector[2] * dL * 1e9 * Ex.shape[3])),
                          dim=3) - Ex
        dEzdx = torch.cat((Ez[:, 1:, :, :], Ez[:, 0:1, :, :] * torch.exp(-1j * bloch_vector[0] * dL * 1e9 * Ez.shape[1])),
                          dim=1) - Ez

    Hy = (dExdz - dEzdx) / (-1j * dxes * omega * MU_0)
    return Hy

def E_to_Hz(Ex, Ey, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dEydx = torch.roll(Ey, shifts=-1, dims=1) - Ey  # Shift along x-axis
        dExdy = torch.roll(Ex, shifts=-1, dims=2) - Ex  # Shift along y-axis
    else:
        dEydx = torch.cat((Ey[:, 1:, :, :], Ey[:, 0:1, :, :] * torch.exp(-1j * bloch_vector[0] * dL * 1e9 * Ey.shape[1])),
                          dim=1) - Ey
        dExdy = torch.cat((Ex[:, :, 1:, :], Ex[:, :, 0:1, :] * torch.exp(-1j * bloch_vector[1] * dL * 1e9 * Ex.shape[2])),
                          dim=2) - Ex

    Hz = (dEydx - dExdy) / (-1j * dxes * omega * MU_0)
    return Hz

def H_to_E(Hx, Hy, Hz, dxes, omega, eps_grid, bloch_vector=None):
    Ex = H_to_Ex(Hy, Hz, dxes, omega, eps_grid, bloch_vector=bloch_vector)
    Ey = H_to_Ey(Hz, Hx, dxes, omega, eps_grid, bloch_vector=bloch_vector)
    Ez = H_to_Ez(Hx, Hy, dxes, omega, eps_grid, bloch_vector=bloch_vector)
    return (Ex, Ey, Ez)

def H_to_Ex(Hy, Hz, dxes, omega, eps_grid, bloch_vector=None):
    if bloch_vector is None:
        dHzdy = -torch.roll(Hz, shifts=1, dims=2) + Hz  # np.roll([1,2,3],1) = [3,1,2]
        dHydz = -torch.roll(Hy, shifts=1, dims=3) + Hy
    else:
        dHzdy = -torch.cat((Hz[:, -1:, :] * torch.exp(1j * bloch_vector[1] * dL * 1e9 * Hz.shape[1]), Hz[:, :-1, :]),
                                axis=1) + Hz
        dHydz = -torch.cat((Hy[:, :, -1:] * torch.exp(1j * bloch_vector[2] * dL * 1e9 * Hy.shape[2]), Hy[:, :, :-1]),
                                axis=2) + Hy

    Ex = (dHzdy - dHydz ) / (1j * dxes * omega * EPSILON_0 * eps_grid)
    return Ex

def H_to_Ey(Hz, Hx, dxes, omega, eps_grid, bloch_vector=None):
    if bloch_vector is None:
        dHxdz = -torch.roll(Hx, shifts=1, dims=3) + Hx
        dHzdx = -torch.roll(Hz, shifts=1, dims=1) + Hz
    else:
        dHxdz = -torch.cat((Hx[:, :, -1:] * torch.exp(1j * bloch_vector[2] * dL * 1e9 * Hx.shape[2]), Hx[:, :, :-1]),
                                axis=2) + Hx
        dHzdx = -torch.cat((Hz[-1:, :, :] * torch.exp(1j * bloch_vector[0] * dL * 1e9 * Hz.shape[0]), Hz[:-1, :, :]),
                                axis=0) + Hz

    Ey = (dHxdz - dHzdx) / (1j * dxes * omega * EPSILON_0 * eps_grid)
    return Ey

def H_to_Ez(Hx, Hy, dxes, omega, eps_grid, bloch_vector=None):
    if bloch_vector is None:
        dHydx = -torch.roll(Hy, shifts=1, dims=1) + Hy
        dHxdy = -torch.roll(Hx, shifts=1, dims=2) + Hx
    else:
        dHydx = -torch.cat((Hy[-1:, :, :] * torch.exp(1j * bloch_vector[0] * dL * 1e9 * Hy.shape[0]), Hy[:-1, :, :]),
                                axis=0) + Hy
        dHxdy = -torch.cat((Hx[:, -1:, :] * torch.exp(1j * bloch_vector[1] * dL * 1e9 * Hx.shape[1]), Hx[:, :-1, :]),
                                axis=1) + Hx

    Ez = (dHydx - dHxdy) / (1j * dxes * omega * EPSILON_0 * eps_grid)
    return Ez
#######################################################################################################
############################### for single data (no batch dimension) ##################################
#######################################################################################################
# def E_to_H(Ex, Ey, Ez, dxes, omega, bloch_vector=None):
#     # dxes: List[List[np.ndarray]], dxes[0] is the grid spacing for E and dxes[1] for H, dxes[0][0] is the grid spacing for x
#     Hx = E_to_Hx(Ey, Ez, dxes, omega, bloch_vector=bloch_vector)
#     Hy = E_to_Hy(Ez, Ex, dxes, omega, bloch_vector=bloch_vector)
#     Hz = E_to_Hz(Ex, Ey, dxes, omega, bloch_vector=bloch_vector)
#     return (Hx, Hy, Hz)
    

# def E_to_Hx(Ey, Ez, dxes, omega, bloch_vector=None):
#     if bloch_vector is None:
#         dEzdy = torch.roll(Ez, -1, dims=1) - Ez # np.roll([1,2,3],-1) = [2,3,1]
#         dEydz = torch.roll(Ey, -1, dims=2) - Ey
#     else:
#         dEzdy = np.concatenate((Ez[:,1:,:], Ez[:,0:1,:]*np.exp(-1j*bloch_vector[1]*dL*1e9*Ez.shape[1])), axis=1) - Ez
#         dEydz = np.concatenate((Ey[:,:,1:], Ey[:,:,0:1]*np.exp(-1j*bloch_vector[2]*dL*1e9*Ey.shape[2])), axis=2) - Ey

#     Hx = (dEzdy / dxes[0][1][None,:,None] - dEydz / dxes[0][2][None,None,:]) / (-1j*omega)
#     return Hx

# def E_to_Hy(Ez, Ex, dxes, omega, bloch_vector=None):
#     if bloch_vector is None:
#         dExdz = torch.roll(Ex, -1, dims=2) - Ex
#         dEzdx = torch.roll(Ez, -1, dims=0) - Ez
#     else:
#         dExdz = np.concatenate((Ex[:,:,1:], Ex[:,:,0:1]*np.exp(-1j*bloch_vector[2]*dL*1e9*Ex.shape[2])), axis=2) - Ex
#         dEzdx = np.concatenate((Ez[1:,:,:], Ez[0:1,:,:]*np.exp(-1j*bloch_vector[0]*dL*1e9*Ez.shape[0])), axis=0) - Ez

#     Hy = (dExdz / dxes[0][2][None,None,:] - dEzdx / dxes[0][0][:,None,None]) / (-1j*omega)
#     return Hy

# def E_to_Hz(Ex, Ey, dxes, omega, bloch_vector=None):
#     if bloch_vector is None:
#         dEydx = torch.roll(Ey, -1, dims=0) - Ey
#         dExdy = torch.roll(Ex, -1, dims=1) - Ex
#     else:
#         dEydx = np.concatenate((Ey[1:,:,:], Ey[0:1,:,:]*np.exp(-1j*bloch_vector[0]*dL*1e9*Ey.shape[0])), axis=0) - Ey
#         dExdy = np.concatenate((Ex[:,1:,:], Ex[:,0:1,:]*np.exp(-1j*bloch_vector[1]*dL*1e9*Ex.shape[1])), axis=1) - Ex

#     Hz = (dEydx / dxes[0][0][:,None,None] - dExdy / dxes[0][1][None,:,None]) / (-1j*omega)
#     return Hz

# def H_to_E(Hx, Hy, Hz, dxes, omega, bloch_vector=None):
#     Ex = H_to_Ex(Hy, Hz, dxes, omega, bloch_vector=bloch_vector)
#     Ey = H_to_Ey(Hz, Hx, dxes, omega, bloch_vector=bloch_vector)
#     Ez = H_to_Ez(Hx, Hy, dxes, omega, bloch_vector=bloch_vector)
#     return (Ex, Ey, Ez)

# def H_to_Ex(Hy, Hz, dxes, omega, bloch_vector=None):
#     if bloch_vector is None:
#         dHzdy = -torch.roll(Hz, 1, dims=1) + Hz # np.roll([1,2,3],1) = [3,1,2]
#         dHydz = -torch.roll(Hy, 1, dims=2) + Hy
#     else:
#         dHzdy = -np.concatenate((Hz[:,-1:,:]*np.exp(1j*bloch_vector[1]*dL*1e9*Hz.shape[1]), Hz[:,:-1,:]), axis=1) + Hz
#         dHydz = -np.concatenate((Hy[:,:,-1:]*np.exp(1j*bloch_vector[2]*dL*1e9*Hy.shape[2]), Hy[:,:,:-1]), axis=2) + Hy

#     Ex = (dHzdy / dxes[1][1][None,:,None] - dHydz / dxes[1][2][None,None,:]) / (1j*omega)
#     return Ex

# def H_to_Ey(Hz, Hx, dxes, omega, bloch_vector=None):
#     if bloch_vector is None:
#         dHxdz = -torch.roll(Hx, 1, dims=2) + Hx
#         dHzdx = -torch.roll(Hz, 1, dims=0) + Hz
#     else:
#         dHxdz = -np.concatenate((Hx[:,:,-1:]*np.exp(1j*bloch_vector[2]*dL*1e9*Hx.shape[2]), Hx[:,:,:-1]), axis=2) + Hx
#         dHzdx = -np.concatenate((Hz[-1:,:,:]*np.exp(1j*bloch_vector[0]*dL*1e9*Hz.shape[0]), Hz[:-1,:,:]), axis=0) + Hz

#     Ey = (dHxdz / dxes[1][2][None,None,:] - dHzdx / dxes[1][0][:,None,None]) / (1j*omega)
#     return Ey

# def H_to_Ez(Hx, Hy, dxes, omega, bloch_vector=None):
#     if bloch_vector is None:
#         dHydx = -torch.roll(Hy, 1, dims=0) + Hy
#         dHxdy = -torch.roll(Hx, 1, dims=1) + Hx
#     else:
#         dHydx = -np.concatenate((Hy[-1:,:,:]*np.exp(1j*bloch_vector[0]*dL*1e9*Hy.shape[0]), Hy[:-1,:,:]), axis=0) + Hy
#         dHxdy = -np.concatenate((Hx[:,-1:,:]*np.exp(1j*bloch_vector[1]*dL*1e9*Hx.shape[1]), Hx[:,:-1,:]), axis=1) + Hx

#     Ez = (dHydx / dxes[1][0][:,None,None] - dHxdy / dxes[1][1][None,:,None]) / (1j*omega)
#     return Ez
########################################################################################################


########################################################################################################
######################################## for batched data ##############################################
########################################################################################################

def E_to_H_batch(Ex, Ey, Ez, dxes, omega, bloch_vector=None):
    Hx = E_to_Hx_batch(Ey, Ez, dxes, omega, bloch_vector=bloch_vector)
    Hy = E_to_Hy_batch(Ez, Ex, dxes, omega, bloch_vector=bloch_vector)
    Hz = E_to_Hz_batch(Ex, Ey, dxes, omega, bloch_vector=bloch_vector)
    return (Hx, Hy, Hz)

def E_to_Hx_batch(Ey, Ez, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dEzdy = torch.roll(Ez, shifts=-1, dims=2) - Ez  # Shift along y-axis
        dEydz = torch.roll(Ey, shifts=-1, dims=3) - Ey  # Shift along z-axis
    else:
        dEzdy = torch.cat((Ez[:, :, 1:, :], Ez[:, :, 0:1, :] * torch.exp(-1j * bloch_vector[1] * dL * 1e9 * Ez.shape[2])),
                          dim=2) - Ez
        dEydz = torch.cat((Ey[:, :, :, 1:], Ey[:, :, :, 0:1] * torch.exp(-1j * bloch_vector[2] * dL * 1e9 * Ey.shape[3])),
                          dim=3) - Ey
    Hx = (dEzdy / dxes[0][1][None,None,:,None] - dEydz / dxes[0][2][None,None,None,:]) / (-1j*omega)
    return Hx

def E_to_Hy_batch(Ez, Ex, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dExdz = torch.roll(Ex, shifts=-1, dims=3) - Ex  # Shift along z-axis
        dEzdx = torch.roll(Ez, shifts=-1, dims=1) - Ez  # Shift along x-axis
    else:
        dExdz = torch.cat((Ex[:, :, :, 1:], Ex[:, :, :, 0:1] * torch.exp(-1j * bloch_vector[2] * dL * 1e9 * Ex.shape[3])),
                          dim=3) - Ex
        dEzdx = torch.cat((Ez[:, 1:, :, :], Ez[:, 0:1, :, :] * torch.exp(-1j * bloch_vector[0] * dL * 1e9 * Ez.shape[1])),
                          dim=1) - Ez

    Hy = (dExdz / dxes[0][2][None,None,None,:] - dEzdx / dxes[0][0][None,:,None,None]) / (-1j*omega)
    return Hy

def E_to_Hz_batch(Ex, Ey, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dEydx = torch.roll(Ey, shifts=-1, dims=1) - Ey  # Shift along x-axis
        dExdy = torch.roll(Ex, shifts=-1, dims=2) - Ex  # Shift along y-axis
    else:
        dEydx = torch.cat((Ey[:, 1:, :, :], Ey[:, 0:1, :, :] * torch.exp(-1j * bloch_vector[0] * dL * 1e9 * Ey.shape[1])),
                          dim=1) - Ey
        dExdy = torch.cat((Ex[:, :, 1:, :], Ex[:, :, 0:1, :] * torch.exp(-1j * bloch_vector[1] * dL * 1e9 * Ex.shape[2])),
                          dim=2) - Ex

    Hz = (dEydx / dxes[0][0][None,:,None,None] - dExdy / dxes[0][1][None,None,:,None]) / (-1j*omega)
    return Hz

def H_to_E_batch(Hx, Hy, Hz, dxes, omega, bloch_vector=None):
    Ex = H_to_Ex_batch(Hy, Hz, dxes, omega, bloch_vector=bloch_vector)
    Ey = H_to_Ey_batch(Hz, Hx, dxes, omega, bloch_vector=bloch_vector)
    Ez = H_to_Ez_batch(Hx, Hy, dxes, omega, bloch_vector=bloch_vector)
    return (Ex, Ey, Ez)

def H_to_Ex_batch(Hy, Hz, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dHzdy = -torch.roll(Hz, shifts=1, dims=2) + Hz  # np.roll([1,2,3],1) = [3,1,2]
        dHydz = -torch.roll(Hy, shifts=1, dims=3) + Hy
    else:
        dHzdy = -torch.cat((Hz[:, -1:, :] * torch.exp(1j * bloch_vector[1] * dL * 1e9 * Hz.shape[1]), Hz[:, :-1, :]),
                                axis=1) + Hz
        dHydz = -torch.cat((Hy[:, :, -1:] * torch.exp(1j * bloch_vector[2] * dL * 1e9 * Hy.shape[2]), Hy[:, :, :-1]),
                                axis=2) + Hy

    Ex = (dHzdy / dxes[1][1][None, None,:,None] - dHydz / dxes[1][2][None,None,None,:]) / (1j*omega)
    return Ex


def H_to_Ey_batch(Hz, Hx, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dHxdz = -torch.roll(Hx, shifts=1, dims=3) + Hx
        dHzdx = -torch.roll(Hz, shifts=1, dims=1) + Hz
    else:
        dHxdz = -torch.cat((Hx[:, :, -1:] * torch.exp(1j * bloch_vector[2] * dL * 1e9 * Hx.shape[2]), Hx[:, :, :-1]),
                                axis=2) + Hx
        dHzdx = -torch.cat((Hz[-1:, :, :] * torch.exp(1j * bloch_vector[0] * dL * 1e9 * Hz.shape[0]), Hz[:-1, :, :]),
                                axis=0) + Hz

    Ey = (dHxdz / dxes[1][2][None, None,None,:] - dHzdx / dxes[1][0][None, :,None,None]) / (1j*omega)
    return Ey

def H_to_Ez_batch(Hx, Hy, dxes, omega, bloch_vector=None):
    if bloch_vector is None:
        dHydx = -torch.roll(Hy, shifts=1, dims=1) + Hy
        dHxdy = -torch.roll(Hx, shifts=1, dims=2) + Hx
    else:
        dHydx = -torch.cat((Hy[-1:, :, :] * torch.exp(1j * bloch_vector[0] * dL * 1e9 * Hy.shape[0]), Hy[:-1, :, :]),
                                axis=0) + Hy
        dHxdy = -torch.cat((Hx[:, -1:, :] * torch.exp(1j * bloch_vector[1] * dL * 1e9 * Hx.shape[1]), Hx[:, :-1, :]),
                                axis=1) + Hx

    Ez = (dHydx / dxes[1][0][None,:,None,None] - dHxdy / dxes[1][1][None,None,:,None]) / (1j*omega)
    return Ez
########################################################################################################

def MAE(d1, d2, boundary_space=0, boundary_space_z=(0, 0)):
    a, b, c = d1.shape
    d1 = d1[boundary_space:a - boundary_space, boundary_space:b - boundary_space,
         boundary_space_z[0]:c - boundary_space_z[1]]
    d2 = d2[boundary_space:a - boundary_space, boundary_space:b - boundary_space,
         boundary_space_z[0]:c - boundary_space_z[1]]
    return torch.mean(torch.abs(d1 - d2)) / torch.mean(torch.abs(d2))

def eps_to_yee(eps):
    # eps shape: (bs, sx, sy, sz)
    # ex = (eps[:,1:,1:,1:] + eps[:,1:,0:-1,1:] + eps[:,1:,1:,0:-1] + eps[:,1:,0:-1,0:-1])/4
    # ey = (eps[:,1:,1:,1:] + eps[:,0:-1,1:,1:] + eps[:,1:,1:,0:-1] + eps[:,0:-1,1:,0:-1])/4
    # ez = (eps[:,1:,1:,1:] + eps[:,0:-1,1:,1:] + eps[:,1:,0:-1,1:] + eps[:,0:-1,0:-1,1:])/4

    ex = (torch.roll(eps, (-1,-1,-1), dims=(1,2,3)) + \
          torch.roll(eps, (-1, 0,-1), dims=(1,2,3)) + \
          torch.roll(eps, (-1,-1, 0), dims=(1,2,3)) + \
          torch.roll(eps, (-1, 0, 0), dims=(1,2,3)))/4
    
    ey = (torch.roll(eps, (-1,-1,-1), dims=(1,2,3)) + \
          torch.roll(eps, ( 0,-1,-1), dims=(1,2,3)) + \
          torch.roll(eps, (-1,-1, 0), dims=(1,2,3)) + \
          torch.roll(eps, ( 0,-1, 0), dims=(1,2,3)))/4
    
    ez = (torch.roll(eps, (-1,-1,-1), dims=(1,2,3)) + \
          torch.roll(eps, ( 0,-1,-1), dims=(1,2,3)) + \
          torch.roll(eps, (-1, 0,-1), dims=(1,2,3)) + \
          torch.roll(eps, ( 0, 0,-1), dims=(1,2,3)))/4

    return torch.stack((ex, ey, ez), dim=-1)

@gin.configurable
def residue_E(E, eps, source, pml_layers, dL, wavelength, bloch_vector=None, batched_compute=False, input_yee=False, Aop=False, ln_R=-10, scale_PML=False):
    # full equation: 
    #                     nabla x (1/μ nabla x E) - ω^2/c^2 yee_ε(x) E = -iωJ 
    # using natural units: 
    #                     C_0 = 1, ε_0 = 1, μ_0 = 1, also here μ = 1
    # simplified equation: 
    #                     1/ω * del E + ω * yee_ε(x) * E = i J

    # dL: in nm, wavelength: in nm, pml_layers: list of 6 integers
    if E.dtype == torch.float32:
        complex_type = torch.complex64
    elif E.dtype == torch.float64:
        complex_type = torch.complex128
    else:
        raise ValueError(f"E must be float32 or float64, but got {E.dtype}")

    bs, x_size, y_size, z_size, _ = E.shape # (bs, sx, sy, sz, 6)

    # create the yee grid eps:
    if not input_yee:
        eps_grids = eps_to_yee(eps)
    else:
        eps_grids = eps

    omega = 2*np.pi/wavelength # narutal unit with C_0 as 1
    dxes = ([np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)], [np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)])
    dxes = apply_scpml(dxes, pml_layers, omega, ln_R=ln_R)
    dxes = [[torch.tensor(i).to(E.device).to(complex_type) for i in dxes[0]], [torch.tensor(i).to(E.device).to(complex_type) for i in dxes[1]]]

    if batched_compute:
        Hx, Hy, Hz = E_to_H_batch(torch.view_as_complex(E[...,0:2]), torch.view_as_complex(E[...,2:4]), torch.view_as_complex(E[...,4:6]), dxes, omega, bloch_vector=bloch_vector)
        # not real E, difference is 1/yee_eps
        Ex, Ey, Ez = H_to_E_batch(Hx, Hy, Hz, dxes, omega, bloch_vector=bloch_vector)
    else:
        Ex, Ey, Ez = [], [], []
        for i in range(bs):
            Hx, Hy, Hz = E_to_H(torch.view_as_complex(E[i,...,0:2]), torch.view_as_complex(E[i,...,2:4]), torch.view_as_complex(E[i,...,4:6]), dxes, omega, bloch_vector=bloch_vector)
            # not real E, difference is 1/yee_eps
            Ex_FD, Ey_FD, Ez_FD = H_to_E(Hx, Hy, Hz, dxes, omega, bloch_vector=bloch_vector)
            Ex.append(Ex_FD)
            Ey.append(Ey_FD)
            Ez.append(Ez_FD)
        Ex, Ey, Ez = torch.stack(Ex), torch.stack(Ey), torch.stack(Ez)
        # these Es are: -1/(ω^2) del E
    
    # -1/(ω^2) del E
    combined_del_E = torch.cat((torch.view_as_real(Ex), torch.view_as_real(Ey), torch.view_as_real(Ez)), dim=-1)
    # yee_eps * E
    yee_times_E = torch.cat((eps_grids[...,0:1]*E[...,0:2], eps_grids[...,1:2]*E[...,2:4], eps_grids[...,2:3]*E[...,4:6]), dim=-1)
    Ax = omega * (-combined_del_E + yee_times_E) # 1/ω del E + ω * yee_eps * E
    
    pre_step, _ = conditioners(dxes, dL)
    if Aop:
        return c2r(pre_step(r2c(Ax))) if scale_PML else Ax
    
    if source is not None:
        rhs = torch.zeros_like(E)
        rhs[..., 0:2] = torch.view_as_real(1j*torch.view_as_complex(source[...,0:2]))
        rhs[..., 2:4] = torch.view_as_real(1j*torch.view_as_complex(source[...,2:4]))
        rhs[..., 4:6] = torch.view_as_real(1j*torch.view_as_complex(source[...,4:6]))
    
    residual = rhs - Ax # residual = i J - 1/w * (del E + w**2 eps E)
    
    return c2r(pre_step(r2c(residual))) if scale_PML else residual

def residual_E_Dinv(E, eps, source, pml_layers, dL, wavelength, bloch_vector=None, batched_compute=False, input_yee=False, Aop=False, ln_R=-10):
    # full equation: 
    #                     nabla x (1/μ nabla x E) - ω^2/c^2 yee_ε(x) E = -iωJ 
    # using natural units: 
    #                     C_0 = 1, ε_0 = 1, μ_0 = 1, also here μ = 1
    # simplified equation: 
    #                     1/ω * del E + ω * yee_ε(x) * E = i J
    # this function is used to compute the inverse of the diagonal part of the operator, D_inv
    if E.dtype == torch.float32:
        complex_type = torch.complex64
    elif E.dtype == torch.float64:
        complex_type = torch.complex128
    else:
        raise ValueError(f"E must be float32 or float64, but got {E.dtype}")

    bs, x_size, y_size, z_size, _ = E.shape # (bs, sx, sy, sz, 6)

    # create the yee grid eps:
    if not input_yee:
        eps_grids = eps_to_yee(eps)
    else:
        eps_grids = eps

    omega = 2*np.pi/wavelength # narutal unit with C_0 as 1
    dxes = ([np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)], [np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)])
    dxes = apply_scpml(dxes, pml_layers, omega, ln_R=ln_R)
    dxes = [[torch.tensor(i).to(E.device).to(complex_type) for i in dxes[0]], [torch.tensor(i).to(E.device).to(complex_type) for i in dxes[1]]]

    # diagonal part for Ex: D_ex = 1/omega * (1/dx11* (1/dx01 + roll_y(1/dx01) + 1/dx12 * (1/dx02 + roll_z(1/dx02))) + omega * yee[...,0]
    # diagonal part for Ey: D_ey = 1/omega * (1/dx12* (1/dx02 + roll_z(1/dx02) + 1/dx10 * (1/dx00 + roll_x(1/dx00))) + omega * yee[...,1]
    # diagonal part for Ez: D_ez = 1/omega * (1/dx10* (1/dx00 + roll_x(1/dx00) + 1/dx11 * (1/dx01 + roll_y(1/dx01))) + omega * yee[...,2]
    ## dx_ij is a 1d vector
    D = torch.zeros((bs, x_size, y_size, z_size, 3), dtype=complex_type)
    D[...,0] = -1/omega * ( (1/dxes[1][1] * (1/dxes[0][1] + torch.roll(1/dxes[0][1], shifts=1)))[None,None,:,None] + \
                            (1/dxes[1][2] * (1/dxes[0][2] + torch.roll(1/dxes[0][2], shifts=1)))[None,None,None,:] ) \
               + omega * eps_grids[...,0]
    D[...,1] = -1/omega * ( (1/dxes[1][2] * (1/dxes[0][2] + torch.roll(1/dxes[0][2], shifts=1)))[None,None,None,:] + \
                            (1/dxes[1][0] * (1/dxes[0][0] + torch.roll(1/dxes[0][0], shifts=1)))[None,:,None,None] ) \
               + omega * eps_grids[...,1]
    D[...,2] = -1/omega * ( (1/dxes[1][0] * (1/dxes[0][0] + torch.roll(1/dxes[0][0], shifts=1)))[None,:,None,None] + \
                            (1/dxes[1][1] * (1/dxes[0][1] + torch.roll(1/dxes[0][1], shifts=1)))[None,None,:,None] ) \
               + omega * eps_grids[...,2]

    D_inv = 1/D
    # D_inv[:, pml_layers[0]:x_size-pml_layers[1], pml_layers[2]:y_size-pml_layers[3], pml_layers[4]:z_size-pml_layers[5], :] = 0
    return D_inv


#################################################################################################
#################### Simplified boundary layer: damping Helmholtz equation ######################
#################################################################################################

# Equation: - 1/ω * nabla x (1/μ nabla x E) + ω yee_ε(x) (1-iγ) E = i J 
# use natural unit: C_0 = 1, ε_0 = 1, μ_0 = 1, also here μ = 1
# so equivalently: 1/ω * del E + ω yee_ε(x) (1-iγ) E = i J

mask_precompute = None
def residue_E_damping(E, eps, source, pml_layers, dL, wavelength, gamma=20.0, bloch_vector=None, batched_compute=False, Aop=False, ln_R=None):
    global mask_precompute
    # dL: in nm, wavelength: in nm, pml_layers: list of 6 integers
    bs, x_size, y_size, z_size, _ = E.shape # (bs, sx, sy, sz, 6)

    # create the yee grid eps:
    eps_grids = eps_to_yee(eps)

    omega = 2*np.pi/wavelength # narutal unit with C_0 as 1
    dxes = ([np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)], [np.array([dL]*x_size), np.array([dL]*y_size), np.array([dL]*z_size)])
    dxes = [[torch.tensor(i).to(E.device).to(torch.complex64) for i in dxes[0]], [torch.tensor(i).to(E.device).to(torch.complex64) for i in dxes[1]]]

    if batched_compute:
        Hx, Hy, Hz = E_to_H_batch(torch.view_as_complex(E[...,0:2]), torch.view_as_complex(E[...,2:4]), torch.view_as_complex(E[...,4:6]), dxes, omega, bloch_vector=bloch_vector)
        Ex, Ey, Ez = H_to_E_batch(Hx, Hy, Hz, dxes, omega, bloch_vector=bloch_vector)
    else:
        Ex, Ey, Ez = [], [], []
        for i in range(bs):
            Hx, Hy, Hz = E_to_H(torch.view_as_complex(E[i,...,0:2]), torch.view_as_complex(E[i,...,2:4]), torch.view_as_complex(E[i,...,4:6]), dxes, omega, bloch_vector=bloch_vector)
            Ex_FD, Ey_FD, Ez_FD = H_to_E(Hx, Hy, Hz, dxes, omega, bloch_vector=bloch_vector)
            Ex.append(Ex_FD)
            Ey.append(Ey_FD)
            Ez.append(Ez_FD)
        # these Es are -1/ω^2 del E
        Ex, Ey, Ez = torch.stack(Ex), torch.stack(Ey), torch.stack(Ez)
    
    # apply the damping in PML layers instead of the actual SC-PML
    if mask_precompute is None:
        mask = torch.ones(Ex.shape[1:], dtype=torch.complex64).to(E.device) # (sx, sy, sz)
        # assume pml thickness on all 6 sides are the same
        for i in range(pml_layers[0]):
            frac_dis = (i+1) / pml_layers[0]
            value = 1 - 1j*frac_dis**2*gamma
            mask[pml_layers[0]-1-i, :, :] = value
            mask[-pml_layers[0]+i, :, :] = value
            mask[:, :pml_layers[0]-i, :] = value
            mask[:, -pml_layers[0]+i, :] = value
            mask[:, :, :pml_layers[0]-i] = value
            mask[:, :, -pml_layers[0]+i] = value
        mask_precompute = mask
    else:
        mask = mask_precompute
    
    E = torch.view_as_complex(E.reshape(bs, x_size, y_size, z_size, 3, 2))
    E = E*mask.unsqueeze(0).unsqueeze(-1)
    E = torch.view_as_real(E).reshape(bs, x_size, y_size, z_size, 6)

    # -1/ω^2 del E
    combined_del_E = torch.cat((torch.view_as_real(Ex), torch.view_as_real(Ey), torch.view_as_real(Ez)), dim=-1)
    
    # yee_eps (1-i gamma) E
    yee_times_E = torch.cat((eps_grids[...,0:1]*E[...,0:2], eps_grids[...,1:2]*E[...,2:4], eps_grids[...,2:3]*E[...,4:6]), dim=-1)
    
    Ax = omega*(-combined_del_E + yee_times_E) # 1/ω del E + ω yee_eps (1-i gamma) E
    if Aop:
        return Ax

    if source is not None:
        rhs = torch.zeros_like(E)
        rhs[..., 0:2] = torch.view_as_real(1j*torch.view_as_complex(source[...,0:2]))
        rhs[..., 2:4] = torch.view_as_real(1j*torch.view_as_complex(source[...,2:4]))
        rhs[..., 4:6] = torch.view_as_real(1j*torch.view_as_complex(source[...,4:6]))

    return rhs - Ax # residual = i J - 1/w * (del E + w**2 yee_eps (1-i gamma) E)


##############################################################################################################
#################################### Helmholtz equation, scalar wave #########################################
##############################################################################################################

def Helmholtz3d_ABC(E, eps, source, pml_layers, dL, wavelength, gamma=1.0, bloch_vector=None, batched_compute=False):
    """
        implement Helmholtz operator in real type on two channels
        E: (batch, N, N, N, 2)
        source: (batch, N, N, N, 2)
        eps: (batch, N, N, N, 1),
    """
    global mask_precompute
    bs, x_size, y_size, z_size, _ = E.shape # (bs, sx, sy, sz, 2)

    E_conv = E.permute(0, 4, 1, 2, 3).reshape(bs*2, 1, x_size, y_size, z_size)    # (batch*2, 1, N, N, N)

    nabla = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],\
                          [[0, 1, 0], [1, -6, 1], [0, 1, 0]],\
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], device=E.device, dtype=E.dtype).unsqueeze(0).unsqueeze(0)
    # the kernel is not trainable
    nabla.requires_grad = False
    nabla_E = torch.zeros((bs*2, 1, x_size, y_size, z_size), device=E.device, dtype=E.dtype)
    nabla_E = torch.nn.functional.conv3d(E_conv, nabla, padding=1)
    nabla_E = nabla_E.reshape(bs, 2, x_size, y_size, z_size)
    nabla_E = nabla_E.permute(0, 2, 3, 4, 1) # (batch, N, N, N, 2)

    k = 2*np.pi * torch.sqrt(eps)/wavelength * dL # wavenumber * dL
    k2 = torch.pow(k, 2)

    Au = -nabla_E - k2[...,None] * E

    if mask_precompute is None:
        mask = torch.zeros((x_size, y_size, z_size), dtype=torch.complex64).to(eps.device) # (sx, sy, sz)
        # assume pml thickness on all 6 sides are the same
        for i in range(pml_layers[0]):
            frac_dis = (i+1) / pml_layers[0]
            value = 1j*frac_dis**2*gamma
            mask[pml_layers[0]-1-i, :, :] = value
            mask[-pml_layers[0]+i, :, :] = value
            mask[:, :pml_layers[0]-i, :] = value
            mask[:, -pml_layers[0]+i, :] = value
            mask[:, :, :pml_layers[0]-i] = value
            mask[:, :, -pml_layers[0]+i] = value
        mask_precompute = mask
    else:
        mask = mask_precompute

    gamma_E = torch.view_as_complex(E) # (bs, sx, sy, sz)
    gamma_E = gamma_E*mask.unsqueeze(0)
    gamma_E = torch.view_as_real(gamma_E)
    
    Au += gamma_E*k2[...,None]
    
    return source-Au


def Helmholtz2d_ABC(E, eps, source, pml_layers, dL, wavelength, gamma=1.0, bloch_vector=None, batched_compute=False):
    """
        implement Helmholtz operator in real type on two channels
        E: (batch, N, N, 2)
        source: (batch, N, N, 2)
        eps: (batch, N, N, 1),
    """
    global mask_precompute
    bs, x_size, y_size, _ = E.shape # (bs, sx, sy, 2)

    E_conv = E.permute(0, 3, 1, 2).reshape(bs*2, 1, x_size, y_size)    # (batch*2, 1, N, N)

    nabla = torch.tensor([[0,  1,  0],\
                          [1, -4,  1],\
                          [0,  1,  0]], device=E.device, dtype=E.dtype).unsqueeze(0).unsqueeze(0)
    # the kernel is not trainable
    nabla.requires_grad = False
    nabla_E = torch.zeros((bs*2, 1, x_size, y_size), device=E.device, dtype=E.dtype)
    nabla_E = torch.nn.functional.conv2d(E_conv, nabla, padding=1)
    nabla_E = nabla_E.reshape(bs, 2, x_size, y_size)
    nabla_E = nabla_E.permute(0, 2, 3, 1) # (batch, N, N, 2)

    k = 2*np.pi * torch.sqrt(eps)/wavelength * dL # wavenumber * dL
    k2 = torch.pow(k, 2)

    Au = -nabla_E - k2[...,None] * E

    if mask_precompute is None:
        mask = torch.zeros((x_size, y_size), dtype=torch.complex64).to(eps.device) # (sx, sy)
        # assume pml thickness on all 6 sides are the same
        for i in range(pml_layers[0]):
            frac_dis = (i+1) / pml_layers[0]
            value = 1j*frac_dis**2*gamma
            mask[pml_layers[0]-1-i, :] = value
            mask[-pml_layers[0]+i, :] = value
            mask[:, :pml_layers[0]-i] = value
            mask[:, -pml_layers[0]+i] = value
        mask_precompute = mask
    else:
        mask = mask_precompute

    gamma_E = torch.view_as_complex(E) # (bs, sx, sy)
    gamma_E = gamma_E*mask.unsqueeze(0)
    gamma_E = torch.view_as_real(gamma_E)
    
    Au += gamma_E*k2[...,None]
    
    return source-Au