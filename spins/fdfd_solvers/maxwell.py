""" Code pertaining to MaxwellFDFD solver. """
import h5py
import logging
import numpy as np
import os
import random
import requests
import shutil
import tempfile
import time
from typing import List, Optional
import uuid

import spins.fdfd_tools as fdfd_tools

logger = logging.getLogger(__name__)




def upload_files(server_url: str, directory: str, filenames: List[str]):
    """ Uploads a list of files to the given server. """
    for filename in filenames:
        with open(os.path.join(directory, filename), 'rb') as f:
            # Keep retrying in case of network failure.
            succeeded = False
            while not succeeded:
                try:
                    requests.post(server_url,
                                  data={'key': filename},
                                  files={'file': f})
                    succeeded = True
                except requests.exceptions.ConnectionError:
                    logger.exception('ConnectionError during upload: ' +
                                     filename)
                    # Random retry time to prevent DoS on any machine.
                    # Choose somewhere between 5 and 15 seconds.
                    time.sleep(5 + random.uniform(0, 10))


def download_files(server_url: str, directory: str, filenames: List[str]):
    for filename in filenames:
        with open(os.path.join(directory, filename), 'wb') as f:
            # Keep retrying in case of network failure.
            succeeded = False
            while not succeeded:
                try:
                    r = requests.get(server_url + filename)
                    f.write(r.content)
                    succeeded = True
                except requests.exceptions.ConnectionError:
                    logger.exception('ConnectionError during download: ' +
                                     filename)
                    # Random retry time to prevent DoS on any machine.
                    # Choose somewhere between 5 and 15 seconds.
                    time.sleep(5 + random.uniform(0, 10))


def write_to_hd5(filename: str, dataset_name: str, data):
    """ Write a single dataset to a given file. """
    with h5py.File(filename, 'w') as f:
        f.create_dataset(dataset_name, data=data)


def write_field(filename_prefix: str, field):
    """ Write a 3D field to HD5 files.

    Real and imaginary parts of each component are sent as separate files
    for a total of 6 different HD5 files.
    """
    xyz = 'xyz'
    for k in range(3):
        file_prefix = filename_prefix + '_' + xyz[k]
        write_to_hd5(file_prefix + 'r', 'data',
                     np.real(field[k]).astype(np.float64))
        write_to_hd5(file_prefix + 'i', 'data',
                     np.imag(field[k]).astype(np.float64))


class MaxwellSolver:
    # Define default values for solver.
    DEFAULT_ERROR_THRESHOLD = 1e-4  #1e-5 for fwd, 5e-2 for adj
    DEFAULT_MAX_ITERS = 40000
    DEFAULT_MAXWELL_SERVER_PORT = 9041

    def __init__(self,
                 shape: np.ndarray,
                 server=os.getenv("MAXWELL_SERVER", "localhost:9041"),
                 err_thresh=DEFAULT_ERROR_THRESHOLD,
                 max_iters=DEFAULT_MAX_ITERS,
                 solver='CG',
                 ln_R=-16,
                 m=4):
        """ Construct MaxwellFDFD solver.

        Args:
            shape: Shape of simulation domain in grid units.
            server: URL of Maxwell server.
            err_thresh: Relative threshold for terminating solver.
            max_iters: Maximum number of iterations for solver.
        """
        # If there is no port specified for server, append the default.
        if ':' not in server:
            server += ':' + str(MaxwellSolver.DEFAULT_MAXWELL_SERVER_PORT)
        self.shape = shape
        self.server = server
        self.err_thresh = err_thresh
        self.max_iters = max_iters
        self.solver = solver
        self.ln_R = ln_R
        self.m = m
        print("MaxwellSolver with ln_R = ", self.ln_R, "m = ", self.m)

    def solve(self,
              omega: complex,
              dxes: List[List[np.ndarray]],
              J: np.ndarray,
              epsilon: np.ndarray,
              pml_layers: Optional[fdfd_tools.PmlLayers] = None,
              mu: np.ndarray = None,
              pec: np.ndarray = None,
              pmc: np.ndarray = None,
              pemc: np.ndarray = None,
              bloch_vec: np.ndarray = None,
              symmetry: np.ndarray = None,
              adjoint: bool = False,
              E0: np.ndarray = None,
              solver_info: bool = False,
              n_eig: int = 1,
              store_dir: Optional[str] = None):

        
        if store_dir is None:
            store_dir = '/home/maxcmk/Desktop/spins/spins-b/data_gen/'

        input_criteria_exist = os.path.isfile(store_dir+'err_thresh.npy')
        if input_criteria_exist:
            self.err_thresh = np.load(store_dir+'err_thresh.npy')
            self.max_iters = np.load(store_dir+'max_iters.npy')


        if symmetry is None:
            symmetry = np.zeros(3)
        if pemc is None:
            pemc = np.zeros(6)

        server_url = 'http://' + self.server + '/'

        dxes = fdfd_tools.grid.apply_scpml(dxes, pml_layers, omega, ln_R=self.ln_R, m=self.m)

        # Set initial condition to all zeros if not specified.
        if E0 is None:
            E0 = np.zeros_like(epsilon)
        # Set mu to 1 if not specified.
        if mu is None:
            mu = np.ones_like(epsilon)
        if bloch_vec is None:
            bloch_phase = np.ones([3, 3])
        else:
            # TODO(Dries): check this phase calculation for non-uniform grid
            sim_length = np.array([np.real(np.sum(a)) for a in dxes[0]])
            bloch_phase_uniform = np.exp(1j * (sim_length * bloch_vec))
            bloch_phase = np.transpose([bloch_phase_uniform]) @ np.ones([1, 3])
        # Set up using symmetry
        J_unvec = fdfd_tools.unvec(J, self.shape)
        eps_unvec = fdfd_tools.unvec(epsilon, self.shape)
        mu_unvec = fdfd_tools.unvec(mu, self.shape)
        E0_unvec = fdfd_tools.unvec(E0, self.shape)
        slices = [slice(0, sh) for sh in self.shape]
        if symmetry[0] == 1:
            slices[0] = slice(self.shape[0] // 2, self.shape[0])
            pemc[:2] = 1
        elif symmetry[0] == 2:
            slices[0] = slice(self.shape[0] // 2, self.shape[0])
            pemc[:2] = 2
        if symmetry[1] == 1:
            slices[1] = slice(self.shape[1] // 2, self.shape[1])
            pemc[2:4] = 1
        elif symmetry[1] == 2:
            slices[1] = slice(self.shape[1] // 2, self.shape[1])
            pemc[2:4] = 2
        if symmetry[2] == 1:
            slices[2] = slice(self.shape[2] // 2, self.shape[2])
            pemc[4:6] = 1
        elif symmetry[2] == 2:
            slices[2] = slice(self.shape[2] // 2, self.shape[2])
            pemc[4:6] = 2
        shape = [sl.stop - sl.start for sl in slices]
        J = fdfd_tools.vec([j[tuple(slices)] for j in J_unvec])
        E0 = fdfd_tools.vec([e[tuple(slices)] for e in E0_unvec])
        epsilon = fdfd_tools.vec([e[tuple(slices)] for e in eps_unvec])
        mu = fdfd_tools.vec([m[tuple(slices)] for m in mu_unvec])
        dxes = [[dxes[j][i][slices[i]] for i in range(3)] for j in range(2)]



        if adjoint:
            omega = np.conj(omega)
            mu = np.conj(mu)
            epsilon = np.conj(epsilon)
            new_dxes = []
            for i in range(2):
                dx = []
                for j in range(3):
                    dx.append(np.conj(dxes[i][j]))
                new_dxes.append(dx)
            dxes = new_dxes

            spx, spy, spz = np.meshgrid(dxes[1][0],
                                        dxes[1][1],
                                        dxes[1][2],
                                        indexing='ij')
            sdx, sdy, sdz = np.meshgrid(dxes[0][0],
                                        dxes[0][1],
                                        dxes[0][2],
                                        indexing='ij')
            mult = np.multiply
            s = [
                mult(mult(sdx, spy), spz),
                mult(mult(spx, sdy), spz),
                mult(mult(spx, spy), sdz)
            ]
            new_J = fdfd_tools.unvec(J, shape)
            for k in range(3):
                new_J[k] /= np.conj(s[k])
            J = fdfd_tools.vec(new_J)

        # Generate ID based on date, time, and UUID.
        sim_id = time.strftime('%Y%m%d-%H%M%S-') + str(uuid.uuid1())
        sim_name_prefix = 'maxwell-' + sim_id + '.'

        # Create a temporary directory.
        upload_dir = tempfile.mkdtemp()
        local_prefix = os.path.join(upload_dir, sim_name_prefix)

        make_array = lambda a: np.array([a])


        # set solver
        solver = 0
        # If there is a Bloch phase, we must use BiCGSTAB.
        if np.any(bloch_phase != np.ones([3, 3])):
            solver = 1
        if self.solver == 'biCGSTAB':
            solver = 1
        elif self.solver == 'lgmres':
            solver = 2
        elif self.solver == 'Jacobi-Davidson':
            solver = 3

        
        
        epsilon_exist = os.path.isfile(store_dir+'epsilon.npy')
        epsilonDDM_exist = os.path.isfile(store_dir+'epsilon_DDM.npy')
        epsilon_verify_exist = os.path.isfile(store_dir+'epsilon_verify.npy')
        if epsilon_exist:
            e0=np.load(store_dir+'epsilon.npy')
            ex = (e0[1:,1:,1:] + e0[0:-1,1:,1:] + e0[1:,0:-1,1:] + e0[0:-1,0:-1,1:])/4
            ey = (e0[1:,1:,1:] + e0[0:-1,1:,1:] + e0[1:,1:,0:-1] + e0[0:-1,1:,0:-1])/4
            ez = (e0[1:,1:,1:] + e0[1:,0:-1,1:] + e0[1:,1:,0:-1] + e0[1:,0:-1,0:-1])/4

            epsilon=epsilon.reshape(3,-1,e0.shape[1]-1,e0.shape[2]-1)
            z_start_idx=int(epsilon.shape[1]/2)
            z_end_idx = z_start_idx+e0.shape[0]-1
            
            epsilon[0,z_start_idx:z_end_idx,:,:] = epsilon[0,z_start_idx:z_end_idx,:,:]*0.+ex
            epsilon[1,z_start_idx:z_end_idx,:,:] = epsilon[1,z_start_idx:z_end_idx,:,:]*0.+ey
            epsilon[2,z_start_idx:z_end_idx,:,:] = epsilon[2,z_start_idx:z_end_idx,:,:]*0.+ez

            epsilon=epsilon.reshape(-1,)

            print("Store input files")
            e0 = e0[1:,1:,1:]
            e0 = np.swapaxes(e0,0,2)
            np.save(store_dir+'epsilon', e0.astype(np.single))

        elif epsilonDDM_exist:
            e0=np.load(store_dir+'epsilon_DDM.npy')
            ex = (e0[1:,1:,1:] + e0[0:-1,1:,1:] + e0[1:,0:-1,1:] + e0[0:-1,0:-1,1:])/4
            ey = (e0[1:,1:,1:] + e0[0:-1,1:,1:] + e0[1:,1:,0:-1] + e0[0:-1,1:,0:-1])/4
            ez = (e0[1:,1:,1:] + e0[1:,0:-1,1:] + e0[1:,1:,0:-1] + e0[1:,0:-1,0:-1])/4

            epsilon=epsilon.reshape(3,-1,e0.shape[1]-1,e0.shape[2]-1)
            z_start_idx=int(epsilon.shape[1]/2)
            z_end_idx = z_start_idx+e0.shape[0]-1

            epsilon[0,z_start_idx:z_end_idx,:,:] = epsilon[0,z_start_idx:z_end_idx,:,:]*0.+ex
            epsilon[1,z_start_idx:z_end_idx,:,:] = epsilon[1,z_start_idx:z_end_idx,:,:]*0.+ey
            epsilon[2,z_start_idx:z_end_idx,:,:] = epsilon[2,z_start_idx:z_end_idx,:,:]*0.+ez

            print("Store input files DDM")
            epsilon_save = np.real(np.swapaxes(epsilon,1,3))
            np.save(store_dir+'epsilon_DDM', epsilon_save.astype(np.single))

            epsilon=epsilon.reshape(-1,)
        elif epsilon_verify_exist:
            e0=np.load(store_dir+'epsilon_verify.npy')

            ex = (np.roll(e0, (-1,-1,-1), axis=(0,1,2)) + \
                np.roll(e0, (-1, 0,-1), axis=(0,1,2)) + \
                np.roll(e0, (-1,-1, 0), axis=(0,1,2)) + \
                np.roll(e0, (-1, 0, 0), axis=(0,1,2)))/4
            
            ey = (np.roll(e0, (-1,-1,-1), axis=(0,1,2)) + \
                np.roll(e0, ( 0,-1,-1), axis=(0,1,2)) + \
                np.roll(e0, (-1,-1, 0), axis=(0,1,2)) + \
                np.roll(e0, ( 0,-1, 0), axis=(0,1,2)))/4
            
            ez = (np.roll(e0, (-1,-1,-1), axis=(0,1,2)) + \
                np.roll(e0, ( 0,-1,-1), axis=(0,1,2)) + \
                np.roll(e0, (-1, 0,-1), axis=(0,1,2)) + \
                np.roll(e0, ( 0, 0,-1), axis=(0,1,2)))/4

            epsilon = epsilon.reshape(3,-1,e0.shape[1],e0.shape[0])
            epsilon[0,:,:,:] = ex.transpose(2,1,0) # not sure why but following the convention above, i.e. epsilon dimensions are (3, z, y, x), weird!
            epsilon[1,:,:,:] = ey.transpose(2,1,0)
            epsilon[2,:,:,:] = ez.transpose(2,1,0) # not sure why but following the convention above

            epsilon = epsilon.reshape(-1,)

        # Write the grid file.
        gridfile = local_prefix + 'grid'
        with h5py.File(gridfile, 'w') as f:
            f.create_dataset('omega_r', data=make_array(np.real(omega)))
            f.create_dataset('omega_i', data=make_array(np.imag(omega)))
            f.create_dataset('shape', data=shape)
            f.create_dataset('n_eig', data=make_array(n_eig))
            f.create_dataset('max_iters', data=make_array(self.max_iters))
            f.create_dataset('err_thresh', data=make_array(self.err_thresh))
            f.create_dataset('bloch_phase', data=bloch_phase)
            f.create_dataset('pemc', data=pemc)
            f.create_dataset('solver', data=solver)

            xyz = ['x', 'y', 'z']
            for direc in range(3):
                f.create_dataset('sd_' + xyz[direc] + 'r',
                                 data=np.real(dxes[0][direc]).astype(
                                     np.float64))
                f.create_dataset('sd_' + xyz[direc] + 'i',
                                 data=np.imag(dxes[0][direc]).astype(
                                     np.float64))
                f.create_dataset('sp_' + xyz[direc] + 'r',
                                 data=np.real(dxes[1][direc]).astype(
                                     np.float64))
                f.create_dataset('sp_' + xyz[direc] + 'i',
                                 data=np.imag(dxes[1][direc]).astype(
                                     np.float64))
        # Write the rest of the files.
        write_field(local_prefix + 'e', fdfd_tools.unvec(epsilon, shape))
        write_field(local_prefix + 'J', fdfd_tools.unvec(J, shape))
        write_field(local_prefix + 'm', fdfd_tools.unvec(mu, shape))
        write_field(local_prefix + 'A', fdfd_tools.unvec(E0, shape))

        
        
        
        # gridfilestore = store_dir + 'grid'
        # with h5py.File(gridfilestore, 'w') as f:
        #     f.create_dataset('omega_r', data=make_array(np.real(omega)))
        #     f.create_dataset('omega_i', data=make_array(np.imag(omega)))
        #     f.create_dataset('shape', data=shape)
        #     f.create_dataset('n_eig', data=make_array(n_eig))
        #     f.create_dataset('max_iters', data=make_array(self.max_iters))
        #     f.create_dataset('err_thresh', data=make_array(self.err_thresh))
        #     f.create_dataset('bloch_phase', data=bloch_phase)
        #     f.create_dataset('pemc', data=pemc)
        #     f.create_dataset('solver', data=solver)

        #     xyz = ['x', 'y', 'z']
        #     for direc in range(3):
        #         f.create_dataset('sd_' + xyz[direc] + 'r',
        #                          data=np.real(dxes[0][direc]).astype(
        #                              np.float64))
        #         f.create_dataset('sd_' + xyz[direc] + 'i',
        #                          data=np.imag(dxes[0][direc]).astype(
        #                              np.float64))
        #         f.create_dataset('sp_' + xyz[direc] + 'r',
        #                          data=np.real(dxes[1][direc]).astype(
        #                              np.float64))
        #         f.create_dataset('sp_' + xyz[direc] + 'i',
        #                          data=np.imag(dxes[1][direc]).astype(
        #                              np.float64))
        # write_field(store_dir + 'e', fdfd_tools.unvec(epsilon, shape))
        

        # write_field(store_dir + 'J', fdfd_tools.unvec(J, shape))
        # write_field(store_dir + 'm', fdfd_tools.unvec(mu, shape))
        # write_field(store_dir + 'A', fdfd_tools.unvec(E0, shape))



        # Upload files.
        upload_files(server_url, upload_dir, os.listdir(upload_dir))
        # Upload empty request file.
        request_filename = os.path.join(upload_dir, sim_name_prefix + 'request')
        with open(request_filename, 'w') as f:
            f.write('All files uploaded at {0}.'.format(
                time.strftime('%Y-%m-%d-%H:%M:%S')))
        upload_files(server_url, upload_dir, [sim_name_prefix + 'request'])

        # Delete temporary upload directory.
        shutil.rmtree(upload_dir)

        # Create temporary download directory.
        download_dir = tempfile.mkdtemp()

        # Wait for solution.
        def check_existence(filename):
            r = requests.get(server_url + sim_name_prefix + filename)
            return r.status_code == 200

        while True:
            time.sleep(1)  # Wait one second before retry.
            try:
                if check_existence('request'):
                    # Request not yet processed. Wait longer.
                    time.sleep(5)
                elif check_existence('finished'):
                    break
            except requests.exceptions.ConnectionError:
                logger.exception(
                    'ConnectionError while waiting for results. Retrying...')
                # Random retry time to prevent DoS on any machine.
                # Choose somewhere between 5 and 15 seconds.
                time.sleep(5 + random.uniform(0, 10))

        # Download the files.
        if self.solver == 'Jacobi-Davidson':
            filenames = [
                sim_name_prefix + 'Q' + str(i) + '_' + comp + quad
                for comp in 'xyz' for quad in 'ri' for i in range(n_eig)
            ]
            filenames += [sim_name_prefix + 'q' + quad for quad in 'ri']
        else:
            filenames = [
                sim_name_prefix + 'E_' + comp + quad
                for comp in 'xyz'
                for quad in 'ri' 
            ]
            filenames += [sim_name_prefix + 'time_info', sim_name_prefix+'status']
            #filenames = [file+'.h5' for file in filenames]
            

        download_files(server_url, download_dir, filenames)

        # define apply_symmetry
        def apply_symmetry(E, symmetry):
            dummy = np.expand_dims(np.zeros_like(E[1][0, :, :]), 0)
            if symmetry[0] == 1:
                E[0] = np.concatenate((np.flip(E[0], 0), E[0]), axis=0)
                E[1] = np.concatenate(
                    (dummy, -np.flip(E[1][1:, :, :], 0), E[1]), axis=0)
                E[2] = np.concatenate(
                    (dummy, -np.flip(E[2][1:, :, :], 0), E[2]), axis=0)
            elif symmetry[0] == 2:
                E[0] = np.concatenate((-np.flip(E[0], 0), E[0]), axis=0)
                E[1] = np.concatenate((dummy, np.flip(E[1][1:, :, :], 0), E[1]),
                                      axis=0)
                E[2] = np.concatenate((dummy, np.flip(E[2][1:, :, :], 0), E[2]),
                                      axis=0)

            dummy = np.expand_dims(np.zeros_like(E[1][:, 0, :]), 1)
            if symmetry[1] == 1:
                E[0] = np.concatenate(
                    (dummy, -np.flip(E[0][:, 1:, :], 1), E[0]), axis=1)
                E[1] = np.concatenate((np.flip(E[1], 1), E[1]), axis=1)
                E[2] = np.concatenate(
                    (dummy, -np.flip(E[2][:, 1:, :], 1), E[2]), axis=1)
            elif symmetry[1] == 2:
                E[0] = np.concatenate((dummy, np.flip(E[0][:, 1:, :], 1), E[0]),
                                      axis=1)
                E[1] = np.concatenate((-np.flip(E[1], 1), E[1]), axis=1)
                E[2] = np.concatenate((dummy, np.flip(E[2][:, 1:, :], 1), E[2]),
                                      axis=1)

            dummy = np.expand_dims(np.zeros_like(E[1][:, :, 0]), 2)
            if symmetry[2] == 1:
                E[0] = np.concatenate(
                    (dummy, -np.flip(E[0][:, :, 1:], 2), E[0]), axis=2)
                E[1] = np.concatenate(
                    (dummy, -np.flip(E[1][:, :, 1:], 2), E[1]), axis=2)
                E[2] = np.concatenate((np.flip(E[2], 2), E[2]), axis=2)
            elif symmetry[2] == 2:
                E[0] = np.concatenate((dummy, np.flip(E[0][:, :, 1:], 2), E[0]),
                                      axis=2)
                E[1] = np.concatenate((dummy, np.flip(E[1][:, :, 1:], 2), E[1]),
                                      axis=2)
                E[2] = np.concatenate((-np.flip(E[2], 2), E[2]), axis=2)

            return E

        # Load in electric field.
        if self.solver == 'Jacobi-Davidson':
            E = []
            for i in range(n_eig):
                Q = []
                for comp in 'xyz':
                    file_prefix = os.path.join(
                        download_dir,
                        sim_name_prefix + 'Q' + str(i) + '_' + comp)
                    field_comp = None
                    with h5py.File(file_prefix + 'r') as f:
                        field_comp = f['data'][:].astype(np.complex128)
                    with h5py.File(file_prefix + 'i') as f:
                        field_comp += 1j * f['data'][:].astype(np.complex128)
                    Q.append(field_comp)
                E.append(apply_symmetry(Q, symmetry))

        else:
            E = []
            ## solver  = CG

            for comp in 'xyz':
                file_prefix = os.path.join(download_dir,
                                           sim_name_prefix + 'E_' + comp)
                
                field_comp = None
                print("file_prefix: ", file_prefix)
                with h5py.File(file_prefix + 'r') as f:
                    
                    
                    field_comp = f['data'][:].astype(np.complex128)
                with h5py.File(file_prefix + 'i') as f:
                    
                    field_comp += 1j * f['data'][:].astype(np.complex128)
                E.append(field_comp)
            E = apply_symmetry(E, symmetry)

            print("Store output files")
            if epsilon_exist:
                # If forward field
                if np.isnan(E[0][0,0,0].astype(np.csingle)):
                    print("NaN encountered!")
                    print(self.err_thresh)
                    print(self.max_iters)
                elif np.sum(E[0]) == 0:
                    print("All 0 encountered!")
                    print(self.err_thresh)
                    print(self.max_iters)
                else:
                    pref = ''
                    np.save(store_dir+pref+'Ex', E[0][:,:,z_start_idx:z_end_idx].astype(np.csingle))
                    np.save(store_dir+pref+'Ey', E[1][:,:,z_start_idx:z_end_idx].astype(np.csingle))
                    np.save(store_dir+pref+'Ez', E[2][:,:,z_start_idx:z_end_idx].astype(np.csingle))

                    # np.save(store_dir+'Ex', E[0])
                    # np.save(store_dir+'Ey', E[1])
                    # np.save(store_dir+'Ez', E[2])


                # If adjoint field
                # if np.isnan(E[0][0,0,0].astype(np.csingle)):
                #     print("NaN encountered!")
                #     print(MaxwellSolver.DEFAULT_ERROR_THRESHOLD)

                # elif np.sum(E[0]) == 0:
                #     print("All 0 encountered!")
                #     print(MaxwellSolver.DEFAULT_ERROR_THRESHOLD)
                # else:
                #     pref = ''
                #     np.save(store_dir+pref+'Ex_adj', E[0][:,:,z_start_idx:z_end_idx].astype(np.csingle))
                #     np.save(store_dir+pref+'Ey_adj', E[1][:,:,z_start_idx:z_end_idx].astype(np.csingle))
                #     np.save(store_dir+pref+'Ez_adj', E[2][:,:,z_start_idx:z_end_idx].astype(np.csingle))
            else:
                pref = str(self.err_thresh)
                np.save(store_dir+pref+'Ex', E[0])
                np.save(store_dir+pref+'Ey', E[1])
                np.save(store_dir+pref+'Ez', E[2])

        # Undo right preconditioner for adjoint calculations.
        if adjoint:
            for i in range(3):
                E[i] = np.multiply(E[i], np.conj(s[i]))

        # Remove downloaded files.
        if solver_info:
            file_prefix = os.path.join(download_dir, sim_name_prefix)
            field_comp = None
            # print("all files: ", os.listdir(download_dir))
            with h5py.File(file_prefix + 'time_info') as f:
                solve_time = f['data'][0]
            text_file = open(file_prefix + 'status', "r")
            lines = text_file.read().split('\n')
            error = [float(l) for l in lines[:-1]]

            shutil.rmtree(download_dir)
            return fdfd_tools.vec(E), solve_time, error
        elif self.solver == 'Jacobi-Davidson':
            file_prefix = os.path.join(download_dir, sim_name_prefix)
            q = None
            with h5py.File(file_prefix + 'qr') as f:
                q = f['data'][()].astype(np.complex128)
            with h5py.File(file_prefix + 'qi') as f:
                q += 1j * f['data'][()].astype(np.complex128)

            shutil.rmtree(download_dir)

            return [fdfd_tools.vec(Q) for Q in E], q**(0.5)
        else:
            shutil.rmtree(download_dir)
            return fdfd_tools.vec(E)
