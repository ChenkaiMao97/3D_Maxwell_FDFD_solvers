import torch

def MAE(a, b):
    return torch.mean(torch.abs(a-b))/torch.mean(torch.abs(b))

def c2r(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_real(x).reshape(bs, sx, sy, sz, 6)

def r2c(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_complex(x.reshape(bs, sx, sy, sz, 3, 2))

class mygmres():
    def __init__(self):
        self.myop = None

    def matvec(self, x):
        raise NotImplementedError
    
    def dot(self, x, y):
        raise NotImplementedError
    
    def zeros_like(self, x):
        raise NotImplementedError

    def scale(self, x, a):
        raise NotImplementedError
    
    def axby(self, a, x, b, y):
        raise NotImplementedError

    def vecnorm(self, x):
        return torch.sqrt(self.dot(x, x))
    
    @torch.no_grad()
    def solve(self, b, tol=1e-6, max_iter=100, restart=300, verbose=False, return_xr_history=False, plot_iters=None, complex_type=torch.complex128):
        assert torch.is_complex(b), "b must be complex"
        b = b.to(complex_type)

        beta = self.vecnorm(b)
        # print("Initial residual norm: ", beta)
        if verbose:
            # print("Iteration: ", 0, "Residual norm: ", beta, "Relative residual norm: ", 1.0)
            print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (0, torch.abs(beta), 1.0))
        V = []
        Z = []
        V.append(self.scale(b, 1/beta))
        H = torch.zeros((max_iter + 1, max_iter), dtype=complex_type)
        # num_iter = max_iter

        # Arnoldi process
        relres_history = [1.0]

        x_history = []
        r_history = []
        if return_xr_history:
            assert plot_iters is not None, "plot_iters must be provided if return_xr_history is True"
            
        for j in range(max_iter):
            z = self.M(V[j].to(torch.complex64)).to(complex_type)
            Z.append(z)
            w = self.myop(z)
            for i in range(j + 1):
                H[i, j] = self.dot(w, V[i])
                w = self.axby(1, w, -H[i, j], V[i])
                assert torch.is_complex(w), "w must be complex"

            H[j + 1, j] = self.vecnorm(w)
            V.append(self.scale(w, 1/H[j + 1, j]))

            num_iter = j + 1
            # Solve the least squares problem
            e1 = torch.zeros(num_iter + 1, dtype=complex_type)
            e1[0] = beta
            result = torch.linalg.lstsq(H[:num_iter + 1, :num_iter], e1, rcond=None)
            y = result.solution
            # compute residual norm using y: ||H*y - e1||
            residual_norm = self.vecnorm(H[:num_iter + 1, :num_iter]@y - e1)
            relres_history.append(torch.abs(residual_norm)/torch.abs(beta))

            # Check for convergence
            if verbose:
                print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (num_iter, torch.abs(residual_norm), torch.abs(residual_norm)/torch.abs(beta)))
                
            if torch.abs(residual_norm)/torch.abs(beta) < tol:
                print("break at iteration: ", j+1)
                break

            if return_xr_history and j in plot_iters:
                x = self.zeros_like(b).to(complex_type)
                for i in range(num_iter):
                    x = self.axby(1, x, y[i], Z[i])
                assert x.dtype == complex_type, f"x must be {complex_type}, but got {x.dtype}"

                # Compute the residual
                r = self.axby(1, b, -1, self.myop(x))
                assert torch.is_complex(r) and r.dtype == complex_type, f"r must be type {complex_type}, but got {r.dtype}"
                x_history.append(x)
                r_history.append(r)
        
        x = self.zeros_like(b).to(complex_type)
        for i in range(j+1):
            x = self.axby(1, x, y[i], Z[i])
        assert x.dtype == complex_type, f"x must be {complex_type}, but got {x.dtype}"

        if 1 and not verbose:
            print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (num_iter, torch.abs(residual_norm), torch.abs(residual_norm)/torch.abs(beta)))
        return x, relres_history, x_history, r_history

class mygmrestorch(mygmres):
    # def __init__(self, model, myop, tol=1e-8, max_iter=3):
    def __init__(
        self, 
        model, 
        myop, 
        tol=1e-8, 
        max_iter=3, 
        pre_step=None, 
        post_step=None,
        complex_type=torch.complex128
    ):
        super().__init__()
        self.model = model
        self.myop = myop
        self.M = None
        self.tol = tol
        self.max_iter = max_iter
        self.complex_type = complex_type

        self.pre_step = pre_step if pre_step is not None else lambda x: x
        self.post_step = post_step if post_step is not None else lambda x: x
        
    def setup_eps(self, eps, freq):
        self.model.setup(eps, freq)
        self.M = lambda src: r2c(self.model(c2r(src), freq))
    
    def dot(self, x, y):
        # return torch.sum(x * y)
        prod = torch.sum(torch.conj(x) * y)
        # print('>>> dot product: ', prod)
        return prod
    
    def zeros_like(self, x):
        return torch.zeros_like(x)
    
    def scale(self, x, a):
        return a * x
    
    def axby(self, a, x, b, y):
        return a * x + b * y

    def vecnorm(self, x):
        # return torch.norm(x)    
        _norm = torch.sqrt(self.dot(x, x))
        # print('>>> norm: ', _norm)
        return _norm
    
    @torch.no_grad()
    def solve(self, b, verbose=False, return_xr_history=False, plot_iters=None):
        # b = self.model(b) # left preconditioning
        # check if b is complex
        assert torch.is_complex(b), "b must be complex"

        # left preconditioning
        b = self.pre_step(b)

        with torch.no_grad():
            x, relres_history, x_history, r_history = super().solve(
                b, tol=self.tol, max_iter=self.max_iter, verbose=verbose, 
                return_xr_history=return_xr_history, plot_iters=plot_iters,
                complex_type=self.complex_type)
        
        # right preconditioning
        x = self.post_step(x)
        return x, relres_history, x_history, r_history
    
    def solve_with_restart(self, b, tol, max_iter, restart, verbose):
        # check if b is complex
        assert torch.is_complex(b), "b must be complex"
        
        with torch.no_grad():
            print("Using restart solve with restart: ", restart)
            relres = 1
            norm_b = self.vecnorm(b)
            x = self.zeros_like(b)
            sum_iters = 0
            relres_history = [1.0]
            res = b
            res_norm = norm_b
            relres_restart_history = []
            while(relres > tol and sum_iters < max_iter):
                e, e_relres_history, _, _ = super().solve(res, tol/relres, restart, verbose=verbose)
                sum_iters += len(e_relres_history) - 1
                e_relres_history = [val*res_norm for val in e_relres_history]
                relres_history += e_relres_history[1:]
                x = self.axby(1, x, 1, e)
                res = b - self.myop(x)
                res_norm = self.vecnorm(res)
                relres = torch.abs(res_norm / norm_b)
                relres_restart_history.append(relres)
                print(">>> Relative residual: ", f"{relres:.2e}", "absolute residual: ", f"{res_norm:.2e}")
                if len(e_relres_history) <= 2 or relres < 1.1 * tol: # 1.1 for numerical stability
                    break
                # if len(relres_restart_history) > 1 and relres_restart_history[-1] > 0.95*relres_restart_history[-2]:
                #     print("!!! reduce too small, stop")
                #     break

        # print('>>> residual norm: ', self.vecnorm(b - self.myop(x)))
        print('ITERATION: ', sum_iters)
        return x, relres_history
    
    # def setup_64(self, myop_64, model_64):
    #     self.myop_64 = myop_64
    #     self.model_64 = model_64
    
    # def matvec_64(self, x):
    #     return self.myop_64(self.model_64(x))
        
    # def matvec_mixed(self, x):
    #     return self.myop_64(self.model(x))
    
    # # 64-bit computation is slow on GPU, therefore, we utilize mix-precision computation
    # # i.e. 32-bit computation on solve, and 64-bit computation on residual update
    # def solve_with_restart_mix_precision2(self, b, tol, max_iter, restart, verbose):
    #     print("Using restart solve with restart: ", restart)
    #     print("MIX PRECISION 2")
    #     relres = 1
        
    #     b_64 = b.to(torch.float64, copy=True)
        
    #     norm_b = self.vecnorm(b_64)
    #     x_64 = self.zeros_like(b_64)
    #     sum_iters = 0
    #     relres_history = [1.0]
    #     res_64 = b_64
    #     res_norm = norm_b
    #     relres_restart_history = []
    #     x_restart_history = []
        
    #     while(relres > tol and sum_iters < max_iter):
    #         res = res_64.to(torch.float32, copy=True)
    #         pc_e, e_relres_history = super().solve(res, tol/relres, restart, verbose=verbose)
    #         sum_iters += len(e_relres_history) - 1
    #         e_relres_history = [val*res_norm for val in e_relres_history]
    #         relres_history += e_relres_history[1:]
    #         e = self.model(pc_e)
    #         e_64 = e.to(torch.float64, copy=True)            
    #         x_64 = self.axby(1, x_64, 1, e_64)
    #         res_64 = b_64 - self.myop_64(x_64)
    #         res_norm = self.vecnorm(res_64)
    #         relres = res_norm / norm_b
    #         relres_restart_history.append(relres)
    #         print(">>> Relative residual: ", relres)
    #         x_restart_history += [x_64]
    #         if len(e_relres_history) <= 2:
    #             break
    #         # if len(relres_restart_history) > 1 and (relres_restart_history[-2] - relres_restart_history[-1]) < 0.97*relres_restart_history[-1]:
    #             # break
    #     # print('>>> residual norm: ', self.vecnorm(b - self.myop(x)))
    #     greenprint('ITERATION: %d'%sum_iters)
    #     return x_64, relres_history, x_restart_history
    