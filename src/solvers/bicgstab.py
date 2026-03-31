import torch
from tqdm import tqdm

def complex_max(t, th=1e-6):
	t_abs = torch.abs(t)
	return t*(t_abs > th) + th*torch.exp(1j*torch.angle(t))*(t_abs <= th)

def c2r(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_real(x).reshape(bs, sx, sy, sz, 6)

def r2c(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_complex(x.reshape(bs, sx, sy, sz, 3, 2))


class mybicgstab:
    def __init__(self, model, myop, max_iter=100, tol=1e-6, apply_M_steps=None):
        super().__init__()
        self.model = model
        self.myop = myop
        self.M = None
        self.max_iter = max_iter
        self.apply_M_steps = self.max_iter if apply_M_steps is None else apply_M_steps
        self.tol = tol
   
    def setup_eps(self, eps, freq):
        self.model.setup(eps, freq)
        self.M = lambda src: r2c(self.model(c2r(src), freq))

    def matvec(self, x):
        raise NotImplementedError
    
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
        _norm = torch.norm(x)
        # print('>>> norm: ', _norm)
        return _norm
    
    @torch.no_grad()
    def solve(self, b, tol=1e-6, max_iter=None, apply_M_steps=None, return_xr_history=False, plot_iters=None, verbose=False):
        max_iter = self.max_iter if max_iter is None else max_iter
        apply_M_steps = self.apply_M_steps if apply_M_steps is None else apply_M_steps
        assert torch.is_complex(b), "b must be complex"

        r = b.clone()
        r0_hat = r.clone()
        rho = self.dot(r0_hat, r)
        p = r.clone()

        x = self.zeros_like(b)

        beta0 = self.vecnorm(r)

        relres_history = [1.0]

        x_history = []
        r_history = []
        if return_xr_history:
            assert plot_iters is not None, "plot_iters must be provided if return_xr_history is True"
        
        if verbose:
            pbar = tqdm(range(max_iter), total=max_iter, desc="BICGSTAB", leave=False)
        else:
            pbar = range(max_iter)

        for j in pbar:
            if j < apply_M_steps:
                y = self.M(p)
            else:
                y = p
            v = self.myop(y)

            r0_hat_v = self.dot(r0_hat, v)

            alpha = rho/complex_max(r0_hat_v, 1e-16)

            h = x + alpha*y
            s = r - alpha * v

            if j < apply_M_steps:
                z = self.M(s)
            else:
                z = s
            t = self.myop(z)

            w = self.dot(t, s)/self.dot(t, t)

            x = h + w * z

            r = s - w*t

            r0_hat_r = self.dot(r0_hat, r)
            beta = r0_hat_r/complex_max(rho, 1e-16) * alpha/complex_max(w, 1e-16)
            rho = r0_hat_r
            p = r + beta*(p-w*v)

            # restart:
            if torch.mean(torch.abs(r0_hat_r)) < 1e-0:
                print("restart")
                r0_hat = r.clone()
                p = r.clone()

            residual_norm = self.vecnorm(p)

            relres_history.append((torch.abs(residual_norm)/torch.abs(beta0)).item())

            if return_xr_history and j in plot_iters:
                x_history.append(x.clone())
                r_history.append(r.clone())

            # Check for convergence
            if verbose:
                pbar.set_description(f"BICGSTAB: Iteration {j}, Res norm: {torch.abs(residual_norm):.2e}, rel-Res norm: {torch.abs(residual_norm)/torch.abs(beta0):.2e}")

            if torch.abs(residual_norm)/torch.abs(beta0) < tol:
                break

        return x, relres_history, x_history, r_history