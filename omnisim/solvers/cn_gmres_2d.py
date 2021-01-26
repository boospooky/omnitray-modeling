from __future__ import division, print_function, absolute_import
import numpy as np
from skimage.transform import downscale_local_mean, rescale
from scipy.linalg import norm
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu, gmres, norm as spnorm
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, EPS, num_jac, validate_first_step,
                     warn_extraneous)
from .base import OdeSolver, DenseOutput
from . import dop853_coefficients
from .rk import RK45
from .. import split_omnisim as oms

def gauss_seidel(A, b, x, n_iter=np.inf, conv_eps=1e-4, w=0.5):
    # Check sparsity
    n = len(x)
    if issparse(A):
        #matnorm = norm
        matnorm = np.max
        diag = A.diagonal()
        # Perform relaxation
        residual = matnorm(A.dot(x)-b)
        while n_iter > 0 and residual > conv_eps:
            for i in np.arange(n):
                sigma = 0
                Arow = A.getrow(i)
                _, j_inds = Arow.nonzero()
                for j in j_inds:
                    if j != i:
                        sigma = sigma + x[j]*Arow[0,j]
                x[i] = x[i] + w*((b[i]-sigma)/diag[i] -x[i])
            n_iter -= 1
            residual = matnorm(A.dot(x)-b)
    else:
        matnorm = norm
        # Perform relaxation
        residual = matnorm(A.dot(x)-b)
        while n_iter > 0 and residual > conv_eps:
            for i in np.arange(n):
                sigma = 0
                for j in np.arange(n):
                    if j != i:
                        sigma = sigma + x[j]*A[i,j]
                x[i] = x[i] + w*((b[i]-sigma)/A[i,i] -x[i])
            n_iter -= 1
            residual = matnorm(A.dot(x)-b)

class cn_mg():
    def __init__(self, simmer, dt):
        self.simmer = simmer
        self.dt = dt
        basedims = simmer.basedims
        simdims = simmer.dims
        species = simdims[0]
        n_levels = np.int(np.log2(simmer.scale))+1
        level_jacobian = list(range(n_levels))
        level_shape = list(range(n_levels))
        scale = simmer.scale
        for i in np.arange(n_levels):
            nh, nw = basedims*np.power(2,n_levels-1-i)
            dx = np.power(scale/2.25,2)
            dims = (species, nh, nw, dx)
            level_jacobian[i] = oms.Jacobian(dims)
            level_jacobian[i].set_p0(simmer.p0)
            level_shape[i] = (species, nh, nw)
        self.level_jacobian = level_jacobian
        self.level_shape = level_shape
        self.n_levels = n_levels

    def cn_rhsb(self, y, h):
        dt = self.dt
        imat = eye(np.prod(self.level_shape[h]), dtype=np.float64)
        jacmat = self.level_jacobian[h].get_dif_jac()
        return y + (dt/2)*(jacmat.dot(y))

    def cn_rhsb_wrxn(self, y):
        dt = self.dt
        difmat = self.level_jacobian[0].get_dif_jac()
        rxnmat = self.level_jacobian[0].get_rxn_jac(0,y)
        f_rxn = self.simmer.f_rxn_wrapper(0,y)
        return y + (dt/2)*(difmat.dot(y) + (1/2)*(dt*rxnmat.dot(y) + 2*f_rxn))

    def cn_lhsA(self, h):
        dt = self.dt
        ns, nh, nw = self.level_shape[h]
        imat = eye(np.prod(self.level_shape[h]), dtype=np.float64)
        jacmat = self.level_jacobian[h].get_dif_jac()
        return imat - (dt/2)*jacmat

    def downscale(self, arr, h_in, h_out):
        assert h_out > h_in
        h_diff = h_out - h_in
        arr.shape = self.level_shape[h_in]
        arr = downscale_local_mean(arr, (1,2**h_diff,2*h_diff))
        arr.shape = np.prod(self.level_shape[h_out])
        return arr

    def upscale(self, arr, h_in, h_out):
        assert h_in > h_out
        h_diff = h_in - h_out
        arr.shape = self.level_shape[h_in]
        arr = rescale(arr, (1,2**h_diff,2**h_diff))
        arr.shape = np.prod(self.level_shape[h_out])
        return arr

    def v_cycle(self, x, b, h, n_iter=50, conv_eps=1e-4):
        A = self.cn_lhsA(h)
        x[:], info = gmres(A, b, x)
        if info != 0:
            print('gmres failed')
            raise Exception
        r = b-A.dot(x)
        rhs = self.downscale(r, h, h+1)
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            Ap = self.cn_lhsA(h+1)
            epsilon, info = gmres(Ap, rhs, epsilon)
            if info != 0:
                print('gmres failed')
                raise Exception
        else:
            self.v_cycle(epsilon, rhs, h+1)
        epsilon = self.upscale(epsilon, h+1, h)
        x[:] = x + epsilon
        x[:], info = gmres(A, b, x)
        if info != 0:
            print('gmres failed')
            raise Exception

    def f_cycle(self, x, b, h, n_iter=50, conv_eps=1e-4):
        A = self.cn_lhsA(h)
        x[:], info = gmres(A, b, x)
        if info != 0:
            print('gmres failed')
            raise Exception
        r = b-A.dot(x)
        rhs = self.downscale(r, h, h+1)
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            Ap = self.cn_lhsA(h+1)
            epsilon, info = gmres(Ap, rhs, epsilon)
            if info != 0:
                print('gmres failed')
                raise Exception
        else:
            self.f_cycle(epsilon, rhs, h+1)
        epsilon = self.upscale(epsilon, h+1, h)
        x[:] = x + epsilon
        x[:], info = gmres(A, b, x)
        if info != 0:
            print('gmres failed')
            raise Exception
        r = b-A.dot(x)
        rhs = self.downscale(r, h, h+1)
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            epsilon, info = gmres(Ap, rhs, epsilon)
        else:
            self.v_cycle(epsilon, rhs, h+1)
        epsilon = self.upscale(epsilon, h+1, h)
        x[:] = x + epsilon
        x[:], info = gmres(A, b, x)
        if info != 0:
            print('gmres failed')
            raise Exception

    def w_cycle(self, x, b, h, n_iter=50, conv_eps=1e-4):
        A = self.cn_lhsA(h)
        x[:], _ = gmres(A, b, x)
        r = b-A.dot(x)
        rhs = self.downscale(r, h, h+1)
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            Ap = self.cn_lhsA(h+1)
            epsilon, _ = gmres(Ap, epsilon, rhs)
        else:
            self.w_cycle(epsilon, rhs, h+1)
        epsilon = self.upscale(epsilon, h+1, h)
        x[:] = x + epsilon
        x[:], _ = gmres(A, b, x)
        r = b-A.dot(x)
        rhs = self.downscale(r, h, h+1)
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            epsilon, _ = gmres(Ap, epsilon, rhs)
        else:
            self.w_cycle(epsilon, rhs, h+1)
        epsilon = self.upscale(epsilon, h+1, h)
        x[:] = x + epsilon
        x[:], _ = gmres(A, b, x)

class CNMGRK(OdeSolver):
    def __init__(self, simmer, t0, t_bound, max_step=np.inf,
                 vectorized=False, first_step=None, rk_step=None, **extraneous):
        self.simmer = simmer
        self.dims = simmer.dims
        species, nh, nw, dx = simmer.dims
        scale = simmer.scale
        self.dt = first_step
        self.rk_dt = rk_step
        self.yshape = (species, nh, nw)
        warn_extraneous(extraneous)
        rxn_fun = simmer.f_rxn_wrapper
        y0 = simmer.initial_array.flatten()
        self.y = y0.copy()
        self.y_rk = y0.copy()
        self.t=t0
        super(CNMGRK, self).__init__(rxn_fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        atol, rtol = self.simmer.atol, self.simmer.rtol
        #self.rk_solver = RK45(rxn_fun, t0, y0, t_bound, max_step,
        #         rtol, atol, vectorized, first_step, **extraneous)

    def cn_rhsb(self, y, difmat):
        dt = self.dt
        n_jac = difmat.shape[0]
        imat = eye(n_jac, dtype=np.float64)
        return (imat+dt*(difmat)/2).dot(y.flatten())

    def cn_lhsA(self, difmat):
        dt = self.dt
        n_jac = difmat.shape[0]
        imat = eye(n_jac, dtype=np.float64)
        return imat - (dt/2)*(difmat)

    def _step_impl(self):
        # shorten variables
        species, nh, nw, dx = self.dims
        simmer = self.simmer
        yshape = self.yshape
        t, dt, dt_rk, y = self.t, self.dt, self.rk_dt, self.y
        # calc derivatives and jacobians
        difmat = self.simmer.jacobian.get_dif_jac()
        # perform step
        A = self.cn_lhsA(difmat)
        b = self.cn_rhsb(y, difmat)
        simmer.f_dif_wrapper(t, y)
        # finish f_ivp_wrapper calculation to make prediction
        d_y = simmer.f_dif_wrapper(t, y)
        y_new, info = gmres(A,b,y+d_y*dt)
        if info != 0:
            return False, 'gmres failed {}'.format(info)
        i = 0
        while i*dt_rk <= dt:
          d_y = simmer.f_rxn_wrapper(t,y_new)
          y_new += d_y * dt_rk
          i += 1
        self.y = y_new
        return True, None
