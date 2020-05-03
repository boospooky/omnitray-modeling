from __future__ import division, print_function, absolute_import
import numpy as np
from skimage.transform import downscale_local_mean, rescale
from scipy.linalg import norm
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu, norm as spnorm
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, EPS, num_jac, validate_first_step,
                     warn_extraneous)
from .base import OdeSolver, DenseOutput
from . import dop853_coefficients
from .. import omnisim as oms

def gauss_seidel(A, b, x, n_iter=np.inf, conv_eps=1e-4, w=0.5):
    # Check sparsity
    if issparse(A):
        matnorm = spnorm
    else:
        matnorm = norm
    # Perform relaxation
    n = len(x)
    #x.shape = (n,1)
    #b.shape = (n,1)
    residual = norm(A.dot(x)-b)
    while n_iter > 0 and residual > conv_eps:
        for i in np.arange(n):
            sigma = 0
            for j in np.arange(n):
                if j != i:
                    sigma = sigma + x[j]*A[i,j]
            x[i] = x[i] + w*((b[i]-sigma)/A[i,i] -x[i])
        n_iter -= 1
        residual = matnorm(A.dot(x)-b)
    return x

class cn_mg():
    def __init__(self, simmer, dt):
        self.dt = dt
        basedims = simmer.basedims
        simdims = simmer.dims
        species = simmer.species
        n_levels = np.floor(np.log2(simmer.scale))
        level_jacobian = [None]*n_levels
        level_shape = [None]*n_levels
        scale = simmer.scale
        for i in np.arange(n_levels):
            nh, nw = basedims*scale
            dx = np.power(scale/2.25,2)
            dims = (species, nh, nw, dx)
            level_jacobian[i] = oms.Jacobian(dims)
            level_shape [i] = (species, nh, nw)

    def cn_rhsb(self, y, h):
        dt = self.dt
        ns, nh, nw = self.level_shape[h]
        y.shape = (ns*nw*nh,1)
        imat = eye(np.prod(self.level_shape[h]), dtype=np.float64)
        jacmat = self.level_jacobian[h].get_dif_jac()
        return y + (dt/2)*(jacmat.dot(y))

    def cn_lhsA(self, h):
        dt = self.dt
        ns, nh, nw = self.level_shape[h]
        y.shape = (ns*nw*nh,1)
        imat = eye(np.prod(self.level_shape[h]), dtype=np.float64)
        jacmat = self.level_jacobian[h].get_dif_jac()
        return imat - (dt/2)*jacmat

    def v_cycle(self, x, h):
        A = self.cn_lhsA(h)
        b = self.cn_rhsb(x.copy(),h)
        x = gauss_seidel(A, b, x, n_iter=50)
        r = A.dot(x)-b
        r.shape = self.level_shape[h]
        rhs = downscale_local_mean(r, (1,2,2))
        rhs.shape = np.prod(self.level_shape[h+1])
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            Ap = self.cn_lhsA(h+1)
            bp = self.cn_rhsb(x.copy(),h+1)
            epsilon = gauss_seidel(Ap, bp, epsilon, n_iter=50)
        else:
            epsilon = self.v_cycle(epsilon, h+1)
        x = x + rescale(epsilon, (1,2,2))
        x = gauss_seidel(A, b, x, n_iter=50)
        return x

    def f_cycle(self, x, h):
        A = self.cn_lhsA(h)
        b = self.cn_rhsb(x.copy(),h)
        x = gauss_seidel(A, b, x, n_iter=50)
        r = A.dot(x)-b
        r.shape = self.level_shape[h]
        rhs = downscale_local_mean(r, (1,2,2))
        rhs.shape = np.prod(self.level_shape[h+1])
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            Ap = self.cn_lhsA(h+1)
            bp = self.cn_rhsb(x.copy(),h+1)
            epsilon = gauss_seidel(Ap, bp, epsilon, n_iter=50)
        else:
            epsilon = self.f_cycle(epsilon, h+1)
        x = x + rescale(epsilon, (1,2,2))
        x = gauss_seidel(A, b, x, n_iter=50)
        r = A.dot(x)-b
        r.shape = self.level_shape[h]
        rhs = downscale_local_mean(r, (1,2,2))
        rhs.shape = np.prod(self.level_shape[h+1])
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            Ap = self.cn_lhsA(h+1)
            bp = self.cn_rhsb(x.copy(),h+1)
            epsilon = gauss_seidel(Ap, bp, epsilon, n_iter=50)
        else:
            epsilon = self.v_cycle(epsilon, h+1)
        x = x + rescale(epsilon, (1,2,2))
        x = gauss_seidel(A, b, x, n_iter=50)
        return x

    def w_cycle(self, x, h):
        A = self.cn_lhsA(h)
        b = self.cn_rhsb(x.copy(),h)
        x = gauss_seidel(A, b, x, n_iter=50)
        r = A.dot(x)-b
        r.shape = self.level_shape[h]
        rhs = downscale_local_mean(r, (1,2,2))
        rhs.shape = np.prod(self.level_shape[h+1])
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            Ap = self.cn_lhsA(h+1)
            bp = self.cn_rhsb(x.copy(),h+1)
            epsilon = gauss_seidel(Ap, bp, epsilon, n_iter=50)
        else:
            epsilon = self.w_cycle(epsilon, h+1)
        x = x + rescale(epsilon, (1,2,2))
        x = gauss_seidel(A, b, x, n_iter=50)
        r = A.dot(x)-b
        r.shape = self.level_shape[h]
        rhs = downscale_local_mean(r, (1,2,2))
        rhs.shape = np.prod(self.level_shape[h+1])
        epsilon = np.zeros_like(rhs)
        if h == self.n_levels-2:
            Ap = self.cn_lhsA(h+1)
            bp = self.cn_rhsb(x.copy(),h+1)
            epsilon = gauss_seidel(Ap, bp, epsilon, n_iter=50)
        else:
            epsilon = self.w_cycle(epsilon, h+1)
        x = x + rescale(epsilon, (1,2,2))
        x = gauss_seidel(A, b, x, n_iter=50)
        return x

