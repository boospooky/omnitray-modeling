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
from .. import nodc_3d_omnisim as oms

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
        scale = simmer.scale
        self.dt = dt
        basedims = simmer.basedims
        nz, nh, nw = scale*basedims
        arr_z = nz+5
        self.yshape = (arr_z, nh, nw)

    def cn_rhsb(self, y, d_y, difmat, rxnmat):
        dt = self.dt
        n_jac = difmat.shape[0]
        imat = eye(n_jac, dtype=np.float64)
        return (imat+dt*(difmat-rxnmat)/2).dot(y.flatten())+dt*d_y.flatten()

    def cn_lhsA(self, difmat, rxnmat):
        dt = self.dt
        n_jac = difmat.shape[0]
        imat = eye(n_jac, dtype=np.float64)
        return imat - (dt/2)*(rxnmat + difmat)

class CNMGRK(OdeSolver):
    def __init__(self, simmer, t0, t_bound, max_step=np.inf,
                 vectorized=False, first_step=None, **extraneous):
        self.simmer = simmer
        self.dims = simmer.dims
        warn_extraneous(extraneous)
        rxn_fun = simmer.f_rxn_wrapper
        y0 = simmer.initial_array.flatten()
        self.y = y0.copy()
        self.t=t0
        self.dt=first_step
        super(CNMGRK, self).__init__(rxn_fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        atol, rtol = self.simmer.atol, self.simmer.rtol
        #self.rk_solver = RK45(rxn_fun, t0, y0, t_bound, max_step,
        #         rtol, atol, vectorized, first_step, **extraneous)
        self.cnmg_solver = cn_mg(simmer, self.dt)

    def _step_impl(self):
        # shorten variables
        species, nz, nh, nw, dx = self.dims
        arr_z = nz+5
        simmer = self.simmer
        yshape = simmer.yshape
        z0_slice = simmer.z0_slice
        t, dt, y = self.t, self.dt, self.y
        d_y, diff_terms = simmer.d_y, simmer.diff_terms
        # calc derivatives and jacobians
        difmat = self.simmer.jacobian.get_dif_jac()
        rxnmat = self.simmer.jacobian.get_rxn_jac(0,y)
        simmer.f_rxn_wrapper(t, y)
        y.shape = yshape
        diff_terms[z0_slice] = d_y
        y.shape = np.prod(yshape)
        # perform step
        A = self.cnmg_solver.cn_lhsA(difmat, rxnmat)
        b = self.cnmg_solver.cn_rhsb(y, diff_terms.flatten(), difmat, rxnmat)
        # finish f_ivp_wrapper calculation to make prediction
        simmer.f_dif_wrapper(t, y)
        diff_terms[z0_slice] = d_y + diff_terms[z0_slice]
        y_new, info = gmres(A,b,y+diff_terms.flatten()*dt, tol=1e-3, atol=1e-3)
        if info != 0:
            return False, 'gmres failed {}'.format(info)
        self.y = y_new
        return True, None
