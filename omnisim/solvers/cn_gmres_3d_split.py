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

class CNMGRK(OdeSolver):
    def __init__(self, simmer, t0, t_bound, max_step=np.inf,
                 vectorized=False, first_step=None, rk_step=None, **extraneous):
        if rk_step is None:
            rk_step = first_step/4
        self.dt_rk = rk_step
        self.simmer = simmer
        self.dims = simmer.dims
        warn_extraneous(extraneous)
        y0 = simmer.initial_array.flatten()
        self.y = y0.copy()
        self.t=t0
        self.dt=first_step
        dt = self.dt
        atol, rtol = self.simmer.atol, self.simmer.rtol
        #self.rk_solver = RK45(rxn_fun, t0, y0, t_bound, max_step,
        #         rtol, atol, vectorized, first_step, **extraneous)
        scale = simmer.scale
        species, nz, nh, nw, dx = self.dims
        arr_z = 2*nz+species-2
        self.yshape = (arr_z, nh, nw)
        difmat = self.simmer.jacobian.get_dif_jac()
        n_jac = difmat.shape[0]
        self.A = self.cn_lhsA(difmat)
        imat = eye(n_jac, dtype=np.float64)
        self.rhsA = imat+self.dt*(difmat)/2

    def cn_rhsb(self, y):
        return self.rhsA.dot(y.flatten())

    def cn_lhsA(self, difmat):
        dt = self.dt
        n_jac = difmat.shape[0]
        imat = eye(n_jac, dtype=np.float64)
        return imat - (dt/2)*(difmat)

    def _step_impl(self):
        # shorten variables
        species, nz, nh, nw, dx = self.dims
        dt_rk = self.dt_rk
        simmer = self.simmer
        yshape = simmer.yshape
        dyshape = (species, nh, nw)
        z0_slice = simmer.z0_slice
        t, dt, y = self.t, self.dt, self.y
        d_y, diff_terms = simmer.d_y, simmer.diff_terms
        # calc derivatives and jacobians
        # perform step
        b = self.cn_rhsb(y)
        # finish f_ivp_wrapper calculation to make prediction
        diff_d_y = simmer.f_dif_wrapper(t, y)
        y_new, info = gmres(self.A,b,y.flatten()+diff_d_y*dt)
        if info != 0:
            return False, 'gmres failed {}'.format(info)
        i = 0
        while i*dt_rk <= dt:
          d_y = simmer.f_rxn_wrapper(t,y_new)*dt_rk
          d_y.shape = dyshape
          y_new.shape = yshape
          y_new[z0_slice] += d_y
          i += 1
        self.y = y_new
        return True, None
