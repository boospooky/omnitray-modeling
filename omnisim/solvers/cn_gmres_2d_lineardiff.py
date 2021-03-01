from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import gmres
from .common import (validate_max_step, validate_tol, select_initial_step,
                     norm, EPS, num_jac, validate_first_step,
                     warn_extraneous)
from .base import OdeSolver, DenseOutput

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
        difmat = self.simmer.jacobian.get_dif_jac()
        n_jac = difmat.shape[0]
        self.A = self.cn_lhsA(difmat)
        imat = eye(n_jac, dtype=np.float64)
        self.rhsA = imat+self.dt*(difmat)/2
        super(CNMGRK, self).__init__(rxn_fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        #self.rk_solver = RK45(rxn_fun, t0, y0, t_bound, max_step,
        #         rtol, atol, vectorized, first_step, **extraneous)

    def cn_rhsb(self, y):
        return self.rhsA.dot(y.flatten())

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
        # perform step
        b = self.cn_rhsb(y)
        # finish f_ivp_wrapper calculation to make prediction
        d_y = simmer.f_dif_wrapper(t, y)
        y_new, info = gmres(self.A,b,y+d_y*dt)
        if info != 0:
            return False, 'gmres failed {}'.format(info)
        i = 0
        while i*dt_rk <= dt:
          d_y = simmer.f_rxn_wrapper(t,y_new)
          y_new += d_y * dt_rk
          i += 1
        self.y = y_new
        return True, None
