# imports
import itertools as itt

import numpy as np
import pandas as pd
import os
import sys
import scipy.integrate as itg
import scipy.sparse as sparse

import skimage.io
import skimage.transform
import numba
import emcee

from multiprocessing import Pool, Process, cpu_count
species = 7
col_thresh = 0.01
cs_i, cn_i, cp_i, n_i, a_i, s_i, r_i  = np.arange(species)
cell_inds = (cs_i, cn_i, cp_i)
protein_inds = (s_i, r_i)
ahl_inds = (a_i, )
syn_inds = (s_i, )
rep_inds = (r_i, )

@numba.jit('void(float64[:,:,:],float64[:,:,:])',nopython=True, cache=True)
def laplace_op_noflux_boundaries(A, D):
    # Middle
    D[:,1:-1,1:-1] = A[:,1:-1, 2:] + A[:,1:-1, :-2] + A[:,:-2, 1:-1] + A[:,2:, 1:-1] - 4*A[:,1:-1, 1:-1]
    # Edges
    D[:,0,1:-1] = A[:,0, 2:] + A[:,0, :-2] + A[:,1, 1:-1] - 3*A[:,0, 1:-1]
    D[:,-1,1:-1] = A[:,-1, 2:] + A[:,-1, :-2] + A[:,-2, 1:-1] - 3*A[:,-1, 1:-1]
    D[:,1:-1,0] = A[:,2:,0] + A[:,:-2,0] + A[:,1:-1,1] - 3*A[:,1:-1,0]
    D[:,1:-1,-1] = A[:,2:,-1] + A[:,:-2,-1] + A[:,1:-1,-2] - 3*A[:,1:-1,-1]
    # Corners
    D[:,0,0] = A[:,0,1] + A[:,1,0] - 2*A[:,0,0]
    D[:,-1,0] = A[:,-1,1] + A[:,-2,0] - 2*A[:,-1,0]
    D[:,0,-1] = A[:,0,-2] + A[:,1,-1] - 2*A[:,0,-1]
    D[:,-1,-1] = A[:,-1,-2] + A[:,-2,-1] - 2*A[:,-1,-1]

@numba.jit('void(float64[:,:],float64[:,:])',nopython=True, cache=True)
def laplace_op_noflux_boundaries_onespec(A, D):
    # Middle
    D[1:-1,1:-1] = A[1:-1, 2:] + A[1:-1, :-2] + A[:-2, 1:-1] + A[2:, 1:-1] - 4*A[1:-1, 1:-1]
    # Edges
    D[0,1:-1] = A[0, 2:] + A[0, :-2] + A[1, 1:-1] - 3*A[0, 1:-1]
    D[-1,1:-1] = A[-1, 2:] + A[-1, :-2] + A[-2, 1:-1] - 3*A[-1, 1:-1]
    D[1:-1,0] = A[2:,0] + A[:-2,0] + A[1:-1,1] - 3*A[1:-1,0]
    D[1:-1,-1] = A[2:,-1] + A[:-2,-1] + A[1:-1,-2] - 3*A[1:-1,-1]
    # Corners
    D[0,0] = A[0,1] + A[1,0] - 2*A[0,0]
    D[-1,0] = A[-1,1] + A[-2,0] - 2*A[-1,0]
    D[0,-1] = A[0,-2] + A[1,-1] - 2*A[0,-1]
    D[-1,-1] = A[-1,-2] + A[-2,-1] - 2*A[-1,-1]

@numba.jit('float64[:,:](float64[:,:],float64,float64)',nopython=True, cache=True)
def hill(a, n, k):
    h_ma = 1 - (1 / (1 + (a/k)**n))
    return h_ma

@numba.jit('float64[:,:](float64[:,:],float64,float64)',nopython=True, cache=True)
def dhillda(a, n, k):
    h_ma = (n/k)*((a/k)**(n-1))*(1 / (1 + (a/k)**n)**2)
    return h_ma

@numba.jit('float64[:,:](float64[:,:],float64,float64)',nopython=True, cache=True)
def hillN(a, n, k):
    return 1 / (1 + (a/k)**n)

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:])',nopython=True, cache=True)
def calc_diffusion(y, diff_terms, p0):
    dx,Dc,rc,rS,rR,Hn,Kn,Dn,kn,Da,xa,xs,xS,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=p0
    for c_i in cell_inds+protein_inds:
      laplace_op_noflux_boundaries_onespec(Dc*dx*y[c_i,:,:], diff_terms[c_i,:,:])
    laplace_op_noflux_boundaries_onespec(Dn*dx*y[n_i,:,:], diff_terms[n_i,:,:])
    for a_i in ahl_inds:
      laplace_op_noflux_boundaries_onespec(Da*dx*y[a_i,:,:], diff_terms[a_i,:,:])

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:],float64[:])',cache=True,nopython=True)
def calc_rxn(y, d_y, nut_avail, p0):
    dx,Dc,rc,rS,rR,Hn,Kn,Dn,kn,Da,xa,xs,xS,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=p0
    scale = np.sqrt(dx)
    # Growth term
    nut_avail[:] = hill(y[n_i,:,:], Hn, Kn)

    # Cell growth and diffusion
    for ind in cell_inds:
        #d_y[ind,:,:] = (dx)*Dc*diff_terms[ind,:,:] + rc * nut_avail * y[ind,:,:]
        d_y[ind,:,:] = rc * nut_avail * y[ind,:,:]

    # Nutrient consumption
    d_y[n_i,:,:] =  -kn * nut_avail * np.sum(y[:n_i,:,:],axis=0)

    # AHL and Repressor production
    d_y[a_i,:,:] =  scale * xa * y[s_i,:,:]*(y[cp_i,:,:]+y[cs_i,:,:]) - pa * y[a_i,:,:] 
    d_y[r_i,:,:] = ( xr* hill(y[a_i,:,:], hR, kR)*np.greater(y[cp_i,:,:],od)  - rc * y[r_i,:,:]) * nut_avail - rR * y[r_i,:,:]

    # Synthase production
    d_y[s_i,:,:] = ( xs * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hS, kS) * hillN(y[r_i,:,:], hC, kC) + xS * np.greater(y[cs_i,:,:],od) - rc * y[s_i,:,:]* np.greater(y[cp_i,:,:]+y[cs_i,:,:],od)) * nut_avail - rS * y[s_i,:,:]

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:])',cache=True,nopython=True)
def calc_f(y, d_y, diff_terms, nut_avail, p0):
    calc_diffusion(y, diff_terms, p0)
    calc_rxn(y, d_y, nut_avail, p0)
    d_y[:] = d_y + diff_terms
    # squred x grad
#    diff_terms[:]=0
#    calc_xgrad(y, diff_terms, p0)
#    d_y[:] = d_y + np.power(diff_terms,2)
#    calc_ygrad(y, diff_terms, p0)
#    d_y[:] = d_y + np.power(diff_terms,2)

def f_ivp(t, y, d_y, diff_terms, nut_avail, p0, dims, calc_f):
    y.shape = dims
    calc_f(y, d_y, diff_terms, nut_avail, p0)
    y.shape = np.prod(dims)
    return d_y.flatten()

class Jacobian(object):
    '''
    The Jacobian used is a sparse matrix. On initialization, this class deterines the matrix structure and later can be populated with parameter values
    '''
    def f_ji(self, x, y, spec):
        _, n_h, n_w, _ = self.dims
        return x + n_w*y + n_w*n_h*spec

    def __init__(self, dims):
        species, n_h, n_w, dx = dims
        self.dims = dims
        self.dx = dx
        # Make jacobian array
        n_jac = n_h*n_w*species
        # jacobian terms:
        # diffusion : 5 per x,y point, minus
        n_nz = n_jac*5 - 2*(n_h+n_w)*species # + 4*n_h*n_w
        self.dif_vec = np.zeros(n_nz,dtype=np.float64)
        self.j1_dif = np.zeros(n_nz,dtype=np.int)
        self.j2_dif = np.zeros(n_nz,dtype=np.int)

        offsets = ((0,1), (1,0), (0,-1), (-1,0))
        neigh_diff_indices = [(self.f_ji(x,y,spec), self.f_ji(x+offx,y+offy,spec))
                                   for offy, offx in offsets
                                    for x in np.arange(max([0,-offx]), n_w+min([0,-offx]))
                                     for y in np.arange(max([0,-offy]), n_h+min([0,-offy]))
                                          for spec in np.arange(species)]

        center_diff_indices = [(self.f_ji(x,y,spec), self.f_ji(x,y,spec))
                                for x in np.arange(1,n_w-1)
                                 for y in np.arange(1,n_h-1)
                                   for spec in np.arange(species)]


        x_ranges = (np.arange(1,n_w-1), [0,n_w-1])
        y_ranges = ([0,n_h-1],np.arange(1,n_h-1))
        zipped_ranges = zip(x_ranges, y_ranges)
        edge_diff_indices = [(self.f_ji(x,y,spec), self.f_ji(x,y,spec))
                                  for x_range, y_range in zipped_ranges
                                for x in x_range
                                 for y in y_range
                                   for spec in np.arange(species)]

        corner_diff_indices = [(self.f_ji(x,y,spec), self.f_ji(x,y,spec))
                                                    for x in [0,n_w-1]
                                                     for y in [0,n_h-1]
                                                       for spec in np.arange(species)]

        self.dif_indices_list = [neigh_diff_indices, center_diff_indices, edge_diff_indices, corner_diff_indices]

    def set_p0(self, p0):
        self.p0 = p0
        Dc,rc,rS,rR,Hn,Kn,Dn,kn,Da,xa,xs,xS,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=p0
        self.D_vec = np.zeros(species)
        for spec in np.arange(species):
          if spec in cell_inds+protein_inds:
            self.D_vec[spec] = Dc
          elif spec in ahl_inds:
            self.D_vec[spec] = Da
          elif spec in [n_i]:
            self.D_vec[spec] = Dn
          else:
            self.D_vec[spec] = 0
        self.m_vec = np.ones_like(self.D_vec)
        self.calc_dif_jac()

    def assign_dif_vals(self, val, ind_list, i):
        n_inds = len(ind_list)
        ij_arr = np.array(ind_list)
        update_slice = slice(i,i+n_inds)
        self.j1_dif[update_slice] = ij_arr[:,0]
        self.j2_dif[update_slice] = ij_arr[:,1]
        self.dif_vec[update_slice] = val[ij_arr[:,0]]
        return i + n_inds

    def calc_dif_jac(self):
        D_vec = np.array(self.D_vec)
        m_vec = np.array(self.m_vec)
        species, nh, nw, dx = self.dims
        val_arr = np.zeros((species, nh, nw), dtype=np.float64,order='C')
        for D_val, m, spec_ind in zip(D_vec, m_vec, np.arange(species)):
            val_arr[spec_ind,:,:] = dx*D_val
        i = 0
        ind_list_coeffs = [1, -4, -3, -2]
        for pos_indices, coeff in zip(self.dif_indices_list, ind_list_coeffs):
            i = self.assign_dif_vals(val_arr.flatten()*coeff, pos_indices, i)

    def get_dif_jac(self):
        species, n_h, n_w, dx = self.dims
        n_jac = species*n_h*n_w
        return sparse.coo_matrix((self.dif_vec, (self.j1_dif,self.j2_dif)),shape=(n_jac, n_jac),dtype=np.float64)

class Simulator(object):
    '''
    Instances of this class are initialized with information requried to simulate an experimental pad and compare to data.
    '''
    def __init__(self, scale=4):
        self.basedims = np.array([3,3])
        self.set_scale(scale)
        self.t_eval = np.linspace(0,11*60*60,200)
        ns, nh, nw = self.initial_array.shape

    def set_scale(self,scale):
        logscale = np.log2(scale)
        if not np.isclose(logscale, np.round(logscale)):
            print('rounding scale to nearest power of 2')
            scale = np.int(np.power(2,np.round(logscale)))
        self.scale = scale
        self.dx = np.power(scale,2)
        nh, nw = scale*self.basedims
        self.dims = [species,nh,nw,self.dx]
        self.initial_array = np.zeros(self.dims[:-1])
        atol = np.zeros((species, nh, nw), dtype=np.float64,order='C')# + 1e-7
        for spec in np.arange(species):
          if spec in cell_inds:
            atol[spec,:,:] = 1e-3*np.ones((nh, nw), dtype=np.float64)
          if spec in protein_inds:
            atol[spec,:,:] = 1e-1*np.ones((nh, nw), dtype=np.float64)
          elif spec in ahl_inds:
            atol[spec,:,:]  = 1e-1*np.ones((nh, nw), dtype=np.float64)
          elif spec in [n_i]:
            atol[n_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
        self.atol = atol
        self.atol.shape = species*nh*nw
        self.rtol = np.float64(1e-4)
        self.scale = scale
        self.jacobian = Jacobian(self.dims)

    def f_ivp_wrapper(self, t, y):
        return f_ivp(t, y, *self.args)

    def f_rxn_wrapper(self, t, y):
        d_y, diff_terms, nut_avail, p0, dims, _ = self.args
        y.shape = dims
        calc_rxn(y, d_y, nut_avail, p0)
        y.shape = np.prod(dims)
        return d_y.flatten()

    def f_dif_wrapper(self, t, y):
        d_y, diff_terms, nut_avail, p0, dims, _ = self.args
        y.shape = dims
        calc_diffusion(y, diff_terms, p0)
        y.shape = np.prod(dims)
        return diff_terms.flatten()

    def set_p0(self, p0):
        species, n_h, n_w, dx = self.dims
        self.p0 = p0.astype(np.float64)
        self.params = np.ones(len(p0)+1, dtype=np.float64, order='C')
        self.params[1:] = p0
        self.params[0] = self.dx
        self.jacobian.set_p0(self.p0)
        self.args=(np.zeros((species, n_h, n_w), dtype=np.float64,order='C'),
                   np.zeros((species, n_h, n_w), dtype=np.float64,order='C'),
                   np.zeros((n_h, n_w), dtype=np.float64,order='C'),
                   self.params, (species, n_h, n_w), calc_f)

    def sim(self, p0=None, method='RK45'):
        self.set_p0(p0)
        species, n_h, n_w, dx = self.dims
        self.initial_array.shape = (n_h*n_w*species,)
        out = itg.solve_ivp(self.f_ivp_wrapper,
                            [self.t_eval.min(), self.t_eval.max()],
                            self.initial_array.copy().astype(np.float64),
                            vectorized=True,
                            #method="RK45",
                            method=method,
                            atol=self.atol,
                            rtol=self.rtol,
                            t_eval=self.t_eval)
        self.out = out
        exp_t = out.t
        exp_y = out.y.T
        exp_y.shape = (len(exp_t), species, n_h, n_w)
        self.initial_array.shape = (species,n_h,n_w,)
        self.sim_arr, self.sim_tvc = exp_y, exp_t
