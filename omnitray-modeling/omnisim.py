# imports

import itertools as itt

import autograd.numpy as np
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

@numba.jit(nopython=True, cache=True)
def calc_diffusion(A, D):
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

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:])',cache=True,nopython=True)
def calc_f(y, d_y, diff_terms, p0):
    # Fitzhugh nagumo model as described on scholarpedia
    dx, Dv, bias, pn, tv, a, b, c = p0
    calc_diffusion(y, diff_terms)

    # v
    d_y[0,:,:] = (dx)*Dv*diff_terms[0,:,:] + bias - y[1,:,:] + pn*y[0,:,:] - tv*(y[0,:,:,]**3)

    # w
    d_y[1,:,:] = a*(b*y[0,:,:] - c*y[1,:,:])

def f_ivp(t, y, d_y, diff_terms, p0, dims, calc_f):
    y.shape = dims
    calc_f(y, d_y, diff_terms, p0)
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
        v_i, w_i = np.arange(species)
        # Make jacobian array
        n_jac = n_h*n_w*species
        # jacobian terms:
        # diffusion : 5 per x,y point, minus
        n_nz = n_jac*5 - 2*(n_h+n_w)*species # + 4*n_h*n_w
        self.dif_vec = np.empty(n_nz,dtype=np.float64)
        self.j1_dif = np.empty(n_nz,dtype=np.int)
        self.j2_dif = np.empty(n_nz,dtype=np.int)

        offsets = ((0,1), (1,0), (0,-1), (-1,0))
        neigh_diff_indices = [[(self.f_ji(x,y,spec), self.f_ji(x+offx,y+offy,spec))
                                   for offy, offx in offsets
                                    for x in np.arange(max([0,-offx]), n_w+min([0,-offx]))
                                     for y in np.arange(max([0,-offy]), n_h+min([0,-offy]))]
                                          for spec in np.arange(species)]

        center_diff_indices = [[(self.f_ji(x,y,spec), self.f_ji(x,y,spec))
                                for x in np.arange(1,n_w-1)
                                 for y in np.arange(1,n_h-1)]
                                   for spec in np.arange(species)]


        edge_diff_indices = [[(self.f_ji(x,y,spec), self.f_ji(x,y,spec))
                                for x in np.arange(1,n_w-1)
                                 for y in [0, n_h-1]] +
                              [(self.f_ji(x,y,spec), self.f_ji(x,y,spec))
                                for x in [0,n_w-1]
                                 for y in np.arange(1,n_h-1)]
                                   for spec in np.arange(species)]

        corner_diff_indices = [[(self.f_ji(x,y,spec), self.f_ji(x,y,spec))
                                                    for x in [0,n_w-1]
                                                     for y in [0,n_h-1]]
                                                       for spec in np.arange(species)]

        self.dif_indices_list = [neigh_diff_indices, center_diff_indices, edge_diff_indices, corner_diff_indices]

        #dv/(dvdt)
        dvdvdt_indices = [(x, y, self.f_ji(x,y,v_i), self.f_ji(x,y,v_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #dv/(dwdt)
        dvdwdt_indices = [(x, y, self.f_ji(x,y,v_i), self.f_ji(x,y,w_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #dw/(dwdt)
        dwdwdt_indices = [(x, y, self.f_ji(x,y,w_i), self.f_ji(x,y,w_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #dn/(dcdt)
        dwdvdt_indices = [(x, y, self.f_ji(x,y,w_i), self.f_ji(x,y,v_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

        self.rxn_indices_list = [dvdvdt_indices, dvdwdt_indices, dwdwdt_indices, dwdvdt_indices]
        n_terms = np.sum([len(xx) for xx in self.rxn_indices_list])
        self.rxn_vec = np.zeros(n_terms, dtype=np.float64)
        self.j1_rxn = np.zeros(n_terms, dtype=np.int)
        self.j2_rxn = np.zeros(n_terms, dtype=np.int)

    def set_p0(self, p0):
        self.p0 = p0
        Dv, bias, pn, tv, a, b, c = p0
        self.D_vec = [Dv, 0]
        self.calc_dif_jac()

    def assign_rxn_vals(self, indices, val_arr, i):
        x1,y1,j1,j2 = np.array(indices).T
        n_inds = len(x1)
        update_slice = slice(i,i+n_inds)
        self.rxn_vec[update_slice] = val_arr[y1,x1]
        self.j1_rxn[update_slice] = j1
        self.j2_rxn[update_slice] = j2
        return i+n_inds

    def assign_dif_vals(self, val, ind_list, i):
        n_inds = len(ind_list)
        update_slice = slice(i,i+n_inds)
        self.dif_vec[update_slice] = val
        ij_arr = np.array(ind_list)
        self.j1_dif[update_slice] = ij_arr[:,0]
        self.j2_dif[update_slice] = ij_arr[:,1]
        return i + n_inds

    def calc_dif_jac(self):
        Dv, bias, pn, tv, a, b, c = self.p0
        D_vec = np.array(self.D_vec)
        dx = self.dx
        species, n_h, n_w, _ = self.dims
        v_i, w_i = np.arange(species)
        i = 0
        neigh_diff_indices, center_diff_indices, edge_diff_indices, corner_diff_indices = self.dif_indices_list
        val_arr = D_vec*dx
        for val, ind_list in zip(val_arr, neigh_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

        val_arr = D_vec*(-4*dx)
        for val, ind_list in zip(val_arr, center_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

        val_arr = D_vec*(-3*dx)
        for val, ind_list in zip(val_arr, edge_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

        val_arr = D_vec*(-2*dx)
        for val, ind_list in zip(val_arr, corner_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

    def calc_rxn_jac(self, t, y):
        Dv, bias, pn, tv, a, b, c = self.p0
        dvdvdt_indices, dvdwdt_indices, dwdwdt_indices, dwdvdt_indices = self.rxn_indices_list
        v_i, w_i = 0,1

        i = 0
        #dv/(dvdt)
        val_arr = 1 - y[0,:,:]**2
        i = self.assign_rxn_vals(dvdvdt_indices, val_arr,i)

        #dv/(dwdt)
        val_arr = -y[1,:,:]
        i = self.assign_rxn_vals(dvdwdt_indices, val_arr,i)

        #dw/(dwdt)
        val_arr = -a*c*np.ones_like(y[0,:,:])
        i = self.assign_rxn_vals(dwdwdt_indices, val_arr,i)

        #dw/(dvdt)
        val_arr = a*b*np.ones_like(y[0,:,:])
        i = self.assign_rxn_vals(dwdvdt_indices, val_arr,i)

    def calc_jac_wrapper(self, t, y):
        species, n_h, n_w, _ = self.dims
        y.shape = (species,n_h,n_w)
        n_jac = species*n_h*n_w
        self.calc_rxn_jac(t,y)
        data_vec = np.concatenate([self.dif_vec, self.rxn_vec])
        j1_vec = np.concatenate([self.j1_dif, self.j1_rxn])
        j2_vec = np.concatenate([self.j2_dif, self.j2_rxn])
        y.shape = species*n_h*n_w
        return sparse.coo_matrix((data_vec, (j1_vec,j2_vec)),shape=(n_jac, n_jac),dtype=np.float64)

class Simulator(object):
    '''
    Reduced version of the simulator used to approximate experimental conditions. Contains tolerances,
    jacobian object, setters for dx and parameters. 
    '''
    def __init__(self):
        self.basedims = np.array([32,32])
        self.set_scale(1)
        self.t_eval = np.linspace(0,30,100)

    def set_scale(self,scale):
        nh, nw = scale*self.basedims
        self.dx = np.power(scale,2)
        species = 2
        self.dims = [species,nh,nw,self.dx]
        self.initial_array = np.random.random(self.dims[:-1]).astype(np.float64)
        self.atol = 1e-4*np.ones_like(self.initial_array)
        self.atol.shape = species*nh*nw
        self.rtol = np.float64(1e-5)
        self.scale = scale
        self.jacobian = Jacobian(self.dims)

    def f_ivp_wrapper(self, t, y):
        return f_ivp(t, y, *self.args)

    def sim(self, p0):
        self.p0 = p0.astype(np.float64)
        Dv, bias, pn, tv, a, b, c = self.p0
        params = np.array([self.dx, Dv, bias, pn, tv, a, b, c], dtype=np.float64)

        species, n_h, n_w, scale = self.dims
        self.args=(np.zeros((species, n_h, n_w), dtype=np.float64,order='C'),
                   np.zeros((species, n_h, n_w), dtype=np.float64,order='C'),
                   params, (species, n_h, n_w), calc_f)

        self.initial_array.shape = (n_h*n_w*species,)
        self.jacobian.set_p0(p0)
        out = itg.solve_ivp(self.f_ivp_wrapper,
                            [self.t_eval.min(), self.t_eval.max()],
                            self.initial_array.copy().astype(np.float64),
                            vectorized=True,
                            method="BDF",
                            atol=self.atol,
                            rtol=self.rtol,
                            t_eval=self.t_eval,
                            jac=self.jacobian.calc_jac_wrapper)
        exp_t = out.t
        exp_y = out.y.T
        exp_y.shape = (len(exp_t), species, n_h, n_w)
        self.initial_array.shape = (species,n_h,n_w,)
        self.sim_arr, self.sim_tvc = exp_y, exp_t

    def make_exp_movie(self):
        n_h, n_w = self.cell_init.shape
        n_frames = len(self.t_eval)
        exp_arr = np.zeros((n_frames, n_h, n_w))
        for col_arr, slice_fun in zip(self.exp_list, self.slice_functions):
            _, s_h, s_w = col_arr.shape
            yslice, xslice = slice_fun(n_h, n_w, s_h, s_w, self.scale)
            exp_arr[:,yslice,xslice] = col_arr
        return exp_arr
