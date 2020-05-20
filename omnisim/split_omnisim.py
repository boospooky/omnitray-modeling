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
species = 6
col_thresh = 0.01
cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)
cell_inds = (cs_i, cp_i)
ahl_inds = (a_i)
syn_inds = (r_i)

@numba.jit(nopython=True, cache=True)
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

@numba.jit(nopython=True, cache=True)
def hill(a, n, k):
    h_ma = 1 - (1 / (1 + (a/k)**n))
    return h_ma

@numba.jit(nopython=True, cache=True)
def dhillda(a, n, k):
    h_ma = (n/k)*((a/k)**(n-1))*(1 / (1 + (a/k)**n)**2)
    return h_ma

@numba.jit(nopython=True, cache=True)
def hillN(a, n, k):
    return 1 / (1 + (a/k)**n)

@numba.jit(nopython=True, cache=True)
def calc_diffusion(y, diff_terms, p0):
    dx, Dc,  rc, rp,    Kn,  Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = p0
    laplace_op_noflux_boundaries(y, diff_terms)
    diff_terms[cp_i,:,:] *= Dc*dx
    diff_terms[cs_i,:,:] *= Dc*dx
    diff_terms[n_i,:,:] *= Dn*dx
    diff_terms[a_i,:,:] *= Da*dx
    diff_terms[s_i,:,:] = 0
    diff_terms[r_i,:,:] = 0

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:],float64[:])',cache=True,nopython=True)
def calc_rxn(y, d_y, nut_avail, p0):
    dx, Dc,  rc, rp,    Kn,  Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = p0
    # Growth term
    nut_avail[:] = hill(y[n_i,:,:], 2.5, Kn)

    # Cell growth and diffusion
    for ind in cell_inds:
#         d_y[ind,:,:] = (dx)*Dc*diff_terms[ind,:,:] + rc * nut_avail * y[ind,:,:]
        d_y[ind,:,:] = rc * nut_avail * y[ind,:,:]

    # Nutrient consumption
    d_y[n_i,:,:] =  -kn * nut_avail * (y[cp_i,:,:]+y[cs_i,:,:])

    # AHL production
    d_y[a_i,:,:] =  xa * y[s_i,:,:]*(y[cp_i,:,:]+y[cs_i,:,:]) - pa * y[a_i,:,:]

    # Synthase production
    d_y[s_i,:,:] = ( xs * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hS, kS) * hillN(y[r_i,:,:], hC, kC) + xS * np.greater(y[cs_i,:,:],od) - rc * y[s_i,:,:]) * nut_avail - rp * y[s_i,:,:]

    # Repressor production
    d_y[r_i,:,:] = ( xr * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hR, kR) - rc * y[r_i,:,:]) * nut_avail - rp * y[r_i,:,:]

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:])',cache=True,nopython=True)
def calc_f(y, d_y, diff_terms, nut_avail, p0):
    calc_diffusion(y, diff_terms, p0)
    calc_rxn(y, d_y, nut_avail, p0)
    d_y[:] = d_y + diff_terms

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
        cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)
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

        self.rxn_indices_dict = {}
        def rxn_index_helper(v, u):
            # returns indices for dv/du
            return [(x, y, self.f_ji(x,y,v), self.f_ji(x,y,u)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)
        dict_keys = []
        # cs nonzero partials: n, cs
        v_vec = np.arange(species)
        u_vec_list = [[n_i, cs_i],# cs_i
                     [n_i, cp_i],# cp_i
                     [n_i, cs_i, cp_i],# n_i
                     [s_i, cs_i, cp_i, a_i],# a_i
                     [s_i, n_i, a_i, r_i],# s_i
                     [a_i, r_i, n_i]]# r_i
        for v, u_vec in zip(v_vec, u_vec_list):
            for u in u_vec:
                self.rxn_indices_dict[(v,u)] = rxn_index_helper(v,u)

        # cp nonzero partials: n, cp
        v = cp_i
        u_vec = [n_i, cp_i]
        for u in u_vec:
            self.rxn_indices_dict[(v,u)] = rxn_index_helper(v,u)

        # n_i nonzero partials: cs, cp
        v = n_i
        u_vec = [cs_i, cp_i]
        for u in u_vec:
            self.rxn_indices_dict[(v,u)] = rxn_index_helper(v,u)
        n_terms = np.sum([len(xx) for xx in self.rxn_indices_dict.values()])
#         #dc/(dcdt)
#         dcdcdt_indices = [(x, y, self.f_ji(x,y,c_i), self.f_ji(x,y,c_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

#         #dc/(dndt)
#         dcdndt_indices = [(x, y, self.f_ji(x,y,c_i), self.f_ji(x,y,n_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

#         #dn/(dndt)
#         dndndt_indices = [(x, y, self.f_ji(x,y,n_i), self.f_ji(x,y,n_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

#         #dn/(dcdt)
#         dndcdt_indices = [(x, y, self.f_ji(x,y,n_i), self.f_ji(x,y,c_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

#         self.rxn_indices_list = [dcdcdt_indices, dcdndt_indices, dndndt_indices, dndcdt_indices]
#         n_terms = np.sum([len(xx) for xx in self.rxn_indices_list])
        self.rxn_vec = np.zeros(n_terms, dtype=np.float64)
        self.j1_rxn = np.zeros(n_terms, dtype=np.int)
        self.j2_rxn = np.zeros(n_terms, dtype=np.int)

    def set_p0(self, p0):
        self.p0 = p0
        Dc,  rc, rp,    Kn,  Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = p0
        self.D_vec = [Dc, Dc, Dn, Da, Dc, Dc]
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
        Dc,  rc, rp,    Kn,  Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = self.p0
        D_vec = np.array(self.D_vec)
        dx = self.dx
        species, n_h, n_w, dx = self.dims
        cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)
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
        Dc,  rc, rp,    Kn,  Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = self.p0
#         dcdcdt_indices, dcdndt_indices, dndndt_indices, dndcdt_indices = self.rxn_indices_list
        cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)

        i = 0
        nut_avail = hill(y[n_i,:,:], 2.5, Kn)
        dnut_avail = dhillda(y[n_i,:,:], 2.5, Kn)
        cell_inds = [cs_i, cp_i]

        #dc/(dcdt)
        v, u = cs_i, cs_i
        val_arr = rc*nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        v, u = cp_i, cp_i
        val_arr = rc*nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dc/(dndt)
        v, u = cs_i, n_i
        val_arr = rc*dnut_avail*y[cs_i,:,:]
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        v, u = cp_i, n_i
        val_arr = rc*dnut_avail*y[cp_i,:,:]
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dn/(dcdt)
        v, u = n_i, cp_i
        val_arr = -kn*nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        v, u = n_i, cs_i
        val_arr = -kn*nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)
        
        #dn/(dndt)
        v, u = n_i, n_i
        val_arr = -kn*dnut_avail*(y[cp_i,:,:]+y[cs_i,:,:])
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dsdt)
        v, u = a_i, s_i
        val_arr = xa*y[cell_inds,:,:].sum(axis=0)
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dcsdt)
        v, u = a_i, cs_i
        val_arr = xa*y[s_i,:,:]
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dcpdt)
        v, u = a_i, cp_i
        val_arr = xa*y[s_i,:,:]
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dadt)
        v, u = a_i, a_i
        val_arr = -pa*np.ones_like(nut_avail)
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #ds/(dcpdt)
        #v, u = s_i, cp_i
        #val_arr = 0 * nut_avail# hill(y[a_i,:,:], hS, kS) * hillN(y[r_i,:,:], hC, kC) * nut_avail
        #i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)
        #
        #v, u = s_i, cs_i
        #val_arr = 0 * nut_avail
        #i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #ds/(dndt)
        v, u = s_i, n_i
        val_arr = (xs * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hR, kR) * hillN(y[r_i,:,:], hC,
            kC) + xS*np.greater(y[cs_i,:,:],od)- rc * y[s_i,:,:]) * dnut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #ds/(dadt)
        v, u = s_i, a_i
        val_arr = xs * np.greater(y[cp_i,:,:],od) * dhillda(y[a_i,:,:], hS, kS) * hillN(y[r_i,:,:], hC, kC) * nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #ds/(dsdt)
        v, u = s_i, s_i
        val_arr =  -rc * nut_avail - rp
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #ds/(drdt)
        v, u = s_i, r_i
        val_arr = -xs * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hS, kS) * dhillda(y[r_i,:,:], hC, kC) * nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dr/(dcpdt)
        #v, u = r_i, cp_i
        #val_arr = xr * hill(y[a_i,:,:], hR, kR) * nut_avail
        #i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dr/(dadt)
        v, u = r_i, a_i
        val_arr = xr * y[cp_i,:,:]* dhillda(y[a_i,:,:], hR, kR) * nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dr/(drdt)
        v, u = r_i, r_i
        val_arr = -rc * nut_avail - rp
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dr/(dndt)
        v, u = r_i, n_i
        val_arr = (xr * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hR, kR) - rc * y[r_i,:,:]) * dnut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

    def calc_jac_wrapper(self, t, y):
        species, n_h, n_w, dx = self.dims
        y.shape = (species,n_h,n_w)
        n_jac = species*n_h*n_w
        self.calc_rxn_jac(t,y)
        data_vec = np.concatenate([self.dif_vec, self.rxn_vec])
        j1_vec = np.concatenate([self.j1_dif, self.j1_rxn])
        j2_vec = np.concatenate([self.j2_dif, self.j2_rxn])
        y.shape = species*n_h*n_w
        return sparse.coo_matrix((data_vec, (j1_vec,j2_vec)),shape=(n_jac, n_jac),dtype=np.float64)

    def get_rxn_jac(self, t, y):
        species, n_h, n_w, dx = self.dims
        y.shape = (species,n_h,n_w)
        n_jac = species*n_h*n_w
        self.calc_rxn_jac(t,y)
        y.shape = species*n_h*n_w
        return sparse.coo_matrix((self.rxn_vec, (self.j1_rxn,self.j2_rxn)),shape=(n_jac, n_jac),dtype=np.float64)

    def get_dif_jac(self):
        species, n_h, n_w, dx = self.dims
        n_jac = species*n_h*n_w
        return sparse.coo_matrix((self.dif_vec, (self.j1_dif,self.j2_dif)),shape=(n_jac, n_jac),dtype=np.float64)

class Simulator(object):
    '''
    Instances of this class are initialized with information requried to simulate an experimental pad and compare to data.
    '''
    def __init__(self):
        self.basedims = np.array([4,10])
        self.set_scale(32)
        self.t_eval = np.linspace(0,24*60,200)

    def set_scale(self,scale):
        logscale = np.log2(scale)
        if not np.isclose(logscale, np.round(logscale)):
            print('rounding scale to nearest power of 2')
            scale = np.int(np.power(2,np.round(logscale)))
        self.scale = scale
        self.dx = np.power(scale/2.25,2)
        nh, nw = scale*self.basedims
        species = 6
        self.dims = [species,nh,nw,self.dx]
        self.initial_array = np.zeros(self.dims[:-1])
        atol = np.zeros((species, nh, nw), dtype=np.float64,order='C')# + 1e-7
        atol[cs_i,:,:] = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[cp_i,:,:] = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[n_i,:,:]  = 1e-2*np.ones((nh, nw), dtype=np.float64)
        atol[a_i,:,:]  = 1e-2*np.ones((nh, nw), dtype=np.float64)
        atol[s_i,:,:]  = 1e-2*np.ones((nh, nw), dtype=np.float64)
        atol[r_i,:,:]  = 1e-2*np.ones((nh, nw), dtype=np.float64)
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
                            t_eval=self.t_eval,
                            jac=self.jacobian.calc_jac_wrapper)
        self.out = out
        exp_t = out.t
        exp_y = out.y.T
        exp_y.shape = (len(exp_t), species, n_h, n_w)
        self.initial_array.shape = (species,n_h,n_w,)
        self.sim_arr, self.sim_tvc = exp_y, exp_t
