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
species = 12
col_thresh = 0.01
cr_i, csa_i, csb_i, cpa_i, cpb_i, n_i, aa_i, ab_i, sa_i, sb_i, ra_i, rb_i  = np.arange(species)
cell_inds = (cr_i, csa_i, csb_i, cpa_i, cpb_i)
protein_inds = (sa_i, sb_i, ra_i, rb_i)
ahl_inds = (aa_i, ab_i)
syn_inds = (ra_i, rb_i)
p0_keys='dx','Dc','rc','rS','rR','Hn','Kn','Dn','kn','Daa','Dab','xa','xsa','xsb','xS','xr','hS','kS','hR','kR','hC','kC','pa','leak','od'

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

@numba.jit('void(float64[:,:],float64[:,:])',nopython=True, cache=True)
def grad_x_noflux_onespec(A, D):
    # Middle
    D[:,1:-1] = A[:, 2:] - A[:, :-2]
    # Edges
    D[:,0] = A[:,1] - A[:,0]
    D[:,-1] = A[:,-1]- A[:,-2] 

@numba.jit('void(float64[:,:],float64[:,:])',nopython=True, cache=True)
def grad_y_noflux_onespec(A, D):
    # Middle
    D[1:-1,:] = A[2:,:] - A[:-2,:]
    # Edges
    D[0,:] = A[1,:] - A[0,:]
    D[-1,:] = A[-1,:]- A[-2,:]

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
    dx,Dc,rc,rS,rR,Hn,Kn,Dn,kn,Daa,Dab,xa,xs,xSa,xSb,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=p0
    for c_i in cell_inds+protein_inds:
      laplace_op_noflux_boundaries_onespec(Dc*dx*y[c_i,:,:], diff_terms[c_i,:,:])
    laplace_op_noflux_boundaries_onespec(Dn*dx*y[n_i,:,:], diff_terms[n_i,:,:])
    laplace_op_noflux_boundaries_onespec(Daa*dx*y[aa_i,:,:], diff_terms[aa_i,:,:])
    laplace_op_noflux_boundaries_onespec(Dab*dx*y[ab_i,:,:], diff_terms[ab_i,:,:])

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:])',nopython=True, cache=True)
def calc_xgrad(y, diff_terms, p0):
    dx,Dc,rc,rS,rR,Hn,Kn,Dn,kn,Daa,Dab,xa,xs,xSa,xSb,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=p0
    for cell_ind in cell_inds+protein_inds:
      grad_x_noflux_onespec(Dc*dx*y[cell_ind,:,:], diff_terms[cell_ind,:,:])

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:])',nopython=True, cache=True)
def calc_ygrad(y, diff_terms, p0):
    dx,Dc,rc,rS,rR,Hn,Kn,Dn,kn,Daa,Dab,xa,xs,xSa,xSb,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=p0
    for cell_ind in cell_inds+protein_inds:
      grad_y_noflux_onespec(Dc*dx*y[cell_ind,:,:], diff_terms[cell_ind,:,:])

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:],float64[:])',cache=True,nopython=True)
def calc_rxn(y, d_y, nut_avail, p0):
    dx,Dc,rc,rS,rR,Hn,Kn,Dn,kn,Daa,Dab,xa,xs,xSa,xSb,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=p0
    # Growth term
    nut_avail[:] = hill(y[n_i,:,:], Hn, Kn)

    # Cell growth and diffusion
    for ind in cell_inds:
        #d_y[ind,:,:] = (dx)*Dc*diff_terms[ind,:,:] + rc * nut_avail * y[ind,:,:]
        d_y[ind,:,:] = rc * nut_avail * y[ind,:,:]

    # Nutrient consumption
    d_y[n_i,:,:] =  -kn * nut_avail * np.sum(y[:n_i,:,:],axis=0)

    # AHL production
    d_y[aa_i,:,:] =  xa * y[sa_i,:,:]*(y[cpa_i,:,:]+y[csa_i,:,:]) - pa * y[aa_i,:,:] - dx * Daa * 0.2 * y[aa_i,:,:]
    d_y[ab_i,:,:] =  xa * y[sb_i,:,:]*(y[cpb_i,:,:]+y[csb_i,:,:]) - pa * y[ab_i,:,:] - dx * Dab * 0.2 * y[ab_i,:,:]

    # Synthase production
    d_y[sa_i,:,:] = ( xs * np.greater(y[cpa_i,:,:],od) * hill(y[aa_i,:,:], hS, kS) * hillN(y[ra_i,:,:], hC, kC) + xSa * np.greater(y[csa_i,:,:],od) - rc * y[sa_i,:,:]* np.greater(y[cpa_i,:,:]+y[csa_i,:,:],od)) * nut_avail - rS * y[sa_i,:,:]
    d_y[sb_i,:,:] = ( xs * np.greater(y[cpb_i,:,:],od) * hill(y[ab_i,:,:], hS, kS) * hillN(y[rb_i,:,:], hC, kC) + xSb * np.greater(y[csb_i,:,:],od) - rc * y[sb_i,:,:]* np.greater(y[cpb_i,:,:]+y[csb_i,:,:],od)) * nut_avail - rS * y[sb_i,:,:]

    # Repressor production
    d_y[ra_i,:,:] = ( xr  * hill(y[aa_i,:,:], hR, kR) - rc * y[ra_i,:,:]) * nut_avail * np.greater(y[cpa_i,:,:],od) - rR * y[ra_i,:,:]
    d_y[rb_i,:,:] = ( xr  * hill(y[ab_i,:,:], hR, kR) - rc * y[rb_i,:,:]) * nut_avail * np.greater(y[cpb_i,:,:],od) - rR * y[rb_i,:,:]

    # output production
    # d_y[o_i,:,:] = ( xr  * hill(y[aa_i,:,:], hR, kR) * hill(y[ab_i,:,:], hR, kR) - rc * y[o_i,:,:]) * nut_avail * np.greater(y[cr_i,:,:],od) - rR * y[o_i,:,:]

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

        self.rxn_indices_dict = {}
        def rxn_index_helper(v, u):
            # returns indices for dv/du
            return [(x, y, self.f_ji(x,y,v), self.f_ji(x,y,u)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)
        # csa_i, csb_i, cpa_i, cpb_i, n_i, aa_i, ab_i, sa_i, sb_i, ra_i, rb_i = np.arange(species)
        dict_keys = []
        # cs nonzero partials: n, cs
        v_vec = np.arange(species)
        u_vec_list = [[n_i, cr_i],# cr_i
                     [n_i, csa_i],# csa_i
                     [n_i, csb_i],# csb_i
                     [n_i, cpa_i],# cpa_i
                     [n_i, cpb_i],# cpb_i
                     [n_i, cr_i, csa_i, csb_i, cpa_i, cpb_i],# n_i
                     [sa_i, csa_i, cpa_i, aa_i],# aa_i
                     [sb_i, csb_i, cpb_i, ab_i],# ab_i
                     [sa_i, n_i, aa_i, ra_i],# sa_i
                     [sb_i, n_i, ab_i, rb_i],# sb_i
                     [aa_i, ra_i, n_i],# ra_i
                     [ab_i, rb_i, n_i]]# rb_i
                     #[aa_i, ab_i, o_i, n_i]]# o_i
        for v, u_vec in zip(v_vec, u_vec_list):
            for u in u_vec:
                self.rxn_indices_dict[(v,u)] = rxn_index_helper(v,u)
        n_terms = np.sum([len(xx) for xx in self.rxn_indices_dict.values()])
        self.rxn_vec = np.zeros(n_terms, dtype=np.float64)
        self.j1_rxn = np.zeros(n_terms, dtype=np.int)
        self.j2_rxn = np.zeros(n_terms, dtype=np.int)

    def set_p0(self, p0):
        self.p0 = p0
        Dc,rc,rS,rR,Hn,Kn,Dn,kn,Daa,Dab,xa,xs,xSa,xSb,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=p0
        self.D_vec = [Dc, Dc, Dc, Dc, Dc, Dn, Daa, Dab, Dc, Dc, Dc, Dc, Dc]
        self.m_vec = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

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
        ij_arr = np.array(ind_list)
        update_slice = slice(i,i+n_inds)
        self.j1_dif[update_slice] = ij_arr[:,0]
        self.j2_dif[update_slice] = ij_arr[:,1]
        self.dif_vec[update_slice] = val[ij_arr[:,0]]
        return i + n_inds

    def calc_dif_jac(self, t, y):
        D_vec = np.array(self.D_vec)
        m_vec = np.array(self.m_vec)
        species, nh, nw, dx = self.dims
        y.shape = (species, nh, nw)
        val_arr = np.zeros_like(y)
        for D_val, m, spec_ind in zip(D_vec, m_vec, np.arange(species)):
            if m < 1:
                val_arr[spec_ind,:,:] = 0
            elif m > 1:
                val_arr[spec_ind,:,:] = dx*D_val*m*np.power(y[spec_ind,:,:], (m-1))
            else:
                val_arr[spec_ind,:,:] = dx*D_val
        i = 0
        ind_list_coeffs = [1, -4, -3, -2]
        for pos_indices, coeff in zip(self.dif_indices_list, ind_list_coeffs):
            i = self.assign_dif_vals(val_arr.flatten()*coeff, pos_indices, i)

    def calc_rxn_jac(self, t, y):
        Dc,rc,rS,rR,Hn,Kn,Dn,kn,Daa,Dab,xa,xs,xSa,xSb,xr,hS,kS,hR,kR,hC,kC,pa,leak,od=self.p0
        dx = self.dx
#         dcdcdt_indices, dcdndt_indices, dndndt_indices, dndcdt_indices = self.rxn_indices_list
        i = 0
        nut_avail = hill(y[n_i,:,:], Hn, Kn)
        dnut_avail = dhillda(y[n_i,:,:], Hn, Kn)

        #dc/(dcdt)
        val_arr = rc*nut_avail
        for c_i in cell_inds:
          v, u = c_i, c_i
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dc/(dndt)
        u = n_i
        for c_i in cell_inds:
          v = c_i
          val_arr = rc*dnut_avail*y[c_i,:,:]
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dn/(dcdt)
        v = n_i
        for c_i in cell_inds:
          u = c_i
          val_arr = -kn*nut_avail
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dn/(dndt)
        v, u = n_i, n_i
        val_arr = -kn*dnut_avail*np.sum(y[cell_inds,:,:],axis=0)
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dsdt)
        for v, u, strain_inds in ([aa_i, sa_i, [cpa_i, csa_i]],[ab_i, sb_i, [cpb_i, csb_i]]):
          val_arr = xa*y[strain_inds,:,:].sum(axis=0)
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dcsdt) and /dcpdt
        for v, s_i, u in ([aa_i, sa_i, csa_i],[ab_i, sb_i, csb_i],[aa_i, sa_i, cpa_i],[ab_i, sb_i, cpb_i]):
          val_arr = xa*y[s_i,:,:]
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dadt)
        for a_i, Da in ((aa_i, Daa), (ab_i, Dab)):
          v, u = a_i, a_i
          val_arr = -(pa+dx*0.2*Da)*np.ones_like(nut_avail)
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #ds/(dndt)
        for a_i, s_i, r_i, cp_i, cs_i, xS in ([aa_i, sa_i, ra_i, cpa_i, csa_i, xSa],[ab_i, sb_i, rb_i, cpb_i, csb_i, xSb]):
          v, u = s_i, n_i
          val_arr = (xs * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hR, kR) * hillN(y[r_i,:,:], hC,
              kC) + xS*np.greater(y[cs_i,:,:],od)- rc * y[s_i,:,:] * np.greater(y[cp_i,:,:]+y[cs_i,:,:],od)) * dnut_avail
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

          #ds/(dadt)
          v, u = s_i, a_i
          val_arr = xs * np.greater(y[cp_i,:,:],od) * dhillda(y[a_i,:,:], hS, kS) * hillN(y[r_i,:,:], hC, kC) * nut_avail
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

          #ds/(dsdt)
          v, u = s_i, s_i
          val_arr =  -rc * nut_avail * np.greater(y[cp_i,:,:]+y[cs_i,:,:],od) - rS
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

          #ds/(drdt)
          v, u = s_i, r_i
          val_arr = -xs * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hS, kS) * dhillda(y[r_i,:,:], hC, kC) * nut_avail
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

          #dr/(dadt)
          v, u = r_i, a_i
          val_arr = xr * y[cp_i,:,:]* dhillda(y[a_i,:,:], hR, kR) * nut_avail * np.greater(y[cp_i,:,:],od)
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

          #dr/(drdt)
          v, u = r_i, r_i
          val_arr = -rc * nut_avail - rR
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

          #dr/(dndt)
          v, u = r_i, n_i
          val_arr = (xr * hill(y[a_i,:,:], hR, kR) - rc * y[r_i,:,:]) * dnut_avail * np.greater(y[cp_i,:,:],od)
          i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #do/(daadt)
#        v, u = o_i, aa_i
#        val_arr = xr * y[cr_i,:,:]* dhillda(y[aa_i,:,:], hR, kR) * hill(y[ab_i,:,:], hR, kR) * nut_avail * np.greater(y[cr_i,:,:],od)
#        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)
#
#        #do/(dabdt)
#        v, u = o_i, ab_i
#        val_arr = xr * y[cr_i,:,:]* dhillda(y[ab_i,:,:], hR, kR) * hill(y[aa_i,:,:], hR, kR) * nut_avail * np.greater(y[cr_i,:,:],od)
#        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)
#
#        #do/(dodt)
#        v, u = o_i, o_i
#        val_arr = -rc * nut_avail - rR
#        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)
#
#        #do/(dndt)
#        v, u = o_i, n_i
#        val_arr = (xr * hill(y[aa_i,:,:], hR, kR) * hill(y[ab_i,:,:], hR, kR)  - rc * y[o_i,:,:]) * dnut_avail * np.greater(y[cr_i,:,:],od)
#        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

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

    def get_dif_jac(self, t, y):
        species, n_h, n_w, dx = self.dims
        # y.shape = (species,n_h,n_w)
        n_jac = species*n_h*n_w
        self.calc_dif_jac(t,y)
        # y.shape = species*n_h*n_w
        return sparse.coo_matrix((self.dif_vec, (self.j1_dif,self.j2_dif)),shape=(n_jac, n_jac),dtype=np.float64)

class Simulator(object):
    '''
    Instances of this class are initialized with information requried to simulate an experimental pad and compare to data.
    '''
    def __init__(self, scale=4, n_a=50, n_b=50, n_as=2, n_bs=1):
        self.basedims = np.array([3,3])
        self.set_scale(scale)
        self.t_eval = np.linspace(0,11*60*60,200)
        ns, nh, nw = self.initial_array.shape
        self.initial_array[n_i,:,:] = 100
        for c_i, n_spots in [(cpa_i, n_a), (cpb_i,n_b), (csa_i, n_as), (csb_i,n_bs)]:
          for i in range(n_spots):
            rad = np.random.beta(1,1.5,size=(1,))*0.7*self.scale
            theta = np.random.uniform(0,2*np.pi,size=(1,))
            x = np.int(nw/2 + np.cos(theta)*rad)
            y = np.int(nh/2 + np.sin(theta)*rad)
            self.initial_array[c_i,y,x] = scale*1e-3
        self.initial_array = skimage.filters.gaussian(self.initial_array,(0,scale*1e-2,scale*1e-2),preserve_range=True)

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
        atol[n_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[cr_i,:,:] = 1e-3*np.ones((nh, nw), dtype=np.float64)
        #atol[o_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[csa_i,:,:] = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[cpa_i,:,:] = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[aa_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[sa_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[ra_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[csb_i,:,:] = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[cpb_i,:,:] = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[ab_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[sb_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
        atol[rb_i,:,:]  = 1e-3*np.ones((nh, nw), dtype=np.float64)
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
