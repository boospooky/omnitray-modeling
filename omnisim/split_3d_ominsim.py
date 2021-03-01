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

@numba.jit('void(float64[:,:,:,:],float64[:,:,:,:])',nopython=True, cache=True)
def laplace_op_3d_noflux_boundaries(A, D):
    # Middle
    D[:,1:-1,1:-1,1:-1] = A[:,2:,1:-1,1:-1] + A[:,:-2,1:-1,1:-1] + \
                          A[:,1:-1,1:-1, 2:] + A[:,1:-1,1:-1, :-2] +\
                          A[:,1:-1,:-2, 1:-1] + A[:,1:-1,2:, 1:-1] - \
                          6*A[:,1:-1,1:-1, 1:-1]
    # Faces
    D[:,0,1:-1,1:-1] = A[:,1,1:-1,1:-1] + \
                       A[:,0,2:,1:-1] + A[:,0,:-2,1:-1] + A[:,0,1:-1,2:] + A[:,0,1:-1,:-2] + \
                       5*A[:,0,1:-1, 1:-1]
    D[:,-1,1:-1,1:-1] = A[:,-2,1:-1,1:-1] + \
                       A[:,-1,2:,1:-1] + A[:,-1,:-2,1:-1] + A[:,-1,1:-1,2:] + A[:,-1,1:-1,:-2] + \
                       5*A[:,-1,1:-1, 1:-1]
    D[:,1:-1,0,1:-1] = A[:,1:-1,1,1:-1] + \
                       A[:,2:,0,1:-1] + A[:,:-2,0,1:-1] + A[:,1:-1,0,2:] + A[:,1:-1,0,:-2] + \
                       5*A[:,1:-1,0, 1:-1]
    D[:,1:-1,-1,1:-1] = A[:,1:-1,-2,1:-1] + \
                       A[:,2:,-1,1:-1] + A[:,:-2,-1,1:-1] + A[:,1:-1,-1,2:] + A[:,1:-1,-1,:-2] + \
                       5*A[:,1:-1,-1, 1:-1]
    D[:,1:-1,1:-1,0] = A[:,1:-1,1:-1,1] + \
                       A[:,2:,1:-1,0] + A[:,:-2,1:-1,0] + A[:,1:-1,2:,0] + A[:,1:-1,:-2,0] + \
                       5*A[:,1:-1, 1:-1,0]
    D[:,1:-1,1:-1,-1] = A[:,1:-1,1:-1,-2] + \
                       A[:,2:,1:-1,-1] + A[:,:-2,1:-1,-1] + A[:,1:-1,2:,-1] + A[:,1:-1,:-2,-1] + \
                       5*A[:,1:-1, 1:-1,-1]
    # Edges
    D[:,0,0,1:-1] = A[:,0,0,2:] + A[:,0,0,:-2] + A[:,1,0, 1:-1] + A[:,0,1, 1:-1] - 4*A[:,0, 1:-1]
    D[:,-1,-1,1:-1] = A[:,-1,-1,2:] + A[:,-1,-1,:-2] + A[:,1,0, 1:-1] + A[:,0,1, 1:-1] - 4*A[:,0, 1:-1]
    D[:,0,-1,1:-1] = A[:,0,-1,2:] + A[:,0,-1,:-2] + A[:,1,0, 1:-1] + A[:,0,1, 1:-1] - 4*A[:,0, 1:-1]
    D[:,-1,0,1:-1] = A[:,-1,0,2:] + A[:,-1,0,:-2] + A[:,1,0, 1:-1] + A[:,0,1, 1:-1] - 4*A[:,0, 1:-1]

    D[:,0,1:-1,0] = A[:,0,2:,0] + A[:,0,:-2,0] + A[:,0,1:-1,1] + A[:,1,1:-1,0] - 4*A[:,0,1:-1,0]
    D[:,-1,1:-1,-1] = A[:,-1,2:,-1] + A[:,-1,:-2,-1] + A[:,1,1:-1,0] + A[:,0,1:-1,1] - 4*A[:,-1,1:-1,-1]
    D[:,0,1:-1,-1] = A[:,0,2:,-1] + A[:,0,:-2,-1] + A[:,1,1:-1,0] + A[:,0,1:-1,1] - 4*A[:,0,1:-1,-1]
    D[:,-1,1:-1,0] = A[:,-1,2:,0] + A[:,-1,:-2,0] + A[:,1,1:-1,0] + A[:,0,1:-1,1] - 4*A[:,-1,1:-1,0]

    D[:,1:-1,0,0] = A[:,2:,0,0] + A[:,:-2,0,0] + A[:,1:-1,0,1] + A[:,1:-1,1,0] - 4*A[:,1:-1,0,0]
    D[:,1:-1,-1,-1] = A[:,2:,-1,-1] + A[:,:-2,-1,-1] + A[:,1:-1,1,0] + A[:,1:-1,0,1] - 4*A[:,1:-1,-1,-1]
    D[:,1:-1,0,-1] = A[:,2:,0,-1] + A[:,:-2,0,-1] + A[:,1:-1,1,0] + A[:,1:-1,0,1] - 4*A[:,1:-1,0,-1]
    D[:,1:-1,-1,0] = A[:,2:,-1,0] + A[:,:-2,-1,0] + A[:,1:-1,1,0] + A[:,1:-1,0,1] - 4*A[:,1:-1,-1,0]
    # Corners
    D[:,0,0,0] = A[:,0,0,1] + A[:,0,1,0] + A[:,1,0,0] - 3*A[:,0,0,0]
    D[:,-1,0,0] = A[:,-1,0,1] + A[:,-1,1,0] + A[:,-2,0,0] - 3*A[:,-1,0,0]
    D[:,0,-1,0] = A[:,0,-1,1] + A[:,0,-2,0] + A[:,1,-1,0] - 3*A[:,0,-1,0]
    D[:,-1,-1,0] = A[:,-1,-1,1] + A[:,-1,-2,0] + A[:,-2,-1,0] - 3*A[:,-1,-1,0]
    D[:,0,0,-1] = A[:,0,0,-2] + A[:,0,1,-1] + A[:,1,0,-1] - 3*A[:,0,0,-1]
    D[:,0,-1,-1] = A[:,0,-1,-2] + A[:,0,-2,-1] + A[:,1,-1,-1] - 3*A[:,0,-1,-1]
    D[:,-1,-1,-1] = A[:,-1,-1,-2] + A[:,-1,-2,-1] + A[:,-2,-1,-1] - 3*A[:,-1,-1,-1]

@numba.jit('void(float64[:,:,:],float64[:,:,:])',nopython=True, cache=True)
def laplace_op_3d_noflux_boundaries_onespec(A, D):
    # Middle
    D[1:-1,1:-1,1:-1] = A[2:,1:-1,1:-1] + A[:-2,1:-1,1:-1] + \
                          A[1:-1,1:-1, 2:] + A[1:-1,1:-1, :-2] +\
                          A[1:-1,:-2, 1:-1] + A[1:-1,2:, 1:-1] - \
                          6*A[1:-1,1:-1, 1:-1]
    # Faces
    D[0,1:-1,1:-1] = A[1,1:-1,1:-1] + \
                       A[0,2:,1:-1] + A[0,:-2,1:-1] + A[0,1:-1,2:] + A[0,1:-1,:-2] - \
                       5*A[0,1:-1, 1:-1]
    D[-1,1:-1,1:-1] = A[-2,1:-1,1:-1] + \
                       A[-1,2:,1:-1] + A[-1,:-2,1:-1] + A[-1,1:-1,2:] + A[-1,1:-1,:-2] - \
                       5*A[-1,1:-1, 1:-1]
    D[1:-1,0,1:-1] = A[1:-1,1,1:-1] + \
                       A[2:,0,1:-1] + A[:-2,0,1:-1] + A[1:-1,0,2:] + A[1:-1,0,:-2] - \
                       5*A[1:-1,0, 1:-1]
    D[1:-1,-1,1:-1] = A[1:-1,-2,1:-1] + \
                       A[2:,-1,1:-1] + A[:-2,-1,1:-1] + A[1:-1,-1,2:] + A[1:-1,-1,:-2] - \
                       5*A[1:-1,-1, 1:-1]
    D[1:-1,1:-1,0] = A[1:-1,1:-1,1] + \
                       A[2:,1:-1,0] + A[:-2,1:-1,0] + A[1:-1,2:,0] + A[1:-1,:-2,0] - \
                       5*A[1:-1, 1:-1,0]
    D[1:-1,1:-1,-1] = A[1:-1,1:-1,-2] + \
                       A[2:,1:-1,-1] + A[:-2,1:-1,-1] + A[1:-1,2:,-1] + A[1:-1,:-2,-1] - \
                       5*A[1:-1, 1:-1,-1]
    # Edges
    D[0,0,1:-1] = A[0,0,2:] + A[0,0,:-2] + A[1,0, 1:-1] + A[0,1, 1:-1] - 4*A[0,0,1:-1]
    D[-1,-1,1:-1] = A[-1,-1,2:] + A[-1,-1,:-2] + A[-1,-2,1:-1] + A[-2,-1,1:-1] - 4*A[-1,-1,1:-1]
    D[0,-1,1:-1] = A[0,-1,2:] + A[0,-1,:-2] + A[1,-1,1:-1] + A[0,-2,1:-1] - 4*A[0,-1,1:-1]
    D[-1,0,1:-1] = A[-1,0,2:] + A[-1,0,:-2] + A[-2,0,1:-1] + A[-1,1,1:-1] - 4*A[-1,0,1:-1]

    D[0,1:-1,0] = A[0,2:,0] + A[0,:-2,0] + A[0,1:-1,1] + A[1,1:-1,0] - 4*A[0,1:-1,0]
    D[-1,1:-1,-1] = A[-1,2:,-1] + A[-1,:-2,-1] + A[-2,1:-1,-1] + A[-1,1:-1,-2] - 4*A[-1,1:-1,-1]
    D[0,1:-1,-1] = A[0,2:,-1] + A[0,:-2,-1] + A[1,1:-1,-1] + A[0,1:-1,-2] - 4*A[0,1:-1,-1]
    D[-1,1:-1,0] = A[-1,2:,0] + A[-1,:-2,0] + A[-2,1:-1,0] + A[-1,1:-1,1] - 4*A[-1,1:-1,0]

    D[1:-1,0,0] = A[2:,0,0] + A[:-2,0,0] + A[1:-1,0,1] + A[1:-1,1,0] - 4*A[1:-1,0,0]
    D[1:-1,-1,-1] = A[2:,-1,-1] + A[:-2,-1,-1] + A[1:-1,-2,-1] + A[1:-1,-1,-2] - 4*A[1:-1,-1,-1]
    D[1:-1,0,-1] = A[2:,0,-1] + A[:-2,0,-1] + A[1:-1,1,-1] + A[1:-1,0,-2] - 4*A[1:-1,0,-1]
    D[1:-1,-1,0] = A[2:,-1,0] + A[:-2,-1,0] + A[1:-1,-2,0] + A[1:-1,-1,1] - 4*A[1:-1,-1,0]
    # Corners
    D[0,0,0] = A[0,0,1] + A[0,1,0] + A[1,0,0] - 3*A[0,0,0]
    D[-1,0,0] = A[-1,0,1] + A[-1,1,0] + A[-2,0,0] - 3*A[-1,0,0]
    D[0,-1,0] = A[0,-1,1] + A[0,-2,0] + A[1,-1,0] - 3*A[0,-1,0]
    D[-1,-1,0] = A[-1,-1,1] + A[-1,-2,0] + A[-2,-1,0] - 3*A[-1,-1,0]
    D[0,0,-1] = A[0,0,-2] + A[0,1,-1] + A[1,0,-1] - 3*A[0,0,-1]
    D[0,-1,-1] = A[0,-1,-2] + A[0,-2,-1] + A[1,-1,-1] - 3*A[0,-1,-1]
    D[-1,-1,-1] = A[-1,-1,-2] + A[-1,-2,-1] + A[-2,-1,-1] - 3*A[-1,-1,-1]

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

# @numba.jit('void(float64[:,:,:],float64[:,:,:,:],float64[:])',nopython=True, cache=True)
# def calc_diffusion(y, diff_terms, p0):
#     dx, Dc,  rc, rS, rR,    Hn, Kn, Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = p0
#     laplace_op_3d_noflux_boundaries_onespec(Dn*dx*y[n_i,:,:,:], diff_terms[n_i,:,:,:])
#     laplace_op_3d_noflux_boundaries_onespec(Da*dx*y[a_i,:,:,:], diff_terms[a_i,:,:,:])
#     diff_terms[cp_i,:,:] = 0
#     diff_terms[cs_i,:,:] = 0
#     diff_terms[s_i,:,:] = 0
#     diff_terms[r_i,:,:] = 0

@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:],float64[:])',cache=True,nopython=True)
def calc_rxn(y, d_y, nut_avail, p0):
    dx, Dc,  rc, rS, rR,    Hn, Kn, Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = p0
    # Growth term
    nut_avail[:] = hill(y[n_i,:,:], Hn, Kn)
    scale = np.sqrt(dx)

    # Cell growth and diffusion
    for ind in cell_inds:
        d_y[ind,:,:] = rc * nut_avail * y[ind,:,:]

    # Nutrient consumption
    d_y[n_i,:,:] =  -scale*kn * nut_avail * (y[cp_i,:,:]+y[cs_i,:,:])

    # AHL production
    d_y[a_i,:,:] =  scale*xa * y[s_i,:,:]*(y[cp_i,:,:]+y[cs_i,:,:]) # - pa * y[a_i,:,:]

    # Synthase production
    d_y[s_i,:,:] = ( xs * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hS, kS) * hillN(y[r_i,:,:], hC, kC) + xS * np.greater(y[cs_i,:,:],od) - rc * y[s_i,:,:]) * nut_avail - rS * y[s_i,:,:]

    # Repressor production
    d_y[r_i,:,:] = ( xr  * hill(y[a_i,:,:], hR, kR) - rc * y[r_i,:,:]) * nut_avail * np.greater(y[cp_i,:,:],od) - rR * y[r_i,:,:]

# @numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:])',cache=True,nopython=True)
# def calc_f(y, d_y, diff_terms, nut_avail, p0):
#     calc_diffusion(y, diff_terms, p0)
#     calc_rxn(y, d_y, nut_avail, p0)
#     d_y[:] = d_y + diff_terms

# def f_ivp(t, y, d_y, diff_terms, nut_avail, p0, dims, calc_f):
#     y.shape = dims
#     calc_f(y, d_y, diff_terms, nut_avail, p0)
#     y.shape = np.prod(dims)
#     return d_y.flatten()

class Jacobian(object):
    '''
    The Jacobian used is a sparse matrix. On initialization, this class deterines the matrix structure and later can be populated with parameter values
    '''
    def f_ji(self, x, y, z, spec):
        _, _, n_h, n_w, _ = self.dims
        return x + n_w*y + n_h*n_w*z + self.ji_offsets[spec]

    def __init__(self, dims):
        species, n_z, n_h, n_w, dx = dims
        ji_offsets = np.ones(species)*n_h*n_w
        ji_offsets[a_i] *= n_z
        ji_offsets = np.cumsum(ji_offsets).astype(np.int)
        ji_offsets[1:] = ji_offsets[:-1]
        ji_offsets[0] = 0
        self.ji_offsets = ji_offsets
        self.dims = dims
        self.dx = dx
        # Make jacobian array
        n_diff_jac = n_z*n_h*n_w
        # jacobian terms:
        # diffusion : 7 per x,y,z point, minus faces, for only nutrient and ahl
        n_nz = n_diff_jac*7 - (n_h*n_z+n_w*n_z+n_h*n_w)*2 # + 4*n_h*n_w
        self.dif_vec = np.empty(n_nz,dtype=np.float64)
        self.j1_dif = np.empty(n_nz,dtype=np.int)
        self.j2_dif = np.empty(n_nz,dtype=np.int)

        offsets = [(x,y,z) for x in [1,0,-1] for y in [1,0,-1] for z in [1,0,-1] if np.abs(z)+np.abs(x)+np.abs(y)==1]
        neigh_diff_indices = [[(self.f_ji(x,y,z,spec), self.f_ji(x+offx,y+offy,z+offz,spec))
                                   for offz, offy, offx in offsets
                                    for x in np.arange(max([0,-offx]), n_w+min([0,-offx]))
                                    for z in np.arange(max([0,-offz]), n_z+min([0,-offz]))
                                     for y in np.arange(max([0,-offy]), n_h+min([0,-offy]))]
                                          for spec in [a_i]]

        center_diff_indices = [[(self.f_ji(x,y,z,spec), self.f_ji(x,y,z,spec))
                                for x in np.arange(1,n_w-1)
                                for z in np.arange(1,n_z-1)
                                 for y in np.arange(1,n_h-1)]
                                          for spec in [a_i]]

        face_diff_indices = [[(self.f_ji(x,y,z,spec), self.f_ji(x,y,z,spec))
                                for x in np.arange(1,n_w-1)
                                for z in np.arange(1,n_z-1)
                                 for y in [0, n_h-1]] +
                              [(self.f_ji(x,y,z,spec), self.f_ji(x,y,z,spec))
                                for x in [0,n_w-1]
                                for z in np.arange(1,n_z-1)
                                 for y in np.arange(1,n_h-1)]+
                              [(self.f_ji(x,y,z,spec), self.f_ji(x,y,z,spec))
                                for x in np.arange(1,n_w-1)
                                for z in [0,n_z-1]
                                 for y in np.arange(1,n_h-1)]
                                          for spec in [a_i]]

        edge_diff_indices = [[(self.f_ji(x,y,z,spec), self.f_ji(x,y,z,spec))
                                for x in np.arange(1,n_w-1)
                                for z in [0,n_z-1]
                                 for y in [0, n_h-1]] +
                              [(self.f_ji(x,y,z,spec), self.f_ji(x,y,z,spec))
                                for x in [0,n_w-1]
                                for z in np.arange(1,n_z-1)
                                 for y in [0,n_h-1]]+
                              [(self.f_ji(x,y,z,spec), self.f_ji(x,y,z,spec))
                                for x in [0,n_w-1]
                                for z in [0,n_z-1]
                                 for y in np.arange(1,n_h-1)]
                                          for spec in [a_i]]

        corner_diff_indices = [[(self.f_ji(x,y,z,spec), self.f_ji(x,y,z,spec))
                                for x in [0,n_w-1]
                                for z in [0,n_z-1]
                                 for y in [0, n_h-1]]
                                          for spec in [a_i]]

        self.dif_indices_list = [neigh_diff_indices, center_diff_indices, face_diff_indices, edge_diff_indices, corner_diff_indices]

        self.rxn_indices_dict = {}
        def rxn_index_helper(v, u):
            # returns indices for dv/du
            return [(x, y, self.f_ji(x,y,0,v), self.f_ji(x,y,0,u)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)
        dict_keys = []
        # cs nonzero partials: n, cs
        v_vec = np.arange(species)
        u_vec_list = [[n_i, cs_i],# cs_i
                     [n_i, cp_i],# cp_i
                     [n_i, cs_i, cp_i],# n_i
                     [s_i, cs_i, cp_i], #, a_i],# a_i
                     [s_i, n_i, a_i, r_i],# s_i
                     [a_i, r_i, n_i]]# r_i
        for v, u_vec in zip(v_vec, u_vec_list):
            for u in u_vec:
                self.rxn_indices_dict[(v,u)] = rxn_index_helper(v,u)
        n_terms = np.sum([len(xx) for xx in self.rxn_indices_dict.values()])
        self.rxn_vec = np.zeros(n_terms, dtype=np.float64)
        self.j1_rxn = np.zeros(n_terms, dtype=np.int)
        self.j2_rxn = np.zeros(n_terms, dtype=np.int)

    def set_p0(self, p0):
        self.p0 = p0
        Dc,  rc, rS, rR,    Hn, Kn, Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = p0
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
        Dc,  rc, rS, rR,    Hn, Kn, Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = self.p0
        D_vec = np.array(self.D_vec)
        # dx = self.dx
        species,n_z, n_h, n_w, dx = self.dims
        # cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)
        dif_species = [a_i]
        i = 0
        neigh_diff_indices, center_diff_indices, face_diff_indices,  edge_diff_indices, corner_diff_indices = self.dif_indices_list
        val_arr = D_vec[dif_species]*dx
        for val, ind_list in zip(val_arr, neigh_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

        val_arr = D_vec[dif_species]*(-6*dx) - pa
        for val, ind_list in zip(val_arr, center_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

        val_arr = D_vec[dif_species]*(-5*dx)
        for val, ind_list in zip(val_arr, face_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

        val_arr = D_vec[dif_species]*(-4*dx)
        for val, ind_list in zip(val_arr, edge_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

        val_arr = D_vec[dif_species]*(-3*dx)
        for val, ind_list in zip(val_arr, corner_diff_indices):
            i = self.assign_dif_vals(val, ind_list, i)

    def calc_rxn_jac(self, t, y):
        Dc,  rc, rS, rR,    Hn, Kn, Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = self.p0
        species,n_z, n_h, n_w, dx = self.dims
#         dcdcdt_indices, dcdndt_indices, dndndt_indices, dndcdt_indices = self.rxn_indices_list
        cs_i, cp_i, n_i, a_i, s_i, r_i = np.arange(species)
        scale = np.sqrt(dx)

        i = 0
        nut_avail = hill(y[n_i,:,:], Hn, Kn)
        dnut_avail = dhillda(y[n_i,:,:], Hn, Kn)
        cell_inds = [cs_i, cp_i]

        #dc/(dcdt)
        v, u = cs_i, cs_i
        val_arr = rc*nut_avail*np.greater(y[cs_i,:,:],od)
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        v, u = cp_i, cp_i
        val_arr = rc*nut_avail*np.greater(y[cp_i,:,:],od)
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dc/(dndt)
        v, u = cs_i, n_i
        val_arr = rc*dnut_avail*y[cs_i,:,:]*np.greater(y[cs_i,:,:],od)
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        v, u = cp_i, n_i
        val_arr = rc*dnut_avail*y[cp_i,:,:]*np.greater(y[cp_i,:,:],od)
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dn/(dcdt)
        v, u = n_i, cp_i
        val_arr = -scale*kn*nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        v, u = n_i, cs_i
        val_arr = -scale*kn*nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)
        
        #dn/(dndt)
        v, u = n_i, n_i
        val_arr = -scale*kn*dnut_avail*(y[cp_i,:,:]+y[cs_i,:,:])
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dsdt)
        v, u = a_i, s_i
        val_arr = scale*xa*y[cell_inds,:,:].sum(axis=0)
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dcsdt)
        v, u = a_i, cs_i
        val_arr = scale*xa*y[s_i,:,:]
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #da/(dcpdt)
        v, u = a_i, cp_i
        val_arr = scale*xa*y[s_i,:,:]
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

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
        val_arr =  -rc * nut_avail - rS
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #ds/(drdt)
        v, u = s_i, r_i
        val_arr = -xs * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hS, kS) * dhillda(y[r_i,:,:], hC, kC) * nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dr/(dadt)
        v, u = r_i, a_i
        val_arr = xr * y[cp_i,:,:]* dhillda(y[a_i,:,:], hR, kR) * nut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dr/(drdt)
        v, u = r_i, r_i
        val_arr = -rc * nut_avail - rR
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

        #dr/(dndt)
        v, u = r_i, n_i
        val_arr = (xr * np.greater(y[cp_i,:,:],od) * hill(y[a_i,:,:], hR, kR) - rc * y[r_i,:,:]) * dnut_avail
        i = self.assign_rxn_vals(self.rxn_indices_dict[(v,u)], val_arr,i)

    def calc_jac_wrapper(self, t, y):
        species, n_z, n_h, n_w, dx = self.dims
        arr_z = n_z + 5
        y.shape = (arr_z,n_h,n_w)
        n_jac = arr_z*n_h*n_w
        self.calc_rxn_jac(t,y)
        data_vec = np.concatenate([self.dif_vec, self.rxn_vec])
        j1_vec = np.concatenate([self.j1_dif, self.j1_rxn])
        j2_vec = np.concatenate([self.j2_dif, self.j2_rxn])
        y.shape = arr_z*n_h*n_w
        return sparse.coo_matrix((data_vec, (j1_vec,j2_vec)),shape=(n_jac, n_jac),dtype=np.float64)

    def get_rxn_jac(self, t, y):
        species, n_z, n_h, n_w, dx = self.dims
        arr_z = n_z + 5
        y.shape = (arr_z,n_h,n_w)
        n_jac = arr_z*n_h*n_w
        self.calc_rxn_jac(t,y)
        y.shape = arr_z*n_h*n_w
        return sparse.coo_matrix((self.rxn_vec, (self.j1_rxn,self.j2_rxn)),shape=(n_jac, n_jac),dtype=np.float64)

    def get_dif_jac(self):
        species, n_z, n_h, n_w, dx = self.dims
        arr_z = n_z + 5
        n_jac = arr_z*n_h*n_w
        return sparse.coo_matrix((self.dif_vec, (self.j1_dif,self.j2_dif)),shape=(n_jac, n_jac),dtype=np.float64)

class Simulator(object):
    '''
    Instances of this class are initialized with information requried to simulate an experimental pad and compare to data.
    '''
    def __init__(self, scale=8):
        self.basedims = np.array([2,4,12])
        self.set_scale(scale)
        self.t_eval = np.linspace(0,24*60,200)

    def set_scale(self,scale):
        logscale = np.log2(scale)
        if not np.isclose(logscale, np.round(logscale)):
            print('rounding scale to nearest power of 2')
            scale = np.int(np.power(2,np.round(logscale)))
        self.scale = scale
        self.dx = np.power(scale,2)
        nz, nh, nw = scale*self.basedims
        species = 6
        arr_z = 5 + nz
        self.arr_z = arr_z
        self.dims = [species,nz,nh,nw,self.dx]
        self.yshape = [arr_z,nh,nw]
        # Setup slices. to save memory, you're putting all species and z-stacks along the same ndarray axis
        i = 0
        cs_slice = np.s_[i:i+1,:,:]
        i += 1
        cp_slice = np.s_[i:i+1,:,:]
        i += 1
        n_slice = np.s_[i:i+1,:,:]
        i += 1
        a_slice = np.s_[i:i+nz,:,:]
        i += nz
        s_slice = np.s_[i:i+1,:,:]
        i += 1
        r_slice = np.s_[i:i+1,:,:]
        self.species_slices = (cs_slice, cp_slice, n_slice, a_slice, s_slice, r_slice)
        z0_slice = np.s_[[x_slice[0].start for x_slice in self.species_slices],:,:]
        self.z0_slice = z0_slice

        # setup arrays
        self.initial_array = np.zeros((arr_z, nh, nw))
        atol = np.zeros((arr_z, nh, nw), dtype=np.float64,order='C')# + 1e-7
        atol[cs_slice] = 1e-3*np.ones((1, nh, nw), dtype=np.float64)
        atol[cp_slice] = 1e-3*np.ones((1, nh, nw), dtype=np.float64)
        atol[n_slice]  = 1e-2*np.ones((1, nh, nw), dtype=np.float64)
        atol[a_slice]  = 1e1*np.ones((nz, nh, nw), dtype=np.float64)
        atol[s_slice]  = 1e1*np.ones((1, nh, nw), dtype=np.float64)
        atol[r_slice]  = 1e1*np.ones((1, nh, nw), dtype=np.float64)
        atol.shape = arr_z*nh*nw
        self.atol = atol
        self.rtol = np.float64(1e-3)
        self.scale = scale
        self.jacobian = Jacobian(self.dims)

    def f_ivp_wrapper(self, t, y):
        d_y, diff_terms, nut_avail, p0, dims, _ = self.args
        y.shape = dims
        self.f_rxn_wrapper(t, y)
        self.f_dif_wrapper(t, y)
        diff_terms[self.z0_slice] = d_y + diff_terms[self.z0_slice]
        y.shape = np.prod(dims)
        return diff_terms.copy().flatten()

    def f_rxn_wrapper(self, t, y):
        y.shape = self.yshape
        calc_rxn(y[self.z0_slice], self.d_y, self.nut_avail, self.params)
        y.shape = np.prod(self.yshape)
        return self.d_y.copy().flatten()

    def f_dif_wrapper(self, t, y):
        y.shape = self.yshape
        dx, Dc,  rc, rS, rR,    Hn, Kn, Dn,   kn, Da, xa, xs, xS, xr, hS, kS, hR, kR, hC, kC, pa, leak, od = self.params
        d_y, diff_terms, nut_avail, p0, dims, _ = self.args
        n_slice, a_slice = self.species_slices[n_i], self.species_slices[a_i]
        #laplace_op_3d_noflux_boundaries_onespec(Dn*dx*y[n_slice], diff_terms[n_slice])
        laplace_op_3d_noflux_boundaries_onespec(Da*dx*y[a_slice], diff_terms[a_slice])
        diff_terms[a_slice] -= pa * y[a_slice]
        y.shape = np.prod(self.yshape)
        return diff_terms.copy().flatten()

    def set_p0(self, p0):
        species, n_z, n_h, n_w, dx = self.dims
        self.p0 = p0.astype(np.float64)
        self.params = np.ones(len(p0)+1, dtype=np.float64, order='C')
        self.params[1:] = p0
        self.params[0] = self.dx
        self.jacobian.set_p0(self.p0)
        self.args=(np.zeros((species, n_h, n_w), dtype=np.float64,order='C'), # d_y
                   np.zeros(self.yshape, dtype=np.float64,order='C'), # diff_terms
                   np.zeros((n_h, n_w), dtype=np.float64,order='C'), # nut_avail
                   self.params, self.yshape, self.f_ivp_wrapper) #p0, yshape, calc_f
        self.d_y, self.diff_terms, self.nut_avail, _, _, _ = self.args

    def sim(self, p0=None, method='RK45'):
        self.set_p0(p0)
        species, n_z, n_h, n_w, dx = self.dims
        self.initial_array.shape = np.prod(self.yshape)
        out = itg.solve_ivp(self.f_ivp_wrapper,
                            [self.t_eval.min(), self.t_eval.max()],
                            self.initial_array.copy().astype(np.float64).flatten(),
                            vectorized=True,
                            method=method,
                            dense_output=True,
                            atol=self.atol,
                            rtol=self.rtol,
                            t_eval=self.t_eval,
                            jac=self.jacobian.calc_jac_wrapper)
        self.out = out
        exp_t = out.t
        exp_y = out.y.T
        exp_y.shape = (len(exp_t), self.arr_z, n_h, n_w)
        self.initial_array.shape = self.yshape
        self.sim_arr, self.sim_tvc = exp_y, exp_t
