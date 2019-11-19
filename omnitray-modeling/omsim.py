import itertools as itt

import numpy as np
import pandas as pd
import os
import sys
import string
import scipy.integrate as itg
import scipy.sparse as sparse

import numba
import gc

from multiprocessing import Pool, Process

#@numba.jit('void(float32[:,:,:],float32[:,:,:])', nopython=True, cache=True)
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

#@numba.jit('float32[:,:](float32[:,:],float32,float32)',nopython=True, cache=True)
@numba.jit(nopython=True, cache=True)
def hill(a, n, k):
    h_ma = 1 - (1 / (1 + (a/k)**n))
    return h_ma

@numba.jit(nopython=True, cache=True)
def dhillda(a, n, k):
    h_ma = (n/k)*((a/k)**(n-1))*(1 / (1 + (a/k)**n)**2)
    return h_ma

#@numba.jit('float32[:,:](float32[:,:],float32,float32)',nopython=True, cache=True)
@numba.jit(nopython=True, cache=True)
def hillN(a, n, k):
    return 1 / (1 + (a/k)**n)

# @numba.jit(cache=True)
def f_ivp(t, y, d_y, diff_terms, nut_avail, p0, dims, calc_f):
    species, n_h, n_w, scale = dims
    y.shape = (species, n_h, n_w)
    calc_f(y, d_y, diff_terms, nut_avail, p0)
    y.shape = species*n_h*n_w
    return d_y.flatten()

def wrapper(dims, p0, initial_array, tmax, atol, rtol, calc_f, jac):
    species, n_h, n_w, scale = dims
    args=(np.zeros((species, n_h, n_w), dtype=np.float32,order='C'), 
          np.zeros((species, n_h, n_w), dtype=np.float32,order='C'), 
          np.zeros((n_h, n_w), dtype=np.float32,order='C'), 
          p0, dims, calc_f)
    initial_array.shape = n_h*n_w*species
    f_lambda = lambda t, y : f_ivp(t, y, *args)
    out = itg.solve_ivp(f_lambda, [0, tmax], initial_array.copy(), 
                        vectorized=True, method="BDF", 
                        atol=atol, rtol=rtol, t_eval=np.arange(0,tmax,4), jac=jac)
    exp_t = out.t
    exp_y = out.y.T
    exp_y.shape = (len(exp_t), species, n_h, n_w)
    return exp_y, exp_t

# @numba.jit(cache=True)
def prep_pad_helper(scale, init_cells):
    scale_factor = (scale/4500)/(2.475/4)
    scaled_init = skimage.transform.rescale((init_cells>1).astype(np.float32), 
                                            scale_factor, 
                                            order=1, 
                                            mode='constant', 
                                            multichannel=False,
                                            cval=0)
    return scaled_init

def prep_jac(p0, dims):
    dx, Dc, Dn, rc, Kn, Hn, pn, xs, ps, leak, od0 = p0
    species, n_h, n_w, scale = dims
    # Make jacobian array
    #     print('make jacobian array')
    n_jac = n_h*n_w*species
    # jacobian terms:
    # diffusion : 5 per x,y point, minus 
    n_nz = n_jac*5 - 2*(n_h+n_w)*species # + 4*n_h*n_w
    data_vec = np.empty(n_nz,dtype=np.float32)
    j1_vec = np.empty(n_nz,dtype=np.int)
    j2_vec = np.empty(n_nz,dtype=np.int)
    f_ji = lambda x, y, spec : x + n_w*y + n_w*n_h*spec

    def assign_val(i, val, j1, j2):
        data_vec[i] = val
        j1_vec[i] = j1
        j2_vec[i] = j2
        return i+1
    
    i = 0
    # top diffusion
    for spec in np.arange(species):
        val = dx*D_vec[spec]
        for y in np.arange(1,n_h):
            for x in np.arange(n_w):
                j1 = f_ji(x, y, spec)
                j2 = f_ji(x, y-1, spec)
                i = assign_val(i, val, j1, j2)

    # right diffusion
    for spec in np.arange(species):
        val = dx*D_vec[spec]
        for y in np.arange(0,n_h):
            for x in np.arange(0,n_w-1):
                j1 = f_ji(x, y, spec)
                j2 = f_ji(x+1, y, spec)
                i = assign_val(i, val, j1, j2)

    # bottom diffusion
    for spec in np.arange(species):
        val = dx*D_vec[spec]
        for y in np.arange(0,n_h-1):
            for x in np.arange(n_w):
                j1 = f_ji(x, y, spec)
                j2 = f_ji(x, y+1, spec)
                i = assign_val(i, val, j1, j2)

    # left diffusion
    for spec in np.arange(species):
        val = dx*D_vec[spec]
        for y in np.arange(n_h):
            for x in np.arange(1,n_w):
                j1 = f_ji(x, y, spec)
                j2 = f_ji(x-1, y, spec)
                i = assign_val(i, val, j1, j2)

    # center diffusion, center
    for spec in np.arange(species):
        val = -4*dx*D_vec[spec]
        for y in np.arange(1,n_h-1):
            for x in np.arange(1,n_w-1):
                j1 = f_ji(x, y, spec)
                j2 = f_ji(x, y, spec)
                i = assign_val(i, val, j1, j2)

    # center diffusion, edges
    for spec in np.arange(species):
        val = -3*dx*D_vec[spec]
        for y in np.arange(1,n_h-1):
            for x in [0,n_w-1]:
                j1 = f_ji(x, y, spec)
                j2 = f_ji(x, y, spec)
                i = assign_val(i, val, j1, j2)
        for x in np.arange(1,n_w-1):
            for y in [0,n_h-1]:
                j1 = f_ji(x, y, spec)
                j2 = f_ji(x, y, spec)
                i = assign_val(i, val, j1, j2)

    # center diffusion, corners
    for spec in np.arange(species):
        val = -2*dx*D_vec[spec]
        for y in [0,n_h-1]:
            for x in [0,n_w-1]:
                j1 = f_ji(x, y, spec)
                j2 = f_ji(x, y, spec)
                i = assign_val(i, val, j1, j2)

    dif_jac = (data_vec, j1_vec, j2_vec)

    #dc/(dcdt)
    dcdcdt_indices = [(x, y, f_ji(x,y,c_i), f_ji(x,y,c_i)) \
            for x in np.arange(n_w) for y in np.arange(n_h)]

    #dc/(dndt)
    dcdndt_indices = [(x, y, f_ji(x,y,c_i), f_ji(x,y,n_i))  \
            for x in np.arange(n_w) for y in np.arange(n_h)]

    #dn/(dndt)
    dndndt_indices = [(x, y, f_ji(x,y,n_i), f_ji(x,y,n_i))  \
            for x in np.arange(n_w) for y in np.arange(n_h)]

    #dn/(dcdt)
    dndcdt_indices = [(x, y, f_ji(x,y,n_i), f_ji(x,y,c_i))  \
            for x in np.arange(n_w) for y in np.arange(n_h)]

    #ds/(dcdt)
    dsdcdt_indices = [(x, y, f_ji(x,y,s_i), f_ji(x,y,c_i))  \
            for x in np.arange(n_w) for y in np.arange(n_h)]

    #ds/(dndt)
    dsdndt_indices = [(x, y, f_ji(x,y,s_i), f_ji(x,y,n_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

    #ds/(dsdt)
    dsdsdt_indices = [(x, y, f_ji(x,y,s_i), f_ji(x,y,s_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

    indices_list = [dcdcdt_indices, dcdndt_indices, dndndt_indices, dndcdt_indices, dsdcdt_indices, dsdndt_indices, dsdsdt_indices]
    
    #     @numba.jit(cache=True,nopython=True)
    def calc_jac(t, y):
        n_terms = np.sum([len(xx) for xx in indices_list])
        data_vec = np.zeros(n_terms, dtype=np.float32)
        j1_vec = np.zeros(n_terms, dtype=np.int)
        j2_vec = np.zeros(n_terms, dtype=np.int)

        i = 0
        def assign_vals(indices, val_arr, i):
            for x1,y1,j1,j2 in indices:
                data_vec[i] = val_arr[y1,x1]
                j1_vec[i] = j1
                j2_vec[i] = j2
                return i+1

        #dc/(dcdt)
        nut_avail = hill(y[n_i,:,:], Hn, Kn)
        dnut_avail = dhillda(y[n_i,:,:], Hn, Kn)
        val_arr = rc*nut_avail
        i = assign_vals(dcdcdt_indices, val_arr,i)

        #dc/(dndt)
        val_arr = rc*dnut_avail*y[c_i,:,:]
        i = assign_vals(dcdcdt_indices, val_arr,i)

        #dn/(dndt)
        val_arr = -pn*dnut_avail*y[c_i,:,:]
        i = assign_vals(dcdcdt_indices, val_arr,i)

        #dn/(dcdt)
        val_arr = -pn*nut_avail
        i = assign_vals(dcdcdt_indices, val_arr,i)

        #ds/(dcdt)
        val_arr = xs*(y[c_i,:,:]>col_thresh)
        i = assign_vals(dcdcdt_indices, val_arr,i)

        #ds/(dndt)
        val_arr = -rc*y[s_i,:,:]*dnut_avail
        i = assign_vals(dcdcdt_indices, val_arr,i)

        #ds/(dsdt)
        for x1,y1,j1,j2 in dsdsdt_indices:
    #         rxn_jac[j1,j2] = -ps
            data_vec[i] = -ps
            j1_vec[i] = j1
            j2_vec[i] = j2
            i += 1

        return data_vec, j1_vec, j2_vec

    #     @numba.jit(cache=True)
    def calc_jac_wrapper(t,y):
        dif_vec, j1_dif, j2_dif = dif_jac
        y.shape = (species,n_h,n_w)
        n_jac = species*n_h*n_w
        rxn_vec, j1_rxn, j2_rxn = calc_jac(t,y)
        data_vec = np.concatenate([dif_vec, rxn_vec])
        j1_vec = np.concatenate([j1_dif, j1_rxn])
        j2_vec = np.concatenate([j2_dif, j2_rxn])
        y.shape = species*n_w*n_h
        return sparse.coo_matrix((data_vec, (j1_vec,j2_vec)),dtype=np.float32)

    return calc_jac_wrapper
    
def prep_initial_array(od0, species, cell_init):
    
    n_h, n_w = cell_init.shape
    # Set initial conditions
    initial_array = np.zeros((species, n_h, n_w), dtype=np.float32, order='C')# + 1e-7
    initial_array[n_i,:,:] = 100*np.ones((n_h, n_w), dtype=np.float32)
    initial_array[c_i,:,:] = cell_init*od0
    initial_array[s_i,:,:] = np.greater(cell_init, col_thresh)*(xs/(rc+ps))
    
def sim_pad_prep(p0, scale, tmax, prep_fn):
    # Calculate dx and redefine p0
    Dc, Dn, rc, Kn, Hn, pn, xs, ps, leak, od0 = p0
    dx = np.power((scale/4.5),2)
    p0 = np.array([dx, Dc, Dn, rc, Kn, Hn, pn, xs, ps, leak, od0])
    D_vec = [Dc, Dn, Dc]
    
    # Prep initial array
#     print('load initial cell frame')
    species = 3 # cells, nutrients, mscarlet
    cell_init = prep_fn(scale)
    n_h, n_w = cell_init.shape
    initial_array = prep_initial_array(od0, species, cell_init)
    col_thresh = 1e-4
    
    dims = [species, n_h, n_w, scale]
    c_i, n_i, s_i = np.arange(species)
    
    # Make empty array, and tolerance arrays
#     print('make tolerance arrays')
    atol = np.zeros((species, n_h, n_w), dtype=np.float32,order='C')# + 1e-7
    atol[c_i,:,:] = 1e-3*np.ones((n_h, n_w), dtype=np.float32)
    atol[n_i,:,:] = 1e-2*np.ones((n_h, n_w), dtype=np.float32)
    atol[s_i,:,:] = 1e-2*np.ones((n_h, n_w), dtype=np.float32)
    # atol must be a vector for the solver
    atol.shape = species*n_h*n_w
    rtol = 1e-3
                             
    @numba.jit(cache=True,nopython=True)
    def calc_f(y, d_y, diff_terms, nut_avail, p0):
        c_i, n_i, s_i = np.arange(3)
        dx, Dc, Dn, rc, Kn, Hn, pn, xs, ps, leak, od = p0
        calc_diffusion(y, diff_terms)

        # Nutrient term
        nut_avail[:,:] = hill(y[n_i,:,:], Hn, Kn)

        # Cell growth and diffusion
        d_y[c_i,:,:] = (dx)*Dc*diff_terms[c_i,:,:] + rc * nut_avail * y[c_i,:,:]

        # Nutrient consumption
        d_y[n_i,:,:] = (dx)*Dn*diff_terms[n_i,:,:] - pn * nut_avail * y[c_i,:,:]

        # Synthase production
        d_y[s_i,:,:] = (dx)*Dc*diff_terms[s_i,:,:] + (xs * np.greater(y[c_i,:,:], 1e-4) - rc * y[s_i,:,:]) * nut_avail - ps * y[s_i,:,:]
    
    calc_jac_wrapper = prep_jac(p0, dims)
    
    return dims, p0, initial_array, tmax, atol, rtol, calc_f, calc_jac_wrapper

def sim_pad(p0, scale, tmax, prep_fn):
    return wrapper(*sim_pad_prep(p0, scale, tmax, prep_fn))

