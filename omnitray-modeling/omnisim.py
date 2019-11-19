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

### Define common functions and load metadata ###

pad_times_df = pd.read_csv('pad_times.csv')
pad_times_df = pad_times_df.loc[(pad_times_df.frame>=5)&(pad_times_df.frame<41),:]

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
    
@numba.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:])',cache=True,nopython=True)
def calc_f(y, d_y, diff_terms, nut_avail, p0):
    dx, Dc, Dn, rc, Kn, Hn, pn = p0
    calc_diffusion(y, diff_terms)

    # Nutrient term
    nut_avail[:,:] = hill(y[1,:,:], Hn, Kn)

    # Cell growth and diffusion
    d_y[0,:,:] = (dx)*Dc*diff_terms[0,:,:] + rc * nut_avail * y[0,:,:]

    # Nutrient consumption
    d_y[1,:,:] = (dx)*Dn*diff_terms[1,:,:] - pn * nut_avail * y[0,:,:]

def f_ivp(t, y, d_y, diff_terms, nut_avail, p0, dims, calc_f):
    y.shape = dims
    calc_f(y, d_y, diff_terms, nut_avail, p0)
    y.shape = np.prod(dims)
    return d_y.flatten()

def prep_pad_helper(scale, pad, frame):
    mask_fn_tmpl = 'worker_outputs/initial_frame_masks/pad_{}_frame_{}.tif'
    img_fn = mask_fn_tmpl.format(pad,frame)
    init_cells = skimage.io.imread(img_fn)
    scale_factor = (scale/4500)/(2.475/4)
    scaled_init = skimage.transform.rescale((init_cells>1).astype(np.float64), 
                                            scale_factor, 
                                            order=0, 
                                            mode='constant', 
                                            multichannel=False,
                                            preserve_range=True,
                                            cval=0)
    del init_cells
    return scaled_init

def get_t_eval(pov_ind, frames=np.arange(5,41)):
    return pad_times_df.loc[pad_times_df.pad==pov_ind,'time'].values.astype(np.float64)

def pov_0_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = (np.array((n_h, n_w))//2) - np.array((s_h, s_w))//2
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)

def pov_1_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = (np.array((n_h, n_w))//2) - np.array((s_h, s_w))//2
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)

def pov_2_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = (np.array((n_h, n_w))//2) - np.array((s_h, s_w))//2
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)

def pov_4_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = (np.array((n_h, n_w))//2) - np.array((s_h, s_w))//2
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)

def pov_5_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = np.array(((n_h-s_h)//2, scale//8))
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)

def pov_6_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = np.array(((n_h-s_h)//2, 6*n_w//8))
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)

def pov_7_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = np.array(((n_h-s_h)//2, scale//2))
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)

def pov_8_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = np.array(((n_h-s_h)//2, (n_w-s_w)//2))
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)

def pov_9_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = np.array(((n_h-s_h)//2, (n_w-s_w)//2))
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)
            
def pov_10_slices(n_h, n_w, s_h, s_w, scale):
    y0, x0 = (np.array((0, n_w-s_w))//2)
    y1, x1 = np.array((y0,x0)) + np.array((s_h, s_w))
    return slice(y0,y1), slice(x0,x1)


class PrepPad(object):
    '''
    Prepares initial conditions for solve_ivp and facilitates comparisons of experimental data to simulation.  
    
    This class is defined according to a specific experiment. The pad dimensions and imaging locations are
    defined for each experimental pad in the __init__ function. Passing a scale and pad index parameter
    to the constructor 
    '''
    def get_time_vec(self):
        return get_t_eval(self.pov_inds[0])
    
    def get_fn_lists(self, frames=np.arange(5,41)):
        mask_fn_tmpl = 'worker_outputs/initial_frame_masks/pad_{}_frame_{}.tif'
        fn_format = lambda pov, frame : mask_fn_tmpl.format(pov, frame)
        fn_lists = [[fn_format(pov, frame) for frame in frames] for pov in self.pov_inds]
        return dict(zip(self.pov_inds, fn_lists))
    
    def prep_initial_cell(self):
        n_h, n_w = self.cell_init.shape
        if np.any(self.cell_init!=0):
            self.cell_init[:] = 0
        for pov_ind, slice_fun in zip(self.pov_inds, self.slice_functions):
            scaled_init = prep_pad_helper(self.scale, pov_ind, 5)
            s_h, s_w = scaled_init.shape
            yslice, xslice = slice_fun(n_h, n_w, s_h, s_w, self.scale)
            self.cell_init[yslice, xslice] += scaled_init
            
            
    def __init__(self, scale, pad_ind):
        self.scale = scale
        if pad_ind==0:
            n_h, n_w = (2*scale, 2*scale)
            self.cell_init = np.zeros((n_h, n_w), dtype=np.float64)        
            self.pad = 0
            self.pov_inds = [0]
            self.slice_functions = [pov_0_slices]
        
        elif pad_ind==1:
            n_h, n_w = (2*scale, 2*scale)
            self.cell_init = np.zeros((n_h, n_w), dtype=np.float64)        
            self.pad = 1
            self.pov_inds = [1, 10]
            self.slice_functions = [pov_1_slices, pov_10_slices]
            
        elif pad_ind==2:
            n_h, n_w = (2*scale, 2*scale)
            self.cell_init = np.zeros((n_h, n_w), dtype=np.float64)
            self.pad = 2
            self.pov_inds = [2]
            self.slice_functions = [pov_2_slices]

        elif pad_ind==4:
            n_h, n_w = (3*scale, 3*scale)
            self.cell_init = np.zeros((n_h, n_w), dtype=np.float64)
            self.pad = 4
            self.pov_inds = [4]
            self.slice_functions = [pov_4_slices]

        elif pad_ind==5:
            n_h, n_w = (5*scale, 5*scale)
            self.cell_init = np.zeros((n_h, n_w), dtype=np.float64)
            self.pad = 5
            self.pov_inds = [5,6]
            self.slice_functions = [pov_5_slices, pov_6_slices]
            
        elif pad_ind==6:
            n_h, n_w = (5*scale, 5*scale)
            self.cell_init = np.zeros((n_h, n_w), dtype=np.float64)
            self.pad = 6
            self.pov_inds = [7]
            self.slice_functions = [pov_7_slices]

        elif pad_ind==7:
            n_h, n_w = (3*scale, 3*scale)
            self.cell_init = np.zeros((n_h, n_w), dtype=np.float64)
            self.pad = 7
            self.pov_inds = [8]
            self.slice_functions = [pov_8_slices]

        elif pad_ind==8:
            n_h, n_w = (2*scale, 2*scale)
            self.cell_init = np.zeros((n_h, n_w), dtype=np.float64)
            self.pad = 8
            self.pov_inds = [9]
            self.slice_functions = [pov_9_slices]
            
        else:
            raise ValueError
        
        self.prep_initial_cell()
    

class Jacobian(object):
    '''
    The Jacobian used is a sparse matrix. On initialization, this class deterines the matrix structure and later can be populated with parameter values
    '''
    def f_ji(self, x, y, spec):
        _, n_h, n_w, _ = self.dims
        return x + n_w*y + n_w*n_h*spec
    
    def __init__(self, dims):
        species, n_h, n_w, scale = dims
        self.dims = dims
        self.dx = np.power(scale/4.5,2)
        c_i, n_i = np.arange(species)
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
        
        #dc/(dcdt)
        dcdcdt_indices = [(x, y, self.f_ji(x,y,c_i), self.f_ji(x,y,c_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #dc/(dndt)
        dcdndt_indices = [(x, y, self.f_ji(x,y,c_i), self.f_ji(x,y,n_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #dn/(dndt)
        dndndt_indices = [(x, y, self.f_ji(x,y,n_i), self.f_ji(x,y,n_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

        #dn/(dcdt)
        dndcdt_indices = [(x, y, self.f_ji(x,y,n_i), self.f_ji(x,y,c_i)) for x in np.arange(n_w) for y in np.arange(n_h)]

        self.rxn_indices_list = [dcdcdt_indices, dcdndt_indices, dndndt_indices, dndcdt_indices]
        n_terms = np.sum([len(xx) for xx in self.rxn_indices_list])
        self.rxn_vec = np.zeros(n_terms, dtype=np.float64)
        self.j1_rxn = np.zeros(n_terms, dtype=np.int)
        self.j2_rxn = np.zeros(n_terms, dtype=np.int)

    def set_p0(self, p0):
        self.p0 = p0
        Dc, Dn, rc, Kn, Hn, pn = p0
        self.D_vec = [Dc, Dn]
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
        Dc, Dn, rc, Kn, Hn, pn = self.p0
        D_vec = np.array(self.D_vec)
        dx = self.dx
        species, n_h, n_w, scale = self.dims
        c_i, n_i = np.arange(species)
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
        Dc, Dn, rc, Kn, Hn, pn = self.p0
        dcdcdt_indices, dcdndt_indices, dndndt_indices, dndcdt_indices = self.rxn_indices_list
        c_i, n_i = 0,1
        
        i = 0
        #dc/(dcdt)
        nut_avail = hill(y[n_i,:,:], Hn, Kn)
        dnut_avail = dhillda(y[n_i,:,:], Hn, Kn)
        val_arr = rc*nut_avail
        i = self.assign_rxn_vals(dcdcdt_indices, val_arr,i)

        #dc/(dndt)
        val_arr = rc*dnut_avail*y[c_i,:,:]
        i = self.assign_rxn_vals(dcdcdt_indices, val_arr,i)

        #dn/(dndt)
        val_arr = -pn*dnut_avail*y[c_i,:,:]
        i = self.assign_rxn_vals(dcdcdt_indices, val_arr,i)

        #dn/(dcdt)
        val_arr = -pn*nut_avail
        i = self.assign_rxn_vals(dcdcdt_indices, val_arr,i)

    def calc_jac_wrapper(self, t, y):
        species, n_h, n_w, scale = self.dims
        y.shape = (species,n_h,n_w)
        n_jac = species*n_h*n_w
        self.calc_rxn_jac(t,y)
        data_vec = np.concatenate([self.dif_vec, self.rxn_vec])
        j1_vec = np.concatenate([self.j1_dif, self.j1_rxn])
        j2_vec = np.concatenate([self.j2_dif, self.j2_rxn])
        y.shape = species*n_h*n_w
        return sparse.coo_matrix((data_vec, (j1_vec,j2_vec)),shape=(n_jac, n_jac),dtype=np.float64)


def prep_initial_array(species, cell_init):
    c_i, n_i = np.arange(species)
    col_thresh = 1e-4
    n_h, n_w = cell_init.shape
    # Set initial conditions
    initial_array = np.zeros((species, n_h, n_w), dtype=np.float64, order='C')# + 1e-7
    initial_array[n_i,:,:] = 100*np.ones((n_h, n_w), dtype=np.float64)
    initial_array[c_i,:,:] = cell_init
    return initial_array
    
def sim_pad_prep(scale, cell_init, t_eval):
    species = 2 # cells, nutrients, mscarlet
    n_h, n_w = cell_init.shape
    initial_array = prep_initial_array(species, cell_init)
    col_thresh = 1e-4
    
    dims = [species, n_h, n_w, scale]
    c_i, n_i = np.arange(species)
    
    # Make empty array, and tolerance arrays
    atol = np.zeros((species, n_h, n_w), dtype=np.float64,order='C')# + 1e-7
    atol[c_i,:,:] = 1e-4*np.ones((n_h, n_w), dtype=np.float64)
    atol[n_i,:,:] = 1e-3*np.ones((n_h, n_w), dtype=np.float64)
    # atol must be a vector for the solver
    atol.shape = species*n_h*n_w
    rtol = np.float64(1e-5)
    
    return dims, initial_array, t_eval, atol, rtol


def sim_pad(p0, scale, prep_fn):
    return wrapper(*sim_pad_prep(p0, scale, prep_fn))

class Simulator(object):
    '''
    Instances of this class are initialized with information requried to simulate an experimental pad and compare to data.
    '''
    def __init__(self, pad_prep):
        scale, cell_init, t_eval = pad_prep.scale, pad_prep.cell_init, pad_prep.get_time_vec()
        dims, initial_array, t_eval, atol, rtol = sim_pad_prep(scale, cell_init, t_eval)
        self.cell_init = cell_init
        self.dims = dims
        self.initial_array = initial_array
        self.t_eval = t_eval
        self.atol = atol
        self.rtol = rtol
        self.dx = np.power(scale/4.5,2)
        self.scale = scale
        self.pad = pad_prep.pad
        self.pov_inds = pad_prep.pov_inds
        self.fn_list = pad_prep.get_fn_lists()
        self.slice_functions = pad_prep.slice_functions
        self.jacobian = Jacobian(dims)
        self.load_exp_data()
            
    def load_exp_data(self):
        out_fn_tmpl = 'worker_outputs/downsampled_mask_arrs/pad{}_scale{}_exparr.tif'
        self.exp_list = self.pov_inds.copy()
        scale = np.log2(self.scale).astype(np.int)
        for i, pad_ind in enumerate(self.pov_inds):
            fn = out_fn_tmpl.format(pad_ind, scale)
            if os.path.isfile(fn):
                self.exp_list[i] = skimage.io.imread(fn)
            else:
                print('could not find '+fn)
                im_arr = skimage.io.imread_collection(self.fn_list[pad_ind]).concatenate()>1
                scale_factor = (4/2.475)*(self.scale/4500)
                sf = np.int(1/scale_factor)
                im_dwn = skimage.transform.rescale(im_arr.astype(np.float64),
                                                   (1.0,scale_factor,scale_factor),
                                                   order=0, 
                                                   mode='constant', 
                                                   multichannel=False,
                                                   preserve_range=True,
                                                   cval=0)
                skimage.io.imsave(fn, im_dwn.astype(np.float16))
                self.exp_list[i] = im_dwn
        
    def f_ivp_wrapper(self, t, y):
        return f_ivp(t, y, *self.args)

    def sim(self, p0):

        self.p0 = p0.astype(np.float64)
        Dc, Dn, rc, Kn, Hn, pn = self.p0
        params = np.array([self.dx, Dc, Dn, rc, Kn, Hn, pn], dtype=np.float64)

        species, n_h, n_w, scale = self.dims
        self.args=(np.zeros((species, n_h, n_w), dtype=np.float64,order='C'), 
                   np.zeros((species, n_h, n_w), dtype=np.float64,order='C'), 
                   np.zeros((n_h, n_w), dtype=np.float64,order='C'), 
                   params, (species, n_h, n_w), calc_f)

        self.initial_array.shape = (n_h*n_w*species,)
        self.jacobian.set_p0(self.p0)
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

    def fit_fun(self, p0):
        if np.any(p0<0):
            return np.inf
        try:
            self.sim(p0)
        except RuntimeError:
            return -np.inf
        n_frames, _, n_h, n_w = self.sim_arr.shape
        if n_frames != len(self.t_eval):
            return np.inf
        err_sum = 0
        for exp_arr, slice_fun in zip(self.exp_list, self.slice_functions):
            _, s_h, s_w = exp_arr.shape
            yslice, xslice = slice_fun(n_h, n_w, s_h, s_w, self.scale)
            err_sum += np.sum(np.power(exp_arr - self.sim_arr[:,0,yslice,xslice],2))
        return err_sum
    
    def make_exp_movie(self):
        n_h, n_w = self.cell_init.shape
        n_frames = len(self.t_eval)
        exp_arr = np.zeros((n_frames, n_h, n_w))
        for col_arr, slice_fun in zip(self.exp_list, self.slice_functions):
            _, s_h, s_w = col_arr.shape
            yslice, xslice = slice_fun(n_h, n_w, s_h, s_w, self.scale)
            exp_arr[:,yslice,xslice] = col_arr
        return exp_arr

def fit_wrapper(arg):
    sim_list, p0 = arg
    return np.sum([xx.fit_fun(p0) for xx in sim_list])

class Fitter(object):
    def __init__(self, scale_exp, pad_inds, logprior_fun, res_std=5):
        self.logprior_fun = logprior_fun
        self.res_std=res_std
        self.scale = np.power(2,scale_exp)
        n_pads = len(pad_inds)
        preppad_inputs = zip(n_pads*[self.scale], pad_inds)
        with Pool() as pool:
            self.pad_list = pool.starmap(PrepPad, preppad_inputs)
            self.sim_list = pool.map(Simulator, self.pad_list)
    
    def resfun(self, psub):
        Dn, rc, Kn, Hn, pn = psub
        Dc = 1e-12 # Improves linear algebra calculations
        p0 = np.array([Dc, Dn, rc, Kn, Hn, pn])
        logprior_val = self.logprior_fun(psub)
        if logprior_val == -np.inf:
            return logprior_val
        return self.loglikelihood(p0) + logprior_val

    def par_loglikelihood(self, p0):
        n_proc = np.min([cpu_count(), len(self.sim_list)])
        sim_divisions = [self.sim_list[i::n_proc] for i in np.arange(n_proc)]
        p0_list = n_proc*[p0]
        with Pool(n_proc) as p:
            residuals = np.sum(p.map(fit_wrapper, zip(sim_divisions, p0_list)))
        return -1*(1/2)*residuals/self.res_std**2
    
    def loglikelihood(self, p0):
        n_proc = np.min([cpu_count(), len(self.sim_list)])
        sim_divisions = [self.sim_list[i::n_proc] for i in np.arange(n_proc)]
        p0_list = n_proc*[p0]
        residuals = np.sum(list(map(fit_wrapper, zip(sim_divisions, p0_list))))
        res_std = 5
        return -1*(1/2)*residuals/self.res_std**2
    
    def par_resfun(self, psub):
        Dn, rc, Kn, Hn, pn = psub
        Dc = 1e-12
        p0 = np.array([Dc, Dn, rc, Kn, Hn, pn])
        logprior_val = self.logprior_fun(psub)
        if logprior_val == -np.inf:
            return logprior_val
        return self.par_loglikelihood(p0) + logprior_val

