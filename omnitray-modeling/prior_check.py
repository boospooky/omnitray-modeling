# imports
# from __future__ import division, print_function

import itertools as itt

import numpy as np
import pandas as pd
import os
import sys
import string

import skimage.io
import skimage.transform
import emcee

from multiprocessing import Pool, Process, cpu_count

from omnisim import PrepPad, Fitter, Jacobian, Simulator

def logprior(p0):
    from scipy.stats import cauchy, norm

    if np.any(p0<0):
        return -np.inf
    Dn, rc, Kn, Hn, pn = p0
    if np.any(np.array([Dn,rc,Kn,Hn,pn])==0):
        return -np.inf

    # Boundaries are applied by imposing Gaussian edges to parameters that fall outside a reasonable range
    val = 0

    # Dn is jeffrey's
    val += (-1/2)*np.log(Dn) + norm.logpdf(1e3*np.max([0, Dn-1e-3, 1e3*(1e-6-Dn)]),0,0.1)

    # rc under cauchy distribution
    cauchy_params = (0.00022759430504618882, 1.8699032413798176e-05)
    val += cauchy.logpdf(rc, *cauchy_params)

    # Kn uniform between 0 and 100 (maybe change this for appearance's sake)
    if Kn>200 or Kn<1:
        val += norm.logpdf(np.max([0,Kn-200, 10*(1-Kn)]),0,0.01)

    # Hn uniform between 0.8 and 20
    if Hn > 20 or Hn < 0.5:
        val += norm.logpdf(np.max([0,10*(Hn-20), 10*(0.5-Hn)]),0,0.01)

    # pn is jeffrey's
    val += (-1/2)*np.log(pn) + norm.logpdf(np.max([0,5e2*(1e-3-pn),pn-2]),0,1)

    # 100*rc/pn is the amount of cells that can be produced in a unitary arena
    # let's say it should be able to produce confluence, or 1 unit of cell density, with std of 0.2
    val += norm.logpdf(100*rc/pn, 1, 0.4)
    return val

# Get prior-distributed parameters
nwalkers = 12
ndim = 5
if False:
    nsteps = 10000
    Dn_vals = np.power(10,np.linspace(-5.9,-2.9,nwalkers))
    rc_vals = np.linspace(1.1e-4,2.9e-4,nwalkers)
    Kn_vals = np.linspace(1,99,nwalkers)
    Hn_vals = np.linspace(0.6, 19, nwalkers)
    pn_vals = np.power(np.linspace(1.2e-4,3,nwalkers),2)
    pos = np.array([np.random.choice(Dn_vals, nwalkers), 
                    np.random.choice(rc_vals, nwalkers), 
                    np.random.choice(Kn_vals, nwalkers), 
                    np.random.choice(Hn_vals, nwalkers), 
                    np.random.choice(pn_vals, nwalkers)]).T
#     nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprior, pool=pool)
        sampler.run_mcmc(pos, nsteps);
    flatchain = np.reshape(sampler.get_chain()[3000:,:,:].copy(),(nwalkers*(nsteps-1000),ndim),order='F')
    np.save('prior_flatchain.npy', flatchain)
else:
    flatchain = np.load('prior_flatchain.npy')

def prior_check(flatchain, flatter):
    # pick truth from 

# Setup MCMC from prior-distributed parameters
datecode = '190908'
save_tmpl = 'worker_outputs/pool_outputs/low_sensitivity_all_pads_run_emcee_{}.csv'
nsteps = 20
run_flag = True

scale_exp = 4
# for pad_ind in [0,2,4,5,6,7,8]:
pad_list = [2]
save_file = save_tmpl.format(datecode)
fitter = Fitter(scale_exp, pad_list, logprior,4)
post_pos = flatchain[np.random.choice(np.arange(flatchain.shape[0]),nwalkers),:]

nsteps = 10000
with Pool() as pool:
    post_sampler = emcee.EnsembleSampler(nwalkers, ndim, fitter.resfun, pool=pool)
    print('starting')
    post_sampler.run_mcmc(post_pos, nsteps, progress=True);
    post_flatchain = post_sampler.get_chain(flat=True)
    pd.DataFrame(post_flatchain).to_csv(save_file, header=False, mode='w')
