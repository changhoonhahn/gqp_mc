'''

run provabgs on LGalaxies spectral mocks for the mock challenge

'''
import os, sys
import pickle 
import numpy as np
from functools import partial
from multiprocessing.pool import Pool 
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data
# --- provabgs ---
from provabgs import infer as Infer
from provabgs import models as Models
from provabgs import corrprior as Corrprior

#####################################################################
# input 
#####################################################################
i0 = int(sys.argv[1]) 
i1 = int(sys.argv[2])
niter = int(sys.argv[3])
n_cpu = int(sys.argv[4])
corrp = sys.argv[5] == 'True'
#####################################################################

dat_dir = os.path.join(UT.dat_dir(), 'mini_mocha', 'lgal_mocks')

specs, meta = Data.Spectra(sim='lgal', noise='bgs', sample='mini_mocha')

# set prior 
prior = Infer.load_priors([
    Infer.UniformPrior(9., 12., label='sed'),
    Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
    Infer.UniformPrior(0., 1., label='sed'), # burst fraction
    Infer.UniformPrior(0., 13.27, label='sed'), # tburst
    Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
    Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
    Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1
    Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
    Infer.UniformPrior(-2.2, 0.4, label='sed')     # uniform priors on dust_index
])

# declare model  
m_nmf = Models.NMF(burst=True, emulator=True)

if corrp: 
    # set corrprior object
    CP_kde = Corrprior.CorrectPrior(
        m_nmf,
        prior,
        zred=z_obs,
        props=['logavgssfr_1gyr', 'z_mw'],
        Nprior=100000,
        range=[(-13., -9), (2e-3, 0.035)],
        method='kde',
        bandwidth=0.01,
        debug=True
    )

def run_mcmc(i_obs): 
    if corrp: 
        fchain_npy  = os.path.join(dat_dir, 'lgal.%i.cp.chain.npy' % i_obs)
        fchain_p    = os.path.join(dat_dir, 'lgal.%i.cp.chain.p' % i_obs)
    else: 
        fchain_npy  = os.path.join(dat_dir, 'lgal.%i.chain.npy' % i_obs)
        fchain_p    = os.path.join(dat_dir, 'lgal.%i.chain.p' % i_obs)

    if os.path.isfile(fchain_npy) and os.path.isfile(fchain_p): 
        return None 
    
    # desi MCMC object
    if corrp: 
        desi_mcmc = Infer.desiMCMC(model=m_nmf, prior=prior, corrprior=CP_kde)
    else: 
        desi_mcmc = Infer.desiMCMC(model=m_nmf, prior=prior)

    # run MCMC
    zeus_chain = desi_mcmc.run(
            wave_obs=[specs['wave_b'], specs['wave_r'], specs['wave_z']], 
            flux_obs=[specs['flux_b'][i_obs], specs['flux_r'][i_obs], specs['flux_z'][i_obs]], 
            flux_ivar_obs=[specs['ivar_b'][i_obs], specs['ivar_r'][i_obs], specs['ivar_z'][i_obs]],
            resolution=[specs['res_b'][i_obs], specs['res_r'][i_obs], specs['res_z'][i_obs]],
            zred=meta['redshift'][i_obs], 
            mask='emline',
            vdisp=50.,
            sampler='zeus',
            nwalkers=30, 
            burnin=0,
            opt_maxiter=2000,
            niter=niter, 
            progress=False,
            debug=True)
    chain = zeus_chain['mcmc_chain']

    # save chain 
    np.save(fchain_npy, chain)
    pickle.dump(zeus_chain, open(fchain_p, 'wb'))
    return None 

pool = Pool(processes=n_cpu) 
pool.map(partial(run_mcmc), np.arange(i0, i1+1))
pool.close()
pool.terminate()
pool.join()
