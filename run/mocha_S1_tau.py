'''

S1 test (PROVABGS mocks) using tau model 

The PROVABGS mocks are generated using provabgs. All at redshift z=0.2. They have a
somewhat toned down BGS-like noise 

'''
import os, sys
import pickle 
import numpy as np
from functools import partial
from multiprocessing.pool import Pool 
# --- gqp_mc ---
from gqp_mc import util as UT 
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
#####################################################################

# read mock wavelength, flux, inverse variance, and theta 
dat_dir = os.path.join(UT.dat_dir(), 'mini_mocha')
wave_obs    = np.load(os.path.join(dat_dir, 'mocha_s1.wave.npy')) 
flux_obs    = np.load(os.path.join(dat_dir, 'mocha_s1.flux.npy'))
ivar_obs    = np.load(os.path.join(dat_dir, 'mocha_s1.ivar.npy'))  

# all flux at z = 0.2 
z_obs = 0.2

# declare model  
m_tau = Models.Tau(burst=True, emulator=False)
tage = m_tau.cosmo.age(z_obs).value

# set prior 
prior = Infer.load_priors([
    Infer.UniformPrior(9., 12., label='sed'), 
    Infer.UniformPrior(0.3, 1e1, label='sed'), # tau SFH
    Infer.UniformPrior(0., 0.2, label='sed'), # constant SFH
    Infer.UniformPrior(0., tage-2., label='sed'), # start time
    Infer.UniformPrior(0., 0.5, label='sed'),  # fburst
    Infer.UniformPrior(0., tage, label='sed'),  # tburst
    Infer.UniformPrior(1e-6, 1e-3, label='sed'), # metallicity
    Infer.UniformPrior(0., 4., label='sed')])


def run_mcmc(i_obs): 
    fchain_npy  = os.path.join(dat_dir, 'S1', 'S1.tau_model.%i.chain.npy' % i_obs)
    fchain_p    = os.path.join(dat_dir, 'S1', 'S1.tau_model.%i.chain.p' % i_obs)

    if os.path.isfile(fchain_npy) and os.path.isfile(fchain_p): 
        return None 
    
    # desi MCMC object
    desi_mcmc = Infer.desiMCMC(model=m_tau, prior=prior)

    # run MCMC
    zeus_chain = desi_mcmc.run(
        wave_obs=wave_obs,
        flux_obs=flux_obs[i_obs],
        flux_ivar_obs=ivar_obs,
        zred=z_obs,
        vdisp=0.,
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
