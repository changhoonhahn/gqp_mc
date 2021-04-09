'''

P1 test (PROVABGS mocks) using provabgs model 

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

# read mock photometry, inverse variance, and theta 
dat_dir = os.path.join(UT.dat_dir(), 'mini_mocha')
flux_obs    = np.load(os.path.join(dat_dir, 'mocha_p1.flux.npy'))[:,:3]
ivar_obs    = np.load(os.path.join(dat_dir, 'mocha_p1.ivar.npy'))[:,:3]
theta_obs   = np.load(os.path.join(dat_dir, 'provabgs_mocks', 'provabgs_mock.theta.npy')) 

# all flux at z = 0.2 
z_obs = 0.2

# declare model  
m_nmf = Models.NMF(burst=True, emulator=True)

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


def run_mcmc(i_obs): 
    fchain_npy  = os.path.join(dat_dir, 'provabgs_mocks', 'P1.provabgs.%i.chain.npy' % i_obs)
    fchain_p    = os.path.join(dat_dir, 'provabgs_mocks', 'P1.provabgs.%i.chain.p' % i_obs)

    #if os.path.isfile(fchain_npy) and os.path.isfile(fchain_p): 
    #    return None 
    
    # desi MCMC object
    desi_mcmc = Infer.desiMCMC(model=m_nmf, prior=prior)

    # run MCMC
    zeus_chain = desi_mcmc.run(
            bands='desi', # g, r, z
            photo_obs=flux_obs[i_obs], 
            photo_ivar_obs=ivar_obs[i_obs], 
            zred=z_obs,
            vdisp=0.,
            sampler='zeus',
            theta_start=prior.untransform(theta_obs[i_obs]),
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
