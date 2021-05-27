'''

P2 test (LGalaxies mocks) using provabgs model 

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
flux_obs    = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))[:,:3]
ivar_obs    = np.load(os.path.join(dat_dir, 'mocha_p2.ivar.npy'))[:,:3]
theta       = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb')  )

z_obs = theta['redshift']

# declare model  
m_nmf = Models.NMF(burst=True, emulator=True)

# set prior 
prior = Infer.load_priors([
    Infer.UniformPrior(7., 12.5, label='sed'),
    Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
    Infer.UniformPrior(0., 1., label='sed'), # burst fraction
    Infer.LogUniformPrior(0., 13.27, label='sed'), # tburst
    Infer.LogUniformPrior(4.5e-5, 4.5e-2, label='sed'), # log uniform priors on ZH coeff
    Infer.LogUniformPrior(4.5e-5, 4.5e-2, label='sed'), # log uniform priors on ZH coeff
    Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1
    Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
    Infer.UniformPrior(-3., 1., label='sed')     # uniform priors on dust_index
])


def run_mcmc(i_obs): 
    fchain_npy  = os.path.join(dat_dir, 'L2', 'P2.provabgs.%i.chain.npy' % i_obs)
    fchain_p    = os.path.join(dat_dir, 'L2', 'P2.provabgs.%i.chain.p' % i_obs)

    #if os.path.isfile(fchain_npy) and os.path.isfile(fchain_p): 
    #    return None 
    
    # desi MCMC object
    desi_mcmc = Infer.desiMCMC(model=m_nmf, prior=prior)

    # run MCMC
    zeus_chain = desi_mcmc.run(
            bands='desi', # g, r, z
            photo_obs=flux_obs[i_obs], 
            photo_ivar_obs=ivar_obs[i_obs], 
            zred=z_obs[i_obs],
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
