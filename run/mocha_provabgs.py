'''

simplest test on PROVABGS mocks generated using jupyter notebook:
https://github.com/changhoonhahn/provabgs/blob/main/nb/provabgs_mocks.ipynb

The mocks are generated using provabgs. All at redshift z=0.2. This is the 
simplest that will confirm that the model and inference pipelines are working!

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
corrp = sys.argv[5] == 'True'
#####################################################################

# read mock wavelength, flux, and theta 
dat_dir = os.path.join(UT.dat_dir(), 'mini_mocha', 'provabgs_mocks')
wave_obs    = np.load(os.path.join(dat_dir, 'provabgs_mock.wave.npy')) 
flux_obs    = np.load(os.path.join(dat_dir, 'provabgs_mock.flux.npy')) 
theta_obs   = np.load(os.path.join(dat_dir, 'provabgs_mock.theta.npy')) 

# all flux at z = 0.2 
z_obs = 0.2

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
        fchain_npy  = os.path.join(dat_dir, 'provabgs_mock.%i.cp.chain.npy' % i_obs)
        fchain_p    = os.path.join(dat_dir, 'provabgs_mock.%i.cp.chain.p' % i_obs)
    else: 
        fchain_npy  = os.path.join(dat_dir, 'provabgs_mock.%i.chain.npy' % i_obs)
        fchain_p    = os.path.join(dat_dir, 'provabgs_mock.%i.chain.p' % i_obs)

    #if os.path.isfile(fchain_npy) and os.path.isfile(fchain_p): 
    #    return None 

    if corrp: 
        # desi MCMC object
        desi_mcmc = Infer.desiMCMC(model=m_nmf, prior=prior, corrprior=CP_kde)
    else: 
        # desi MCMC object
        desi_mcmc = Infer.desiMCMC(model=m_nmf, prior=prior)

    print('true theta = ', theta_obs[i_obs])
    # run MCMC
    zeus_chain = desi_mcmc.run(
        wave_obs=wave_obs,
        flux_obs=flux_obs[i_obs],
        flux_ivar_obs=np.ones(wave_obs.shape),
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
