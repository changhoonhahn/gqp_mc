'''

module for population inference using the posteriors  

'''
import numpy as np 
import scipy.optimize as op

import emcee  


def eta_Delta_opt(chains):  
    ''' calculate population hyperparameters eta_Delta for a given set of
    chains along given property
    '''
    # optimization kwargs
    opt_kwargs = {'method': 'L-BFGS-B', 'bounds': ((None, None), (1e-4, None))}
    #, options={'eps': np.array([0.01, 0.005]), 'maxiter': 100})

    _logL_pop = lambda _theta: -1.*logL_pop(_theta[0], _theta[1], chains)
    
    _min = op.minimize(_logL_pop, np.array([0., 0.1]), **opt_kwargs) 
    return _min['x'][0], _min['x'][1]


def eta_Delta_mcmc(chains, nwalkers=10, niter=5000, burnin=500, thin=10):  
    ''' calculate population hyperparameters eta_Delta for a given set of
    chains along given property
    '''
    # initialize 
    mu0, sig0 = eta_Delta_opt(chains)
    eta0 = np.array([mu0, sig0]) + 1e-4 * np.random.randn(nwalkers, 2)

    logProb = lambda _theta: logL_pop(_theta[0], _theta[1], chains)

    sampler = emcee.EnsembleSampler(nwalkers, 2, logProb)
    sampler.run_mcmc(eta0, niter, progress=True)

    return sampler.get_chain(discard=burnin, thin=thin, flat=True)


def logL_pop(mu_pop, sigma_pop, delta_chains, prior=None): 
    ''' log likelihood of population variables mu, sigma for a uninformative prior 
    
    :param mu_pop: 

    :param sigma_pop: 

    :param delta_chains: (default: None) 
        Ngal x Niter 

    :param prior: (default: None) 
        prior function  
    '''
    assert prior is None, 'only implemented for uniformative prior so far'

    logprob = np.sum(np.log(np.nanmean(normal_pdf(delta_chains, mu_pop,
        sigma_pop), axis=1) + 1e-300))

    if np.isnan(logprob): return -np.inf
    else: return logprob


def normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std)**2) / (std * (2 * np.pi)**0.5)

