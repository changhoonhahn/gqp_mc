__all__ = ['test_iSpeculator']

import pytest
import numpy as np 
# --- gqp_mc --- 
from gqp_mc import data as Data
from gqp_mc import fitters as Fitters


@pytest.mark.parametrize("data_type", ('spec', 'photo'))

def test_iSpeculator(data_type): 
    
    # initiate fitter
    iSpec = Fitters.iSpeculator(model_name='emulator') 
    # default prior 
    prior = iSpec._default_prior()

    specs, meta = Data.Spectra(sim='lgal', noise='bgs0', lib='bc03', sample='mini_mocha') 
    photo, _ = Data.Photometry(sim='lgal', noise='legacy', lib='bc03', sample='mini_mocha') 

    zred        = meta['redshift'][0] 
    w_obs       = specs['wave']
    flux_obs    = specs['flux'][0]
    photo_obs   = photo['flux'][0,:3]
    
    if data_type == 'spec': 
        output = iSpec.MCMC_spec(
                w_obs, 
                flux_obs,
                np.ones(len(w_obs)), 
                zred,
                prior=prior, 
                nwalkers=20, 
                burnin=10, 
                niter=10, 
                silent=False)
        assert 'theta_med' in output.keys() 

    elif data_type == 'photo':  
        output = iSpec.MCMC_photo(
                photo_obs, 
                np.ones(len(photo_obs)), 
                zred,
                bands='desi', 
                prior=prior, 
                nwalkers=20, 
                burnin=10, 
                niter=10, 
                opt_maxiter=100,
                silent=False) 
        assert 'theta_med' in output.keys() 
