__all__ = ['test_iFSPS']

import pytest
import numpy as np 
# --- gqp_mc --- 
from gqp_mc import fitters as Fitters


@pytest.mark.parametrize("name, theta", 
        zip(['vanilla', 'dustless_vanilla'], [np.array([10., -2., 11., 0.5, 1.5]), np.array([10., -2., 11., 1.5])]))

def test_iFSPS(name, theta): 

    ifsps = Fitters.iFSPS(model_name=name)

    w, spec = ifsps.model(theta, zred=0.1) 
    assert len(w) == len(spec) 

    output = ifsps.MCMC_spec(w.flatten(), spec.flatten(), np.ones(w.shape[1]), 0.1, nwalkers=10, burnin=10, niter=10, silent=False)
    assert 'theta_med' in output.keys() 

    photo = ifsps.model_photo(theta, zred=0.1, bands='desi') 
    assert len(photo) == 5 # g, r, z, W1, W2

    output = ifsps.MCMC_photo(photo, np.ones(len(photo)), 0.1, bands='desi', nwalkers=10, burnin=10, niter=10, silent=False) 
    assert 'theta_med' in output.keys() 
