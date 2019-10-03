''' 

script to profile various function calls 


'''
import time 
import numpy as np 
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
from gqp_mc import fitters as Fitters


if __name__=='__main__': 
    t0 = time.time() 
    # read Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim='lgal', noise='bgs1', lib='bc03', sample='mini_mocha') 
    print('Data.Spectra takes %.3f sec' % (time.time()-t0)) # 0.049sec

    w_obs           = specs['wave']
    spec_obs        = specs['flux'][0]
    spec_ivar_obs   = specs['ivar'][0]

    # read nonoise Lgal spectra    
    specs_non, _ = Data.Spectra(sim='lgal', noise='none', lib='bc03', sample='mini_mocha') 
    w_obs_non           = specs_non['wave']
    spec_obs_non        = specs_non['flux'][0]
    spec_ivar_obs_non   = np.ones(w_obs_non.shape[0]) 

    t0 = time.time() 
    # read Lgal photometry of the mini_mocha mocks 
    photo, _ = Data.Photometry(sim='lgal', noise='legacy', lib='bc03', sample='mini_mocha') 
    print('Data.Photometry takes %.3f sec' % (time.time()-t0)) # 0.019sec

    photo_obs       = photo['flux'][0,:5]
    photo_ivar_obs  = photo['ivar'][0,:5]

    # initial fitter  
    ifsps = Fitters.iFSPS(model_name='vanilla', prior=None) 
    # middle of prior theta value 
    theta_test = np.average(np.array(ifsps.priors), axis=1)
    
    # get mask 
    t0 = time.time() 
    _mask = ifsps._check_mask('emline', w_obs, spec_ivar_obs, meta['redshift'][0]) 
    print('iFSPS._check_mask takes %.3f sec' % (time.time()-t0)) # <0.001sec

    ifsps._lnPost_spec(theta_test, w_obs, spec_obs, spec_ivar_obs, meta['redshift'][0], mask=_mask, prior_shape='flat')
    # log Posterior for noisy spectral fitting
    t0 = time.time() 
    for i in range(10):
        ifsps._lnPost_spec(theta_test, w_obs, spec_obs, spec_ivar_obs, meta['redshift'][0], mask=_mask, prior_shape='flat')
    print('iFSPS._lnPost_spec takes %.3f sec' % ((time.time()-t0)/10.)) 

    _mask = ifsps._check_mask('emline', w_obs_non, spec_ivar_obs_non, meta['redshift'][0]) 
    # log Posterior for noiseless spectral fitting
    t0 = time.time() 
    for i in range(10): 
        ifsps._lnPost_spec(theta_test, w_obs_non, spec_obs_non, spec_ivar_obs_non, meta['redshift'][0], mask=_mask, prior_shape='flat')
    print('iFSPS._lnPost_spec (no noise) takes %.3f sec' % ((time.time()-t0)/10.)) 

    # log Posterior for photometric fitting
    t0 = time.time()  
    for i in range(10): 
        ifsps._lnPost_photo(theta_test, photo_obs, photo_ivar_obs, meta['redshift'][0], 
                filters=None, bands='desi', prior_shape='flat')
    print('iFSPS._lnPost_photo takes %.3f sec' % ((time.time()-t0)/10.)) 


