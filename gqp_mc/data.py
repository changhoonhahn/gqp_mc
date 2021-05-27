'''

module to deal with spectra and photometry data 
of the GQP mock challenge. 

author(s): ChangHoon Hahn

'''
import os 
import glob
import h5py 
import pickle 
import numpy as np 
import scipy as sp 
# --- astropy --- 
from astropy.io import fits
from astropy.table import Table
# --- gqp_mc ---
from . import fm as FM 
from . import util as UT 


#version = '1.1' # updated 12/10/2020 
version = '1.2' # updated 05/27/2021


def Spectra(sim='lgal', noise='none', lib='fsps', sample='mini_mocha'): 
    ''' read forward modeled spectra generated for simulations

    parameters
    ----------
    sim : str 
        (default: 'lgal')
        name of simulation. currently only supports LGal SAM sim. 
    noise : str 
        (default: 'none') 
        type of noise. 
        'none' == noiseless source spectra. 
        'bgs1' == DESI-BGS like noise 
    lib : str 
        (default: 'fsps') 
        stellar library used to generate the spectra 
    sample : str
        (default: 'spectral_challenge') 
        specify sample from the simulations. default is spectral_challenge, which
        are 100 randomly selected galaxies. 
    
    returns
    -------
    specs : array 
        spectra 
    meta : array 
        dictionary of meta data 
    '''
    meta, mock = read_data(sim=sim, noise=noise, lib=lib, sample=sample)
        
    specs = {} 
    specs['frac_fiber'] = mock ['frac_fiber'][...] # fiber flux scaling factor 

    if noise == 'none': 
        specs['wave']            = mock['spec_wave_source'][...]
        specs['flux']            = mock['spec_fiber_flux_source'][...]
        specs['flux_unscaled']   = mock['spec_flux_source'][...]
    elif noise == 'bgs': 
        # concatenate the 3 spectrograph outputs for convenience. 
        specs['wave'] = np.concatenate([
            mock['spec_wave_b_bgs'][...], 
            mock['spec_wave_r_bgs'][...], 
            mock['spec_wave_z_bgs'][...]])
        
        specs['flux'] = np.concatenate([
            mock['spec_flux_b_bgs'][...], 
            mock['spec_flux_r_bgs'][...], 
            mock['spec_flux_z_bgs'][...]], 
            axis=1)

        specs['ivar'] = np.concatenate([
            mock['spec_ivar_b_bgs'][...], 
            mock['spec_ivar_r_bgs'][...], 
            mock['spec_ivar_z_bgs'][...]], 
            axis=1)

        specs['wave_b'] = mock['spec_wave_b_bgs'][...]
        specs['wave_r'] = mock['spec_wave_r_bgs'][...]
        specs['wave_z'] = mock['spec_wave_z_bgs'][...]

        specs['flux_b'] = mock['spec_flux_b_bgs'][...]
        specs['flux_r'] = mock['spec_flux_r_bgs'][...]
        specs['flux_z'] = mock['spec_flux_z_bgs'][...]

        specs['ivar_b'] = mock['spec_ivar_b_bgs'][...]
        specs['ivar_r'] = mock['spec_ivar_r_bgs'][...]
        specs['ivar_z'] = mock['spec_ivar_z_bgs'][...]
        
        specs['res_b'] = mock['spec_res_b_bgs'][...]
        specs['res_r'] = mock['spec_res_r_bgs'][...]
        specs['res_z'] = mock['spec_res_z_bgs'][...]
    return specs, meta 


def Photometry(sim='lgal', noise='none', lib='fsps', sample='mini_mocha'): 
    ''' read forward modeled photometry generated for simulations

    :param sim: 
        name of simulation. currently only supports LGal SAM sim. (default: 'lgal')

    :param noise: 
        specify the noise of the photometry. Options are 'none' or 'legacy'. (default: none) 
    
    :param lib:
        stellar library used to generate the spectra. lib == 'fsps' only
        supported. (default: 'fsps') 

    :param sample:         
        specify sample from the simulations. default is spectral_challenge, which
        are 100 randomly selected galaxies. (default: 'spectral_challenge') 
    
    returns
    -------
    photo : dict 
        contains all the photometric data. mainly fluxes in maggies
    meta : array 
        dictionary of meta data 
    '''
    meta, mock = read_data(sim=sim, noise=noise, lib=lib, sample=sample)

    bands = ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']

    photo = {} 
    photo['flux'] = np.zeros((mock['photo_flux_r_true'][...].shape[0], 7)) 
    if noise == 'legacy': 
        photo['ivar'] = np.zeros((mock['photo_flux_r_true'][...].shape[0], 7)) 
        for icol, band in enumerate(bands): 
            try: 
                photo['flux'][:,icol] = mock['photo_flux_%s_meas' % band][...] # 'measured'
                photo['ivar'][:,icol] = mock['photo_ivar_%s_true' % band][...]
            except KeyError: 
                pass 
    for band in bands: 
        try: 
            photo['flux_%s_true' % band] = mock['photo_flux_%s_true' % band][...]
        except KeyError: 
            pass 
    photo['fiberflux_r_true'] = mock['photo_fiberflux_r_true'][...]
    photo['fiberflux_r_meas'] = mock['photo_fiberflux_r_meas'][...]
    photo['fiberflux_r_ivar'] = mock['photo_fiberflux_r_ivar'][...]
    return photo, meta


def read_data(sim='lgal', noise='none', lib='fsps', sample='mini_mocha'): 
    ''' read compiled data 

    notes
    -----
    * 4/30/2020: tng not supported to focus on lgal.  
    '''
    if sim not in ['lgal', 'fsps']: raise NotImplementedError 
    
    dir_sample = os.path.join(UT.dat_dir(), sample) 
    
    # description of data (sample, library used, and version number) 
    dat_descrip = '%s.%s.v%s' % (sample, lib, version) 

    # read in meta data 
    meta = pickle.load(open(os.path.join(dir_sample,
        "%s.%s.meta.p" % (sim, dat_descrip)), 'rb')) 
    if sim == 'lgal': meta = _lgal_avg_sfr(meta)

    # read in mock data 
    mock = h5py.File(os.path.join(dir_sample, 
        '%s.%s.hdf5' % (sim, dat_descrip)), 'r') 
    return meta, mock  


def _lgal_avg_sfr(meta): 
    ''' given the mocha metadata calculate average SFRs over 100 Myr and 1Gyr 
    '''
    meta['sfr_1gyr'] = [] 
    meta['sfr_100myr'] = [] 
    for i in range(len(meta['sfh_bulge'])): 
        sfh_total =  meta['sfh_bulge'][i] + meta['sfh_disk'][i] # total sfh 
        dt = meta['dt'][i] 

        in1gyr = (meta['t_lookback'][i] <= 1) 
        in100myr = (meta['t_lookback'][i] <= 0.1) 

        meta['sfr_1gyr'].append(
                np.sum(sfh_total[in1gyr]) / (np.sum(dt[in1gyr]) * 1e9))  
        meta['sfr_100myr'].append(
                np.sum(sfh_total[in100myr]) / (np.sum(dt[in100myr]) * 1.e9))
    return meta  
