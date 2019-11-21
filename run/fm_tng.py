'''

forward model BGS photometry and spectroscopy for TNG spectra

'''
import os 
import glob
import h5py 
import pickle 
import numpy as np 
import scipy as sp 
# --- astropy --- 
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from astropy.table import Table
# --- gqp_mc ---
import gqp_mc.fm as FM 
import gqp_mc.util as UT 


def fm_TNG(): 
    ''' generate spectroscopy and photometry for the mini Mock Challenge (MoCha)
    
    * input: galaxy properties (SFH, ZH, etc), noiseless spectra 
    * "true" photometry directly from noiseless spectra
    * assign photometric uncertainty and fiber flux using legacy imaging 
    * "measured" photometry and fiber flux + fiber source spectra (scaled down noiseless spectra), 
    * "BGS" spectra
    '''
    from scipy.spatial import cKDTree as KDTree
    # read TNG source spectra 
    fwave = os.path.join(UT.dat_dir(), 'tng', 'FSPS_wave_Full.txt') 
    fspec = os.path.join(UT.dat_dir(), 'tng', '0_TNG_Fullspectra_Nebular_onlyAGBdust.txt' ) 
    
    z_bgs = 0.2
    _wave_s = np.loadtxt(fwave) * u.Angstrom
    wave_s = (_wave_s * (1.+z_bgs)).value # redshift the spectra 

    # convert fluxes 
    _spec_s = np.loadtxt(fspec) * u.Lsun / u.Hz
    H0 = 70 * u.km / u.s / u.Mpc
    _spec_s *= 1./(4.*np.pi * (z_bgs * const.c/H0).to(u.cm)**2) / _wave_s**2 * const.c 
    spec_s = (_spec_s.to(u.erg / u.s / u.cm**2 / u.Angstrom)).value * 1e17

    # 1. generate 'true' photometry from noiseless spectra 
    photo_true, _ = FM.Photo_DESI(wave_s, spec_s) 
    
    # 2. assign uncertainties to the photometry using BGS targets from the Legacy survey 
    bgs_targets = h5py.File(os.path.join(UT.dat_dir(), 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    n_targets = len(bgs_targets['ra'][...]) 
    
    bands = ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']

    bgs_photo       = np.zeros((n_targets, len(bands))) 
    bgs_photo_ivar  = np.zeros((n_targets, len(bands)))
    bgs_fiberflux   = np.zeros(n_targets) # r-band fiber flux
    for ib, band in enumerate(bands): 
        bgs_photo[:,ib]         = bgs_targets['flux_%s' % band][...] 
        bgs_photo_ivar[:,ib]    = bgs_targets['flux_ivar_%s' % band][...] 
    bgs_fiberflux = bgs_targets['fiberflux_r'][...]
        
    # construct KD tree from BGS targets (currently downsampled) 
    bgs_features = np.array([bgs_photo[:,0], bgs_photo[:,1], bgs_photo[:,2], 
        bgs_photo[:,0] - bgs_photo[:,1], bgs_photo[:,1] - bgs_photo[:,2]]).T
    tree = KDTree(bgs_features) 
    # match ivars and fiberflux 
    match_features = np.array([photo_true[:,0], photo_true[:,1], photo_true[:,2], 
        photo_true[:,0] - photo_true[:,1], photo_true[:,1] - photo_true[:,2]]).T
    dist, indx = tree.query(match_features)
    photo_ivars = bgs_photo_ivar[indx,:] 
    photo_fiber_true = bgs_fiberflux[indx] 

    # 3.a. apply the uncertainty to the photometry to get "measured" photometry. 
    photo_meas = photo_true + photo_ivars**-0.5 * np.random.randn(photo_true.shape[0], photo_true.shape[1]) 
    # ***this needs to be checked; sometimes it gives greater than 1***
    # ***this needs to be checked; sometimes it gives greater than 1***
    # ***this needs to be checked; sometimes it gives greater than 1***
    # ***this needs to be checked; sometimes it gives greater than 1***
    f_fiber = np.clip(photo_fiber_true/photo_true[:,1], None, 1.) # (r fiber flux) / (r total flux) 

    # apply uncertainty to fiber flux as well 
    photo_fiber_meas = photo_fiber_true + f_fiber * photo_ivars[:,1]**-0.5 * np.random.randn(photo_true.shape[0]) 
    photo_ivar_fiber = f_fiber**-2 * photo_ivars[:,1] 

    # 3.b. get fiber spectra by scaling down noiseless Lgal source spectra
    spectra_fiber = spec_s * f_fiber[:,None] # 10e-17 erg/s/cm2/A

    # 4. generate BGS like spectra
    # read in sampled observing conditions, sky brightness, and exposure time 
    _fsky = os.path.join(UT.dat_dir(), 'mini_mocha', 'bgs.exposure.surveysim.150s.v0p4.sample.hdf5') 
    fsky = h5py.File(_fsky, 'r') 

    nexp = len(fsky['airmass'][...]) # number of exposures 
    wave_sky    = fsky['wave'][...] # sky wavelength 
    sbright_sky = fsky['sky'][...]

    # generate BGS spectra for the exposures 
    spectra_bgs = {} 
    iexp = 0 

    # sky brightness of exposure 
    Isky = [wave_sky, sbright_sky[iexp]]

    fbgs = os.path.join(UT.dat_dir(), 'tng', '0_tng.bgs_spec.fits') 
    
    # interpolate source spectra onto linearly spaced wavelengths 
    wlin = np.linspace(1e3, 2e4, 19000)
    spectra_fiber_lin = np.zeros((spectra_fiber.shape[0], len(wlin)))
    for i in range(spectra_fiber.shape[0]):
        spectra_fiber_lin[i] = np.interp(wlin, wave_s, spectra_fiber[i,:]) 
    
    bgs_spec = FM.Spec_BGS(
            wlin,                   # wavelength  
            spectra_fiber_lin,          # fiber spectra flux 
            fsky['texp_total'][...][iexp],  # exp time
            fsky['airmass'][...][iexp],     # airmass 
            Isky, 
            filename=fbgs) 

    if iexp == 0: 
        spectra_bgs['wave_b'] = bgs_spec.wave['b']
        spectra_bgs['wave_r'] = bgs_spec.wave['r']
        spectra_bgs['wave_z'] = bgs_spec.wave['z']
        spectra_bgs['flux_b'] = np.zeros((bgs_spec.flux['b'].shape[0], bgs_spec.flux['b'].shape[1])) 
        spectra_bgs['flux_r'] = np.zeros((bgs_spec.flux['r'].shape[0], bgs_spec.flux['r'].shape[1])) 
        spectra_bgs['flux_z'] = np.zeros((bgs_spec.flux['z'].shape[0], bgs_spec.flux['z'].shape[1])) 
        spectra_bgs['ivar_b'] = np.zeros((bgs_spec.flux['b'].shape[0], bgs_spec.flux['b'].shape[1])) 
        spectra_bgs['ivar_r'] = np.zeros((bgs_spec.flux['r'].shape[0], bgs_spec.flux['r'].shape[1])) 
        spectra_bgs['ivar_z'] = np.zeros((bgs_spec.flux['z'].shape[0], bgs_spec.flux['z'].shape[1])) 

    spectra_bgs['flux_b'] = bgs_spec.flux['b']
    spectra_bgs['flux_r'] = bgs_spec.flux['r']
    spectra_bgs['flux_z'] = bgs_spec.flux['z']
    
    spectra_bgs['ivar_b'] = bgs_spec.ivar['b']
    spectra_bgs['ivar_r'] = bgs_spec.ivar['r']
    spectra_bgs['ivar_z'] = bgs_spec.ivar['z']

    # write out everything 
    fout = h5py.File(os.path.join(UT.dat_dir(), 'tng', '0_tng.fm.hdf5'), 'w')

    # photometry  
    for i, b in enumerate(bands): 
        # 'true' 
        fout.create_dataset('photo_flux_%s_true' % b, data=photo_true[:,i]) 
        fout.create_dataset('photo_ivar_%s_true' % b, data=photo_ivars[:,i]) 
        # 'measured'
        fout.create_dataset('photo_flux_%s_meas' % b, data=photo_meas[:,i]) 

    # fiber flux 
    fout.create_dataset('photo_fiberflux_r_true', data=photo_fiber_true) 
    fout.create_dataset('photo_fiberflux_r_meas', data=photo_fiber_meas) 
    fout.create_dataset('photo_fiberflux_r_ivar', data=photo_ivar_fiber) 
    fout.create_dataset('frac_fiber', data=f_fiber) # fraction of flux in fiber
    
    # spectroscopy 
    # noiseless source spectra 
    wlim = (wave_s < 2e5) & (wave_s > 1e3) # truncating the spectra  
    fout.create_dataset('spec_wave_source', data=wave_s[wlim]) 
    fout.create_dataset('spec_flux_source', data=spec_s[:,wlim]) 
    # noiseless source spectra in fiber 
    fout.create_dataset('spec_fiber_flux_source', data=spectra_fiber[:,wlim])
    
    # BGS source spectra 
    for k in spectra_bgs.keys(): 
        fout.create_dataset('spec_%s_bgs' % k, data=spectra_bgs[k]) 
    fout.close() 
    return None 


if __name__=="__main__": 
    fm_TNG()
