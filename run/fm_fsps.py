'''

forward model BGS photometry and spectroscopy for spectra generated from FSPS
model. This is to sanity check the pipeline.  

'''
import os 
import glob
import h5py 
import pickle 
import numpy as np 
import scipy as sp 
# --- provabgs --- 
from provabgs import models as Models 
from provabgs import infer as Infer
# --- astropy --- 
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from astropy.table import Table
# --- gqp_mc ---
import gqp_mc.fm as FM 
import gqp_mc.util as UT 


version = '1.0' # 04/30/2020 


def fm_FSPS_mini_mocha(lib='bc03'): 
    ''' generate spectroscopy and photometry for the mini Mock Challenge (MoCha)
    
    * input: galaxy properties (SFH, ZH, etc), noiseless spectra 
    * "true" photometry directly from noiseless spectra
    * assign photometric uncertainty and fiber flux using legacy imaging 
    * "measured" photometry and fiber flux + fiber source spectra (scaled down noiseless spectra), 
    * "BGS" spectra
    '''
    from scipy.spatial import cKDTree as KDTree
    np.random.seed(0) 

    # generate some random parameter values and redshifts
    priors = Infer.load_priors([
            Infer.UniformPrior(8, 12, label='sed'),     # uniform priors on logM*
            Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
            Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
            Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
            Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
            Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
            Infer.UniformPrior(-2.2, 0.4, label='sed')      # uniform priors on dust_index 
            ])
    _theta = np.array([priors.sample() for i in range(1000)])
    theta = priors.transform(_theta) 

    redshifts = np.random.uniform(0.05, 0.3, size=1000) 
    
    # generate noiseless spectra using emulator 
    emu = Models.DESIspeculator()
    wave = 3e3 + 0.8 * np.arange(int((1.15e4 - 3e3)/0.8))
    print(wave) 
    _, flux = emu.sed(theta, redshifts, wavelength=wave) 

    # get meta data 
    meta = {} 
    for i, name in enumerate(['logM_total', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh',
            'beta4_sfh', 'gamma1_zh', 'gamma2_zh', 'dust1', 'dust2',
            'dust_index']): 
        meta[name] = theta[:,i]
    meta['sfr_1gyr']    = np.array([
        emu.avgSFR(tt, zred, dt=1.) for tt, zred in zip(theta, redshifts)])
    meta['sfr_100myr']  = np.array([
        emu.avgSFR(tt, zred, dt=0.1) for tt, zred in zip(theta, redshifts)])
    meta['Z_MW']        = np.array([emu.Z_MW(tt, zred) for tt, zred in zip(theta, redshifts)])
    meta['redshift']    = redshifts 

    # 1. generate 'true' photometry from noiseless spectra 
    photo_true, _ = FM.Photo_DESI(wave, flux) 
    
    # 2. assign uncertainties to the photometry using BGS targets from the Legacy survey 
    bgs_targets = h5py.File(os.path.join(UT.dat_dir(), 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    n_targets = len(bgs_targets['ra'][...]) 
    
    bands = ['g', 'r', 'z']#, 'w1', 'w2']#, 'w3', 'w4']

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

    f_fiber = np.clip(photo_fiber_true/photo_true[:,1], None, 1.) # (r fiber flux) / (r total flux) 
    meta['logM_fiber'] = np.log10(f_fiber) + meta['logM_total']

    # apply uncertainty to fiber flux as well 
    photo_fiber_meas = photo_fiber_true + f_fiber * photo_ivars[:,1]**-0.5 * np.random.randn(photo_true.shape[0]) 
    photo_ivar_fiber = f_fiber**-2 * photo_ivars[:,1] 

    # 3.b. get fiber spectra by scaling down noiseless Lgal source spectra
    spectra_fiber = flux * f_fiber[:,None] # 10e-17 erg/s/cm2/A

    # 4. generate BGS like spectra
    # read in sampled observing conditions, sky brightness, and exposure time 
    _fsky = os.path.join(UT.dat_dir(), 'mini_mocha', 
            'bgs.exposure.surveysim.150s.v0p4.sample.hdf5') 
    fsky = h5py.File(_fsky, 'r') 

    nexp = len(fsky['airmass'][...]) # number of exposures 
    wave_sky    = fsky['wave'][...] # sky wavelength 
    sbright_sky = fsky['sky'][...]

    # store to meta-data 
    for k in ['airmass', 'moon_alt', 'moon_ill', 'moon_sep', 'seeing', 'sun_alt', 'sun_sep', 'texp_total', 'transp']: 
        meta[k] = fsky[k][...]
    meta['wave_sky']    = wave_sky 
    meta['sbright_sky'] = sbright_sky 
    
    # generate BGS spectra for the exposures 
    spectra_bgs = {} 
    for iexp in range(nexp): 

        # sky brightness of exposure 
        Isky = [wave_sky * u.Angstrom, sbright_sky[iexp]]

        fbgs = os.path.join(UT.dat_dir(), 'mini_mocha',
                'fsps.bgs_spec.%s.v%s.%iof%i.fits' % (lib, version, iexp+1, nexp)) 
    
        if not os.path.isfile(fbgs): 
            bgs_spec = FM.Spec_BGS(
                    wave,        # wavelength  
                    spectra_fiber,            # fiber spectra flux 
                    fsky['texp_total'][...][iexp],  # exp time
                    fsky['airmass'][...][iexp],     # airmass 
                    Isky, 
                    filename=fbgs) 
        else: 
            from desispec.io import read_spectra 
            bgs_spec = read_spectra(fbgs) 

        if iexp == 0: 
            spectra_bgs['wave_b'] = bgs_spec.wave['b']
            spectra_bgs['wave_r'] = bgs_spec.wave['r']
            spectra_bgs['wave_z'] = bgs_spec.wave['z']
            spectra_bgs['flux_b'] = np.zeros((nexp, bgs_spec.flux['b'].shape[0], bgs_spec.flux['b'].shape[1])) 
            spectra_bgs['flux_r'] = np.zeros((nexp, bgs_spec.flux['r'].shape[0], bgs_spec.flux['r'].shape[1])) 
            spectra_bgs['flux_z'] = np.zeros((nexp, bgs_spec.flux['z'].shape[0], bgs_spec.flux['z'].shape[1])) 
            spectra_bgs['ivar_b'] = np.zeros((nexp, bgs_spec.flux['b'].shape[0], bgs_spec.flux['b'].shape[1])) 
            spectra_bgs['ivar_r'] = np.zeros((nexp, bgs_spec.flux['r'].shape[0], bgs_spec.flux['r'].shape[1])) 
            spectra_bgs['ivar_z'] = np.zeros((nexp, bgs_spec.flux['z'].shape[0], bgs_spec.flux['z'].shape[1])) 
            spectra_bgs['res_b'] = np.zeros((nexp,
                bgs_spec.resolution_data['b'].shape[0],
                bgs_spec.resolution_data['b'].shape[1], 
                bgs_spec.resolution_data['b'].shape[2])) 
            spectra_bgs['res_r'] = np.zeros((nexp,
                bgs_spec.resolution_data['r'].shape[0],
                bgs_spec.resolution_data['r'].shape[1], 
                bgs_spec.resolution_data['r'].shape[2])) 
            spectra_bgs['res_z'] = np.zeros((nexp,
                bgs_spec.resolution_data['z'].shape[0],
                bgs_spec.resolution_data['z'].shape[1], 
                bgs_spec.resolution_data['z'].shape[2])) 


        spectra_bgs['flux_b'][iexp] = bgs_spec.flux['b']
        spectra_bgs['flux_r'][iexp] = bgs_spec.flux['r']
        spectra_bgs['flux_z'][iexp] = bgs_spec.flux['z']
        
        spectra_bgs['ivar_b'][iexp] = bgs_spec.ivar['b']
        spectra_bgs['ivar_r'][iexp] = bgs_spec.ivar['r']
        spectra_bgs['ivar_z'][iexp] = bgs_spec.ivar['z']

        spectra_bgs['res_b'][iexp] = bgs_spec.resolution_data['b']
        spectra_bgs['res_r'][iexp] = bgs_spec.resolution_data['r']
        spectra_bgs['res_z'][iexp] = bgs_spec.resolution_data['z']

    # write out everything 
    fmeta = os.path.join(UT.dat_dir(), 'mini_mocha',
            'fsps.mini_mocha.%s.v%s.meta.p' % (lib, version))
    fout = h5py.File(os.path.join(UT.dat_dir(), 'mini_mocha', 
        'fsps.mini_mocha.%s.v%s.hdf5' % (lib, version)), 'w')

    pickle.dump(meta, open(fmeta, 'wb')) # meta-data

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
    fout.create_dataset('spec_wave_source', data=wave) 
    fout.create_dataset('spec_flux_source', data=flux) 
    # noiseless source spectra in fiber 
    fout.create_dataset('spec_fiber_flux_source', data=spectra_fiber)
    
    # BGS source spectra 
    for k in spectra_bgs.keys(): 
        fout.create_dataset('spec_%s_bgs' % k, data=spectra_bgs[k]) 
    fout.close() 
    return None 


if __name__=="__main__": 
    fm_FSPS_mini_mocha()
