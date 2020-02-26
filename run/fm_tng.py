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

    notes
    -----
    *   only 5000 galaxies for the mini_mocha 
    '''
    from scipy.spatial import cKDTree as KDTree
    # read TNG source spectra 
    ftng    = h5py.File(os.path.join(UT.dat_dir(), 'tng', 'tng.hdf5'), 'r') 
    wave    = ftng['wave'][...]
    sed     = ftng['sed_neb'][...][:5000,:] # SED with nebular emission lines 
    ngal    = sed.shape[0]
    print('%i illustris tng galaxies' % ngal) 
    
    # compile meta data 
    meta = {} 
    meta['logM_total']  = ftng['logmstar'][...][:ngal]
    meta['sfr_100myr']  = 10.**ftng['logsfr.100'][...][:ngal]
    
    _wave_s = wave * u.Angstrom
    _spec_s = sed * u.Lsun / u.Hz

    H0 = 70 * u.km / u.s / u.Mpc
    
    # 0. assign redshifts to maximize the number of galaxies within r < 20. 
    # **note that this will completely skew dn/dz but kept as a first pass.**
    def rmag_z(z):
        wave_s = (_wave_s * (1.+z)).value # redshift the spectra 

        # convert fluxes 
        _spec = _spec_s/(4.*np.pi * (z * const.c/H0).to(u.cm)**2) / _wave_s**2 * const.c 
        spec_s = (_spec.to(u.erg / u.s / u.cm**2 / u.Angstrom)).value * 1e17

        _, mags = FM.Photo_DESI(wave_s, spec_s) 
        return mags[:,1]

    zarr = np.linspace(0., 0.4, 41) 
    zarr[0] = 0.0001 
    rmags = np.array([rmag_z(_z) for _z in zarr]).T

    zred = np.empty(ngal) 
    for i in range(ngal): 
        imax = np.abs(rmags[i,:] - 20.).argmin() 
        zmax = zarr[imax]
        zred[i] = np.random.uniform(0., zmax)

    meta['redshift'] = zred # save to metadata 

    # 1. generate 'true' photometry from noiseless spectra 
    wave_s = np.tile(_wave_s.value, (ngal,1)) # redshift wavelength
    wave_s *= (1.+zred[:,None])  

    # convert fluxes 
    _spec = _spec_s/(4.*np.pi * (zred[:,None] * const.c/H0).to(u.cm)**2) / _wave_s**2 * const.c 
    spec_s = (_spec.to(u.erg / u.s / u.cm**2 / u.Angstrom)).value * 1e17

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
    meta['logM_fiber'] = np.log10(f_fiber) + meta['logM_total']

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
    
    # store to meta-data 
    for k in ['airmass', 'moon_alt', 'moon_ill', 'moon_sep', 'seeing', 'sun_alt', 'sun_sep', 'texp_total', 'transp']: 
        meta[k] = fsky[k][...]
    meta['wave_sky']    = wave_sky 
    meta['sbright_sky'] = sbright_sky 
    
    # generate BGS spectra for the exposures 
    spectra_bgs = {} 
    for iexp in range(nexp): 
        # sky brightness of exposure 
        Isky = [wave_sky, sbright_sky[iexp]]

        fbgs = os.path.join(UT.dat_dir(), 'tng', 'tng.mini_mocha.bgs_spec.%iof%i.fits' % (iexp+1, nexp)) 
        
        # interpolate source spectra onto linearly spaced wavelengths 
        wlin = np.linspace(1e3, 2e4, 19000)
        spectra_fiber_lin = np.zeros((spectra_fiber.shape[0], len(wlin)))
        for i in range(spectra_fiber.shape[0]):
            spectra_fiber_lin[i] = np.interp(wlin, wave_s[i,:], spectra_fiber[i,:]) 
        
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
            spectra_bgs['flux_b'] = np.zeros((nexp, bgs_spec.flux['b'].shape[0], bgs_spec.flux['b'].shape[1])) 
            spectra_bgs['flux_r'] = np.zeros((nexp, bgs_spec.flux['r'].shape[0], bgs_spec.flux['r'].shape[1])) 
            spectra_bgs['flux_z'] = np.zeros((nexp, bgs_spec.flux['z'].shape[0], bgs_spec.flux['z'].shape[1])) 
            spectra_bgs['ivar_b'] = np.zeros((nexp, bgs_spec.flux['b'].shape[0], bgs_spec.flux['b'].shape[1])) 
            spectra_bgs['ivar_r'] = np.zeros((nexp, bgs_spec.flux['r'].shape[0], bgs_spec.flux['r'].shape[1])) 
            spectra_bgs['ivar_z'] = np.zeros((nexp, bgs_spec.flux['z'].shape[0], bgs_spec.flux['z'].shape[1])) 

        spectra_bgs['flux_b'][iexp] = bgs_spec.flux['b']
        spectra_bgs['flux_r'][iexp] = bgs_spec.flux['r']
        spectra_bgs['flux_z'][iexp] = bgs_spec.flux['z']
        
        spectra_bgs['ivar_b'][iexp] = bgs_spec.ivar['b']
        spectra_bgs['ivar_r'][iexp] = bgs_spec.ivar['r']
        spectra_bgs['ivar_z'][iexp] = bgs_spec.ivar['z']

    # write out everything 
    fmeta = os.path.join(UT.dat_dir(), 'mini_mocha', 'tng.mini_mocha.meta.p')
    pickle.dump(meta, open(fmeta, 'wb')) # meta-data

    fout = h5py.File(os.path.join(UT.dat_dir(), 'tng', 'tng.mini_mocha.fm.hdf5'), 'w')

    fout.create_dataset('redshift', data=zred)
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
    wlim = (_wave_s.value > 1e3) & (_wave_s.value < 2e4) 
    fout.create_dataset('spec_wave_source', data=wave_s[:,wlim]) 
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
