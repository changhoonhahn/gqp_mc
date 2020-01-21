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


def Spectra(sim='lgal', noise='none', lib='bc03', sample='mini_mocha'): 
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
        (default: 'bc03') 
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
    if sim != 'lgal': raise NotImplementedError 
    if lib != 'bc03': raise NotImplementedError 
    
    dir_sample = os.path.join(UT.dat_dir(), sample) 

    # read in meta data 
    meta = pickle.load(open(os.path.join(dir_sample, "lgal.%s.%s.meta.p" % (sample, lib)), 'rb')) 
    meta = _avg_sfr(meta)
    
    # read in mock data 
    mock = h5py.File(os.path.join(dir_sample, 'lgal.%s.%s.hdf5' % (sample, lib)), 'r') 
    
    if 'bgs' in noise: iobs = int(noise.strip('bgs')) 

    specs = {} 
    specs['frac_fiber'] = mock ['frac_fiber'][...] # fiber flux scaling factor 

    if noise == 'none': 
        specs['wave']            = mock['spec_wave_source'][...] 
        specs['flux']            = mock['spec_fiber_flux_source'][...]
        specs['flux_unscaled']   = mock['spec_flux_source'][...]

    elif 'bgs' in noise: 
        # concatenate the 3 spectrograph outputs for convenience. 
        specs['wave'] = np.concatenate([
            mock['spec_wave_b_bgs'][...], 
            mock['spec_wave_r_bgs'][...], 
            mock['spec_wave_z_bgs'][...]])
        
        specs['flux'] = np.concatenate([
            mock['spec_flux_b_bgs'][...][iobs,:,:], 
            mock['spec_flux_r_bgs'][...][iobs,:,:], 
            mock['spec_flux_z_bgs'][...][iobs,:,:]], 
            axis=1)

        specs['ivar'] = np.concatenate([
            mock['spec_ivar_b_bgs'][...][iobs,:,:], 
            mock['spec_ivar_r_bgs'][...][iobs,:,:], 
            mock['spec_ivar_z_bgs'][...][iobs,:,:]], 
            axis=1)

    return specs, meta 


def Photometry(sim='lgal', noise='none', lib='bc03', sample='mini_mocha'): 
    ''' read forward modeled photometry generated for simulations

    :param sim: 
        name of simulation. currently only supports LGal SAM sim. (default: 'lgal')

    :param noise: 
        specify the noise of the photometry. Options are 'none' or 'legacy'. (default: none) 
    
    :param lib:
        stellar library used to generate the spectra. lib == 'bc03' only supported. (default: 'bc03') 

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
    if sim != 'lgal': raise NotImplementedError 
    if lib != 'bc03': raise NotImplementedError 
    
    dir_sample = os.path.join(UT.dat_dir(), sample) 

    # read in meta data 
    meta = pickle.load(open(os.path.join(dir_sample, "lgal.%s.%s.meta.p" % (sample, lib)), 'rb')) 
    meta = _avg_sfr(meta)
    
    # read in mock data 
    mock = h5py.File(os.path.join(dir_sample, 'lgal.%s.%s.hdf5' % (sample, lib)), 'r') 
    
    if 'bgs' in noise: iobs = int(noise.strip('bgs')) 

    bands = ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']

    photo = {} 
    photo['flux'] = np.zeros((mock['photo_flux_r_true'][...].shape[0], 7)) 
    if noise == 'none': 
        for icol, band in enumerate(bands): 
            photo['flux'][:,icol] = mock['photo_flux_%s_true' % band][...] # 'true' 

    elif noise == 'legacy': 
        photo['ivar'] = np.zeros((mock['photo_flux_r_true'][...].shape[0], 7)) 
        for icol, band in enumerate(bands): 
            photo['flux'][:,icol] = mock['photo_flux_%s_meas' % band][...] # 'measured'
            photo['ivar'][:,icol] = mock['photo_ivar_%s_true' % band][...]
    photo['flux_r_true'] = mock['photo_flux_r_true'][...]
    photo['fiberflux_r_true'] = mock['photo_fiberflux_r_true'][...]
    photo['fiberflux_r_meas'] = mock['photo_fiberflux_r_meas'][...]
    photo['fiberflux_r_ivar'] = mock['photo_fiberflux_r_ivar'][...]
    return photo, meta


def make_mini_mocha(lib='bc03'): 
    ''' generate spectroscopy and photometry for the mini Mock Challenge (MoCha)
    
    * input: galaxy properties (SFH, ZH, etc), noiseless spectra 
    * "true" photometry directly from noiseless spectra
    * assign photometric uncertainty and fiber flux using legacy imaging 
    * "measured" photometry and fiber flux + fiber source spectra (scaled down noiseless spectra), 
    * "BGS" spectra
    '''
    from scipy.spatial import cKDTree as KDTree
    # read in mini mocha galids 
    fids = os.path.join(UT.dat_dir(), 'mini_mocha', 'lgal.galids.%s.txt' % lib)
    galids = np.loadtxt(fids, skiprows=1) 

    # get Lgal meta data 
    _meta = _lgal_metadata(galids)
    
    # get noiseless source spectra 
    _meta_spec, spectra_s = _lgal_noiseless_spectra(galids, lib=lib) 

    # compile meta-data 
    meta = {} 
    for k in _meta.keys(): meta[k] = _meta[k]
    for k in _meta_spec.keys(): meta[k] = _meta_spec[k] 
    print('%.2f < z < %.2f' % (meta['redshift'].min(), meta['redshift'].max()))

    # 1. generate 'true' photometry from noiseless spectra 
    photo_true, _ = FM.Photo_DESI(spectra_s['wave'], spectra_s['flux_dust']) 
    
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

    f_fiber = photo_fiber_true/photo_true[:,1] # (r fiber flux) / (r total flux) 
    assert f_fiber.max() <= 1.
    meta['logM_fiber'] = np.log10(f_fiber) + meta['logM_total']

    # apply uncertainty to fiber flux as well 
    photo_fiber_meas = photo_fiber_true + f_fiber * photo_ivars[:,1]**-0.5 * np.random.randn(photo_true.shape[0]) 
    photo_ivar_fiber = f_fiber**-2 * photo_ivars[:,1] 

    # 3.b. get fiber spectra by scaling down noiseless Lgal source spectra
    spectra_fiber = spectra_s['flux_dust'] * f_fiber[:,None] # 10e-17 erg/s/cm2/A

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

        fbgs = os.path.join(UT.dat_dir(), 'mini_mocha', 'lgal.bgs_spec.%s.%iof%i.fits' % (lib, iexp+1, nexp)) 

        bgs_spec = FM.Spec_BGS(
                spectra_s['wave'],        # wavelength  
                spectra_fiber,            # fiber spectra flux 
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
    fmeta = os.path.join(UT.dat_dir(), 'mini_mocha', 'lgal.mini_mocha.%s.meta.p' % lib)
    fout = h5py.File(os.path.join(UT.dat_dir(), 'mini_mocha', 'lgal.mini_mocha.%s.hdf5' % lib), 'w')

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
    wlim = (spectra_s['wave'] < 2e5) & (spectra_s['wave'] > 1e3) # truncating the spectra  
    fout.create_dataset('spec_wave_source', data=spectra_s['wave'][wlim]) 
    fout.create_dataset('spec_flux_source', data=spectra_s['flux_dust'][:,wlim]) 
    # noiseless source spectra in fiber 
    fout.create_dataset('spec_fiber_flux_source', data=spectra_fiber[:,wlim])
    
    # BGS source spectra 
    for k in spectra_bgs.keys(): 
        fout.create_dataset('spec_%s_bgs' % k, data=spectra_bgs[k]) 
    fout.close() 
    return None 


def _avg_sfr(meta): 
    ''' given the mocha metadata calculate average SFRs over 100 Myr and 1Gyr 
    '''
    meta['sfr_1gyr'] = [] 
    meta['sfr_100myr'] = [] 
    for i in range(len(meta['sfh_bulge'])): 
        sfh_total =  meta['sfh_bulge'][i] + meta['sfh_disk'][i] # total sfh 

        in1gyr = (meta['t_lookback'][i] <= 1) 
        in100myr = (meta['t_lookback'][i] <= 0.1) 

        meta['sfr_1gyr'].append(np.sum(sfh_total[in1gyr]) / 1.e9) 
        meta['sfr_100myr'].append(np.sum(sfh_total[in100myr]) / 1.e8) 
    return meta  


def _mini_mocha_galid(lib='bc03'): 
    ''' pick 100 unique Lgal galids that roughly fall under the BGS target selection 
    for the mini mock challenge: r < 20. 
    '''
    # gather all galids 
    galids = [] 
    dir_inputs = os.path.join(UT.lgal_dir(), 'gal_inputs')
    for finput in glob.glob(dir_inputs+'/*'): 
        galids.append(int(os.path.basename(finput).split('_')[2]))
    galids = np.array(galids) 
    n_id = len(galids) 

    # get noiseless source spectra 
    _, spectra_s = _lgal_noiseless_spectra(galids, lib=lib)
    # get DECAM photometry 
    photo, _ = FM.Photo_DESI(spectra_s['wave'], spectra_s['flux_dust']) 

    target_selection = (photo[:,1] <= 20.) 
    print('%i Lgal galaxies within target_selection' % np.sum(target_selection)) 

    # now randomly choose 100 galids 
    mini_galids = np.random.choice(galids[target_selection], size=100, replace=False) 
    fids = os.path.join(UT.dat_dir(), 'mini_mocha', 'lgal.galids.%s.txt' % lib)
    np.savetxt(fids, mini_galids, fmt='%i', header='%i Lgal galids for mini mock challenge' % len(mini_galids)) 
    return None 


def _lgal_noiseless_spectra(galids, lib='bc03'): 
    ''' return noiseless source spectra of Lgal galaxies given the galids and 
    the library. The spectra is interpolated to a standard wavelength grid. 
    '''
    n_id = len(galids) 

    if lib == 'bc03': str_lib = 'BC03_Stelib'

    # noiseless source spectra
    _Fsource = lambda galid: os.path.join(UT.lgal_dir(), 'templates', 
            'gal_spectrum_%i_BGS_template_%s.fits' % (galid, str_lib)) 
    
    wavemin, wavemax = 3000.0, 3e5
    wave = np.arange(wavemin, wavemax, 0.2)
    flux_dust = np.zeros((n_id, len(wave)))
    flux_nodust = np.zeros((n_id, len(wave)))
    
    redshift, cosi, tau_ism, tau_bc, vd_disk, vd_bulge = [], [], [], [], [], [] 
    for i, galid in enumerate(galids): 
        f_source = fits.open(_Fsource(galid)) 
        # grab extra meta data from header
        hdr = f_source[0].header
        redshift.append(    hdr['REDSHIFT'])
        cosi.append(        hdr['COSI'])
        tau_ism.append(     hdr['TAUISM'])
        tau_bc.append(      hdr['TAUBC'])
        vd_disk.append(     hdr['VD_DISK'])
        vd_bulge.append(    hdr['VD_BULGE'])

        specin = f_source[1].data
        
        _flux_dust      = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A
        _flux_nodust    = specin['flux_nodust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A

        interp_flux_dust    = sp.interpolate.interp1d(specin['wave'], _flux_dust, fill_value='extrapolate') 
        interp_flux_nodust  = sp.interpolate.interp1d(specin['wave'], _flux_nodust, fill_value='extrapolate') 

        flux_dust[i,:]      = interp_flux_dust(wave) 
        flux_nodust[i,:]    = interp_flux_nodust(wave) 

    meta = {
            'redshift': np.array(redshift), 
            'cosi':     np.array(cosi), 
            'tau_ism':  np.array(tau_ism), 
            'tau_bc':   np.array(tau_bc), 
            'vd_disk':  np.array(vd_disk), 
            'vd_bulge': np.array(vd_bulge) 
            } 
    spectra = {
            'wave': wave, 
            'flux_dust': flux_dust, 
            'flux_nodust': flux_nodust
            } 
    return meta, spectra


def _lgal_metadata(galids): 
    ''' return galaxy properties (meta data) of Lgal galaxies 
    given the galids 
    '''
    tlookback, dt = [], [] 
    sfh_disk, sfh_bulge, Z_disk, Z_bulge, logM_disk, logM_bulge, logM_total = [], [], [], [], [], [], []
    t_age_MW, Z_MW = [], [] 
    for i, galid in enumerate(galids): 
        f_input = os.path.join(UT.lgal_dir(), 'gal_inputs', 
                'gal_input_%i_BGS_template_FSPS_uvmiles.csv' % galid) 
        gal_input = Table.read(f_input, delimiter=' ')

        tlookback.append(gal_input['sfh_t']) # lookback time (age) 
        dt.append(gal_input['dt'])
        # SF history 
        sfh_disk.append(gal_input['sfh_disk'])
        sfh_bulge.append(gal_input['sfh_bulge'])
        # metalicity history 
        Z_disk.append(gal_input['Z_disk'])
        Z_bulge.append(gal_input['Z_bulge'])
        # formed mass  
        logM_disk.append(np.log10(np.sum(gal_input['sfh_disk'])))
        logM_bulge.append(np.log10(np.sum(gal_input['sfh_bulge'])))
        logM_total.append(np.log10(np.sum(gal_input['sfh_disk']) + np.sum(gal_input['sfh_bulge'])))
        # mass weighted
        t_age_MW.append(np.sum(gal_input['sfh_t'] * (gal_input['sfh_disk'] + gal_input['sfh_bulge'])) / np.sum(gal_input['sfh_disk'] + gal_input['sfh_bulge']))
        Z_MW.append(np.sum(gal_input['Z_disk'] * gal_input['sfh_disk'] + gal_input['Z_bulge'] * gal_input['sfh_bulge']) / np.sum(gal_input['sfh_disk'] + gal_input['sfh_bulge']))
    
    meta = {} 
    meta['galid']       = galids
    meta['t_lookback']  = tlookback
    meta['sfh_disk']    = sfh_disk
    meta['sfh_bulge']   = sfh_bulge
    meta['Z_disk']      = Z_disk
    meta['Z_bulge']     = Z_bulge
    meta['logM_disk']   = logM_disk
    meta['logM_bulge']  = logM_bulge
    meta['logM_total']  = logM_total
    meta['t_age_MW']    = t_age_MW
    meta['Z_MW']        = Z_MW
    return meta
