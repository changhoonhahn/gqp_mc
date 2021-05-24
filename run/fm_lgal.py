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


#version = '1.1' # 12/10/2020 
version = '1.2' # 05/24/2021 


def fm_Lgal_fsps(): 
    ''' construct mock spectra and photometry using L-Galaxies SED constructed
    using FSPS MILES stellar library and MIST isochrones
    '''
    # read in LGal SED
    flgal = os.path.join('/global/cfs/cdirs/desi/mocks/LGal_spectra',
            'Lgal_fsps_mocha.p')
    lgal_dict = pickle.load(open(flgal, 'rb'))
    ################################################################################
    # 0. compile meta-data
    meta = {} 
    meta['t_lookback']  = lgal_dict['t_sfh']
    meta['dt']          = lgal_dict['dt']
    meta['sfh_disk']    = lgal_dict['sfh_disk']
    meta['sfh_bulge']   = lgal_dict['sfh_bulge']
    meta['Z_disk']      = lgal_dict['Z_disk']
    meta['Z_bulge']     = lgal_dict['Z_bulge']
    meta['logM_disk']   = [np.log10(np.sum(sfh)) for sfh in lgal_dict['sfh_disk']]
    meta['logM_bulge']  = [np.log10(np.sum(sfh)) for sfh in lgal_dict['sfh_bulge']]
    meta['logM_total']  = [np.log10(np.sum(sfh0) + np.sum(sfh1)) for sfh0, sfh1
            in zip(lgal_dict['sfh_disk'], lgal_dict['sfh_bulge'])]

    # mass weighted age and metallicity 
    t_age_mw, z_mw = [], [] 
    for i in range(len(lgal_dict['dt'])): 
        t_age_mw.append(
                np.sum(lgal_dict['t_sfh'][i] * (lgal_dict['sfh_disk'][i] + lgal_dict['sfh_bulge'][i])) /
                np.sum(lgal_dict['sfh_disk'][i] + lgal_dict['sfh_bulge'][i])
                )
        z_mw.append(
                np.sum(lgal_dict['Z_disk'][i] * lgal_dict['sfh_disk'][i] +
                    lgal_dict['Z_bulge'][i] * lgal_dict['sfh_bulge'][i]) / 
                np.sum(lgal_dict['sfh_disk'][i] + lgal_dict['sfh_bulge'][i])
                )
    meta['t_age_MW']    = t_age_mw 
    meta['Z_MW']        = z_mw
    meta['redshift']    = lgal_dict['redshift'] 
    meta['cosi']        = lgal_dict['cosi']
    meta['tau_ism']     = lgal_dict['tauISM']
    meta['tau_bc']      = lgal_dict['tauBC']
    meta['vd_disk']     = lgal_dict['vd_disk']
    meta['vd_bulge']    = lgal_dict['vd_bulge']
    print('%.2f < z < %.2f' % (np.min(meta['redshift']), np.max(meta['redshift'])))

    ################################################################################
    # 1. generate 'true' photometry from noiseless spectra 
    wave = np.array(lgal_dict['wave_obs'])
    # convert from Lsun/A/m2 --> 1e-17 erg/s/A/cm2
    flux_dust = np.array(lgal_dict['flux_dust']) * 3.846e33 * 1e-4 * 1e17
    # interpoalte to save wavelength grid
    wave_lin = np.arange(1e3, 3e5, 0.2)
    flux_dust_interp = np.zeros((flux_dust.shape[0], len(wave_lin)))
    for i in range(flux_dust.shape[0]): 
        interp_flux_dust        = sp.interpolate.interp1d(wave[i], flux_dust[i], fill_value='extrapolate') 
        flux_dust_interp[i,:]   = interp_flux_dust(wave_lin) 

    bands = ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']
    photo_true, mag_true = FM.Photo_DESI(wave, flux_dust, bands=bands) 
    ################################################################################
    # 2. assign uncertainties to the photometry and fiberfrac using BGS targets from the Legacy survey 
    bgs_targets = h5py.File(os.path.join(UT.dat_dir(), 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    n_targets = len(bgs_targets['ra'][...]) 

    bgs_photo       = np.zeros((n_targets, len(bands))) 
    bgs_photo_ivar  = np.zeros((n_targets, len(bands)))
    bgs_fiberflux   = np.zeros(n_targets) # r-band fiber flux
    for ib, band in enumerate(bands): 
        bgs_photo[:,ib]         = bgs_targets['flux_%s' % band][...] 
        bgs_photo_ivar[:,ib]    = bgs_targets['flux_ivar_%s' % band][...] 
    bgs_fiberflux = bgs_targets['fiberflux_r'][...]
        
    from scipy.spatial import cKDTree as KDTree
    # construct KD tree from BGS targets (currently downsampled) 
    #bgs_features = np.array([bgs_photo[:,0], bgs_photo[:,1], bgs_photo[:,2], 
    #    bgs_photo[:,0] - bgs_photo[:,1], bgs_photo[:,1] - bgs_photo[:,2]]).T
    bgs_features = np.array([bgs_photo[:,1], bgs_photo[:,0] - bgs_photo[:,1], bgs_photo[:,1] - bgs_photo[:,2]]).T
    tree = KDTree(bgs_features) 
    # match ivars and fiberflux 
    lgal_features = np.array([photo_true[:,1], photo_true[:,0] - photo_true[:,1], photo_true[:,1] - photo_true[:,2]]).T
    dist, indx = tree.query(lgal_features)

    photo_ivars = bgs_photo_ivar[indx,:] 
    photo_fiber_true = bgs_fiberflux[indx] 
    ################################################################################
    # 3. apply noise model to photometry
    # 3.a. apply the uncertainty to the photometry to get "measured" photometry. 
    photo_meas = photo_true + photo_ivars**-0.5 * np.random.randn(photo_true.shape[0], photo_true.shape[1]) 

    f_fiber = photo_fiber_true/photo_true[:,1] # (r fiber flux) / (r total flux) 
    assert f_fiber.max() <= 1.
    meta['logM_fiber'] = np.log10(f_fiber) + meta['logM_total']

    # apply uncertainty to fiber flux as well 
    photo_fiber_meas = photo_fiber_true + f_fiber * photo_ivars[:,1]**-0.5 * np.random.randn(photo_true.shape[0]) 
    photo_ivar_fiber = f_fiber**-2 * photo_ivars[:,1] 

    # 3.b. get fiber spectra by scaling down noiseless Lgal source spectra
    spectra_fiber = flux_dust_interp * f_fiber[:,None] # 10e-17 erg/s/cm2/A

    ################################################################################
    # 4. generate BGS like spectra
    from feasibgs import spectral_sims as BGS_spec_sim
    from feasibgs import forwardmodel as BGS_fm

    Idark = BGS_spec_sim.nominal_dark_sky()
    fdesi = BGS_fm.fakeDESIspec()

    spectra_bgs = {} 

    fbgs = os.path.join(UT.dat_dir(), 'mini_mocha', 'fsps.bgs_spec.fsps.v%s.fits' % version)
    print(fbgs)
    bgs_spec = fdesi.simExposure(
            wave_lin,        # wavelength  
            spectra_fiber,            # fiber spectra flux 
            exptime=180.,
            airmass=1.1,
            Isky=[Idark[0].value, Idark[1].value],
            filename=fbgs
        )

    spectra_bgs['wave_b'] = bgs_spec.wave['b']
    spectra_bgs['wave_r'] = bgs_spec.wave['r']
    spectra_bgs['wave_z'] = bgs_spec.wave['z']
    spectra_bgs['flux_b'] = bgs_spec.flux['b']
    spectra_bgs['flux_r'] = bgs_spec.flux['r']
    spectra_bgs['flux_z'] = bgs_spec.flux['z']
    
    spectra_bgs['ivar_b'] = bgs_spec.ivar['b']
    spectra_bgs['ivar_r'] = bgs_spec.ivar['r']
    spectra_bgs['ivar_z'] = bgs_spec.ivar['z']

    spectra_bgs['res_b'] = bgs_spec.resolution_data['b']
    spectra_bgs['res_r'] = bgs_spec.resolution_data['r']
    spectra_bgs['res_z'] = bgs_spec.resolution_data['z']
    ################################################################################
    # 5. write out everything 
    # meta-data to pickle file
    fmeta = os.path.join(UT.dat_dir(), 'mini_mocha',
            'lgal.mini_mocha.fsps.v%s.meta.p' % (version))
    pickle.dump(meta, open(fmeta, 'wb')) # meta-data
    
    # the rest 
    fout = h5py.File(os.path.join(UT.dat_dir(), 'mini_mocha', 
        'lgal.mini_mocha.fsps.v%s.hdf5' % (version)), 'w')
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
    wlim = (wave_lin < 2e5) & (wave_lin > 1e3) # truncating the spectra  
    fout.create_dataset('spec_wave_source', data=wave_lin[wlim]) 
    fout.create_dataset('spec_flux_source', data=flux_dust_interp[:,wlim]) 
    # noiseless source spectra in fiber 
    fout.create_dataset('spec_fiber_flux_source', data=spectra_fiber[:,wlim])
    
    # BGS source spectra 
    for k in spectra_bgs.keys(): 
        fout.create_dataset('spec_%s_bgs' % k, data=spectra_bgs[k]) 
    fout.close() 
    return None 


def fm_Lgal_mini_mocha(lib='bc03'): 
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
    bands = ['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']
    photo_true, _ = FM.Photo_DESI(spectra_s['wave'], spectra_s['flux_dust'],
            bands=bands) 
    
    # 2. assign uncertainties to the photometry using BGS targets from the Legacy survey 
    bgs_targets = h5py.File(os.path.join(UT.dat_dir(), 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    n_targets = len(bgs_targets['ra'][...]) 

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
    from feasibgs import spectral_sims as BGS_spec_sim
    from feasibgs import forwardmodel as BGS_fm

    Idark = BGS_spec_sim.nominal_dark_sky()
    fdesi = BGS_fm.fakeDESIspec()

    spectra_bgs = {} 

    fbgs = os.path.join(UT.dat_dir(), 'mini_mocha', 'fsps.bgs_spec.%s.v%s.fits' % (lib, version))
    bgs_spec = fdesi.simExposure(
            spectra_s['wave'],        # wavelength  
            spectra_fiber,            # fiber spectra flux 
            exptime=180.,
            airmass=1.1,
            Isky=[Idark[0].value, Idark[1].value],
            filename=fbgs
        )

    spectra_bgs['wave_b'] = bgs_spec.wave['b']
    spectra_bgs['wave_r'] = bgs_spec.wave['r']
    spectra_bgs['wave_z'] = bgs_spec.wave['z']
    spectra_bgs['flux_b'] = bgs_spec.flux['b']
    spectra_bgs['flux_r'] = bgs_spec.flux['r']
    spectra_bgs['flux_z'] = bgs_spec.flux['z']
    
    spectra_bgs['ivar_b'] = bgs_spec.ivar['b']
    spectra_bgs['ivar_r'] = bgs_spec.ivar['r']
    spectra_bgs['ivar_z'] = bgs_spec.ivar['z']

    spectra_bgs['res_b'] = bgs_spec.resolution_data['b']
    spectra_bgs['res_r'] = bgs_spec.resolution_data['r']
    spectra_bgs['res_z'] = bgs_spec.resolution_data['z']

    # write out everything 
    fmeta = os.path.join(UT.dat_dir(), 'mini_mocha',
            'lgal.mini_mocha.%s.v%s.meta.p' % (lib, version))
    fout = h5py.File(os.path.join(UT.dat_dir(), 'mini_mocha', 
        'lgal.mini_mocha.%s.v%s.hdf5' % (lib, version)), 'w')

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
    elif lib == 'fsps': str_lib = 'FSPS_uvmiles'
    else: raise ValueError

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
    meta['dt']          = dt 
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


def QA_fm_Lgal_mini_mocha(lib='bc03'): 
    ''' quality assurance/sanity plots 
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.xmargin'] = 1
    mpl.rcParams['xtick.labelsize'] = 'x-large'
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.labelsize'] = 'x-large'
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['legend.frameon'] = False

    # read mini mocha data 
    fmm = h5py.File(os.path.join(UT.dat_dir(), 'mini_mocha', 
        'lgal.mini_mocha.%s.v%s.hdf5' % (lib, version)), 'r')

    ngal = fmm['spec_flux_source'][...].shape[0]
    
    # plot BGS spectra and source spectra for sanity checks  
    fig = plt.figure(figsize=(15,15))
    for ii, i in enumerate(np.random.choice(np.arange(ngal), size=3, replace=False)): 
        sub = fig.add_subplot(3,1,ii+1)
        for band in ['b', 'r', 'z']: 
            sub.plot(fmm['spec_wave_%s_bgs' % band][...], fmm['spec_flux_%s_bgs' % band][...][0,i,:], c='C0') 
        sub.plot(fmm['spec_wave_source'][...],
                fmm['spec_fiber_flux_source'][...][i,:],
                c='k', ls='--') 
        sub.set_xlim(3.6e3, 9.8e3)
        sub.set_ylim(-2, 10)
    sub.set_xlabel('wavelength', fontsize=25) 
    fig.savefig(os.path.join(UT.dat_dir(), 'mini_mocha', 
        'lgal.mini_mocha.%s.v%s.png' % (lib, version)), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    fm_Lgal_fsps()
    #_mini_mocha_galid(lib='fsps')
    #fm_Lgal_mini_mocha(lib='fsps')
    #QA_fm_Lgal_mini_mocha()
