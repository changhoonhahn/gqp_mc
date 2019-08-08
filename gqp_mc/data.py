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
# --- astropy --- 
from astropy.io import fits
from astropy.table import Table
# --- gqp_mc ---
from . import util as UT 


def Spectra(sim='lgal', noise='none', lib='bc03', sample='spectral_challenge'): 
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
    if lib == 'bc03': 
        str_lib = 'BC03_Stelib'
    else: 
        raise NotImplementedError

    str_sample = ''
    if sample != '': 
        str_sample = '.%s' % sample

    # read in meta data 
    meta = pickle.load(open(os.path.join(UT.lgal_dir(), sample, "lgal.meta%s.p" % str_sample), 'rb')) 
    
    if noise == 'none': # noiseless source spectra
        f = h5py.File(os.path.join(UT.lgal_dir(), sample, 
            'lgal.spectra.%s.nonoise%s.hdf5' % (str_lib, str_sample)), 'r') 
    elif 'bgs' in noise: 
        iobs = int(noise.strip('bgs')) 
        f = h5py.File(os.path.join(UT.lgal_dir(), sample, 
            'lgal.spectra.%s.BGSnoise.obs%i%s.hdf5' % (str_lib, iobs, str_sample)), 'r') 
    else: 
        raise NotImplementedError('%s noise not implemented' % noise) 

    specs = {} 
    for k in f.keys(): 
        specs[k] = f[k][...]
    return specs, meta 


def Photometry(sim='lgal', noise='none', lib='bc03', sample='spectral_challenge'): 
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
    if sim != 'lgal': 
        raise NotImplementedError 
    
    if noise == 'none':
        str_noise = 'nonoise'
    elif noise == 'legacy': 
        str_noise = 'legacy_noise'
    else: 
        raise NotImplementedError

    if lib == 'bc03': 
        str_lib = 'BC03_Stelib'
    else: 
        raise NotImplementedError

    str_sample = ''
    if sample != '': 
        str_sample = '.%s' % sample

    # read in meta data 
    meta = pickle.load(open(os.path.join(UT.lgal_dir(), sample, "lgal.meta%s.p" % str_sample), 'rb')) 
    
    photo = {} 
    # read in photometry without dust 
    phot_nodust = np.loadtxt(os.path.join(UT.lgal_dir(), sample, 
        'lgal.photo.BC03_Stelib.nodust.%s%s.dat' % (str_noise, str_sample)), skiprows=1)
    for icol, band in enumerate(['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']): 
        photo['flux_nodust_%s' % band] = phot_nodust[:,icol+1]
        photo['ivar_nodust_%s' % band] = phot_nodust[:,icol+8]

    phot_dust = np.loadtxt(os.path.join(UT.lgal_dir(), sample,
        'lgal.photo.BC03_Stelib.dust.%s%s.dat' % (str_noise, str_sample)), skiprows=1) 
    for icol, band in enumerate(['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']): 
        photo['flux_dust_%s' % band] = phot_dust[:,icol+1]
        photo['ivar_dust_%s' % band] = phot_dust[:,icol+8]
    return photo, meta


def _make_Lgal_Photometry(): 
    '''
    '''
    # gather all galids 
    galids = [] 
    dir_inputs = os.path.join(UT.lgal_dir(), 'gal_inputs')
    for finput in glob.glob(dir_inputs+'/*'): 
        galids.append(int(os.path.basename(finput).split('_')[2]))
    n_id = len(galids) 
    return None 


def _make_Lgal_Spectra(): 
    '''
    '''
    # gather all galids 
    galids = [] 
    dir_inputs = os.path.join(UT.lgal_dir(), 'gal_inputs')
    for finput in glob.glob(dir_inputs+'/*'): 
        galids.append(int(os.path.basename(finput).split('_')[2]))
    n_id = len(galids) 

    # compile input meta data 
    tlookback, dt = [], [] 
    sfh_disk, sfh_bulge, Z_disk, Z_bulge, logM_disk, logM_bulge, logM_total = [], [], [], [], [], [], []
    for i, galid in enumerate(galids): 
        f_input = os.path.join(dir_inputs, 'gal_input_%i_BGS_template_FSPS_uvmiles.csv' % galid) 
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
    meta = {} 
    meta['t_lookback']  = tlookback
    meta['sfh_disk']    = sfh_disk
    meta['sfh_bulge']   = sfh_bulge
    meta['Z_disk']      = Z_disk
    meta['Z_bulge']     = Z_bulge
    meta['logM_disk']   = logM_disk
    meta['logM_bulge']  = logM_bulge
    meta['logM_total']  = logM_total
    
    # compile noiseless source spectra
    str_lib = 'BC03_Stelib'
    # file name of noiseless source spectra
    _Fsource = lambda galid: os.path.join(UT.lgal_dir(), 'templates', 
            'gal_spectrum_%i_BGS_template_%s.fits' % (galid, str_lib)) 
     
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
        if i == 0: 
            wave = np.zeros((n_id, len(specin['wave'])))
            flux_dust = np.zeros((n_id, len(specin['wave'])))
            flux_nodust = np.zeros((n_id, len(specin['wave'])))
        wave[i,:] = specin['wave'] 
        flux_dust[i,:] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A
        flux_nodust[i,:] = specin['flux_nodust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A

    f = h5py.File(os.path.join(UT.lgal_dir(), 'Lgal_spectra_nonoise.%s.hdf5' % str_lib), 'w') 
    f.create_dataset('wavelength', data=wave) 
    f.create_dataset('flux_dust', data=flux_dust) 
    f.create_dataset('flux_nodust', data=flux_nodust) 
    f.close() 
    meta['redshift']    = redshift
    meta['cosi']        = cosi 
    meta['tau_ism']     = tau_ism
    meta['tau_bc']      = tau_bc
    meta['vd_disk']     = vd_disk 
    meta['vd_bulge']    = vd_bulge
    pickle.dump(meta, open(os.path.join(UT.lgal_dir(), "lgal_meta.p"), 'wb')) 

    # compile DESI BGS-like spectra (this will be updated!) 
    '''
    dir_bgs = '/Users/ChangHoon/data/FOMOspec/spectral_challenge/bgs/'
    for iobs in range(1,9): 
        for i, galid in enumerate(galids):
            fbgs_nodust = os.path.join(dir_bgs, 
                    'BGSsim.%s.nodust.obscond_spacefill.obs%i.fits' % 
                    (os.path.splitext(os.path.basename(_Fsource(galid)))[0], iobs))
            bgs_nodust = UT.readDESIspec(fbgs_nodust) 

            fbgs_dust = os.path.join(dir_bgs, 
                    'BGSsim.%s.dust.obscond_spacefill.obs%i.fits' % 
                    (os.path.splitext(os.path.basename(_Fsource(galid)))[0], iobs))
            bgs_dust = UT.readDESIspec(fbgs_nodust) 
            
            assert np.array_equal(bgs_nodust['wave_b'], bgs_dust['wave_b']) 

            if i == 0: 
                wave_b = bgs_nodust['wave_b']
                wave_r = bgs_nodust['wave_r']
                wave_z = bgs_nodust['wave_z']

                flux_nodust_b = np.zeros((n_id, len(wave_b)))
                flux_nodust_r = np.zeros((n_id, len(wave_r)))
                flux_nodust_z = np.zeros((n_id, len(wave_z)))

                ivar_nodust_b = np.zeros((n_id, len(wave_b)))
                ivar_nodust_r = np.zeros((n_id, len(wave_r)))
                ivar_nodust_z = np.zeros((n_id, len(wave_z)))

                flux_dust_b = np.zeros((n_id, len(wave_b)))
                flux_dust_r = np.zeros((n_id, len(wave_r)))
                flux_dust_z = np.zeros((n_id, len(wave_z)))

                ivar_dust_b = np.zeros((n_id, len(wave_b)))
                ivar_dust_r = np.zeros((n_id, len(wave_r)))
                ivar_dust_z = np.zeros((n_id, len(wave_z)))

            flux_nodust_b[i,:] = bgs_nodust['flux_b']
            flux_nodust_r[i,:] = bgs_nodust['flux_r']
            flux_nodust_z[i,:] = bgs_nodust['flux_z']

            ivar_nodust_b[i,:] = bgs_nodust['ivar_b']
            ivar_nodust_r[i,:] = bgs_nodust['ivar_r']
            ivar_nodust_z[i,:] = bgs_nodust['ivar_z']

            flux_dust_b[i,:] = bgs_dust['flux_b']
            flux_dust_r[i,:] = bgs_dust['flux_r']
            flux_dust_z[i,:] = bgs_dust['flux_z']

            ivar_dust_b[i,:] = bgs_dust['ivar_b']
            ivar_dust_r[i,:] = bgs_dust['ivar_r']
            ivar_dust_z[i,:] = bgs_dust['ivar_z']

        f = h5py.File(os.path.join(UT.lgal_dir(), 'Lgal_spectra_BGSnoise.obs%i.%s.hdf5' % (ibos, str_lib)), 'w') 
        f.create_dataset('wave_b', data=wave_b) 
        f.create_dataset('wave_r', data=wave_r) 
        f.create_dataset('wave_z', data=wave_z) 
        f.create_dataset('flux_nodust_b', data=flux_nodust_b) 
        f.create_dataset('flux_nodust_r', data=flux_nodust_r) 
        f.create_dataset('flux_nodust_z', data=flux_nodust_z) 
        
        f.create_dataset('ivar_nodust_b', data=ivar_nodust_b) 
        f.create_dataset('ivar_nodust_r', data=ivar_nodust_r) 
        f.create_dataset('ivar_nodust_z', data=ivar_nodust_z) 
        
        f.create_dataset('flux_dust_b', data=flux_dust_b) 
        f.create_dataset('flux_dust_r', data=flux_dust_r) 
        f.create_dataset('flux_dust_z', data=flux_dust_z) 

        f.create_dataset('ivar_dust_b', data=ivar_dust_b) 
        f.create_dataset('ivar_dust_r', data=ivar_dust_r) 
        f.create_dataset('ivar_dust_z', data=ivar_dust_z) 
        f.close() 
    '''
    return None 


def _make_Lgal_Spectra_SpectralChallenge(): 
    '''
    '''
    # gather all galids 
    _galids = [] 
    dir_inputs = os.path.join(UT.lgal_dir(), 'gal_inputs')
    finputs = np.loadtxt(os.path.join(UT.lgal_dir(), 'spectral_challenge', 
        'galids.spectral_challenge.txt'), skiprows=1, unpack=True, dtype='S')  
    for finput in finputs: 
        _galids.append(int(os.path.basename(str(finput)).split('_')[2]))
    galids = np.unique(_galids) 
    n_id = len(galids) 
    print('%i unique galaxies in the spectral challenge' % n_id) 

    # compile input meta data 
    tlookback, dt = [], [] 
    sfh_disk, sfh_bulge, Z_disk, Z_bulge, logM_disk, logM_bulge, logM_total = [], [], [], [], [], [], []
    t_age_MW, Z_MW = [], [] 
    for i, galid in enumerate(galids): 
        f_input = os.path.join(dir_inputs, 'gal_input_%i_BGS_template_FSPS_uvmiles.csv' % galid) 
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
    meta['galid']       = np.array(galids) 
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

    # compile noiseless source spectra
    str_lib = 'BC03_Stelib'
    # file name of noiseless source spectra
    _Fsource = lambda galid: os.path.join(UT.lgal_dir(), 'templates', 
            'gal_spectrum_%i_BGS_template_%s.fits' % (galid, str_lib)) 
     
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
        if i == 0: 
            wave = np.zeros((n_id, len(specin['wave'])))
            flux_dust = np.zeros((n_id, len(specin['wave'])))
            flux_nodust = np.zeros((n_id, len(specin['wave'])))
        wave[i,:] = specin['wave'] 
        flux_dust[i,:] = specin['flux_dust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A
        flux_nodust[i,:] = specin['flux_nodust_nonoise'] * 1e-4 * 1e7 *1e17 #from W/A/m2 to 10e-17 erg/s/cm2/A

    f = h5py.File(os.path.join(UT.lgal_dir(), 'spectral_challenge', 
        'lgal.spectra.%s.nonoise.spectral_challenge.hdf5' % str_lib), 'w') 
    f.create_dataset('wavelength', data=wave) 
    f.create_dataset('flux_dust', data=flux_dust) 
    f.create_dataset('flux_nodust', data=flux_nodust) 
    f.close() 
    meta['redshift']    = redshift
    meta['cosi']        = cosi 
    meta['tau_ism']     = tau_ism
    meta['tau_bc']      = tau_bc
    meta['vd_disk']     = vd_disk 
    meta['vd_bulge']    = vd_bulge
    pickle.dump(meta, open(os.path.join(UT.lgal_dir(), 'spectral_challenge', 'lgal.meta.spectral_challenge.p'), 'wb')) 

    # compile DESI BGS-like spectra (this will be updated!) 
    dir_bgs = '/Users/ChangHoon/data/FOMOspec/spectral_challenge/bgs/'
    for iobs in range(1,9): 
        for i, galid in enumerate(galids):
            fbgs_nodust = os.path.join(dir_bgs, 
                    'BGSsim.%s.nodust.obscond_spacefill.obs%i.fits' % 
                    (os.path.splitext(os.path.basename(_Fsource(galid)))[0], iobs))
            bgs_nodust = UT.readDESIspec(fbgs_nodust) 

            fbgs_dust = os.path.join(dir_bgs, 
                    'BGSsim.%s.dust.obscond_spacefill.obs%i.fits' % 
                    (os.path.splitext(os.path.basename(_Fsource(galid)))[0], iobs))
            bgs_dust = UT.readDESIspec(fbgs_nodust) 
            
            assert np.array_equal(bgs_nodust['wave_b'], bgs_dust['wave_b']) 

            if i == 0: 
                wave_b = bgs_nodust['wave_b']
                wave_r = bgs_nodust['wave_r']
                wave_z = bgs_nodust['wave_z']

                flux_nodust_b = np.zeros((n_id, len(wave_b)))
                flux_nodust_r = np.zeros((n_id, len(wave_r)))
                flux_nodust_z = np.zeros((n_id, len(wave_z)))

                ivar_nodust_b = np.zeros((n_id, len(wave_b)))
                ivar_nodust_r = np.zeros((n_id, len(wave_r)))
                ivar_nodust_z = np.zeros((n_id, len(wave_z)))

                flux_dust_b = np.zeros((n_id, len(wave_b)))
                flux_dust_r = np.zeros((n_id, len(wave_r)))
                flux_dust_z = np.zeros((n_id, len(wave_z)))

                ivar_dust_b = np.zeros((n_id, len(wave_b)))
                ivar_dust_r = np.zeros((n_id, len(wave_r)))
                ivar_dust_z = np.zeros((n_id, len(wave_z)))

            flux_nodust_b[i,:] = bgs_nodust['flux_b']
            flux_nodust_r[i,:] = bgs_nodust['flux_r']
            flux_nodust_z[i,:] = bgs_nodust['flux_z']

            ivar_nodust_b[i,:] = bgs_nodust['ivar_b']
            ivar_nodust_r[i,:] = bgs_nodust['ivar_r']
            ivar_nodust_z[i,:] = bgs_nodust['ivar_z']

            flux_dust_b[i,:] = bgs_dust['flux_b']
            flux_dust_r[i,:] = bgs_dust['flux_r']
            flux_dust_z[i,:] = bgs_dust['flux_z']

            ivar_dust_b[i,:] = bgs_dust['ivar_b']
            ivar_dust_r[i,:] = bgs_dust['ivar_r']
            ivar_dust_z[i,:] = bgs_dust['ivar_z']

        f = h5py.File(os.path.join(UT.lgal_dir(), 'spectral_challenge', 
            'lgal.spectra.%s.BGSnoise.obs%i.spectral_challenge.hdf5' % (str_lib, iobs)), 'w') 
        f.create_dataset('wave_b', data=wave_b) 
        f.create_dataset('wave_r', data=wave_r) 
        f.create_dataset('wave_z', data=wave_z) 
        f.create_dataset('flux_nodust_b', data=flux_nodust_b) 
        f.create_dataset('flux_nodust_r', data=flux_nodust_r) 
        f.create_dataset('flux_nodust_z', data=flux_nodust_z) 
        
        f.create_dataset('ivar_nodust_b', data=ivar_nodust_b) 
        f.create_dataset('ivar_nodust_r', data=ivar_nodust_r) 
        f.create_dataset('ivar_nodust_z', data=ivar_nodust_z) 
        
        f.create_dataset('flux_dust_b', data=flux_dust_b) 
        f.create_dataset('flux_dust_r', data=flux_dust_r) 
        f.create_dataset('flux_dust_z', data=flux_dust_z) 

        f.create_dataset('ivar_dust_b', data=ivar_dust_b) 
        f.create_dataset('ivar_dust_r', data=ivar_dust_r) 
        f.create_dataset('ivar_dust_z', data=ivar_dust_z) 
        f.close() 
    return None 
