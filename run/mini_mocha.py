'''

scripts to generate data for the mini Mock Challenge (mini mocha) 


'''
import sys 
import os 
import h5py 
import numpy as np 
import corner as DFM 
from functools import partial
from multiprocessing.pool import Pool 
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
from gqp_mc import fitters as Fitters
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
try: 
    if os.environ['NERSC_HOST'] == 'cori':  pass
except KeyError: 
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


theta_dict = {
        'ifsps_vanilla': ['$\log M_*$', '$\log Z$', 'dust2', r'$\tau$'], 
        'ifsps_complexduxt': ['$\log M_*$', '$\log Z$', 'dust1', 'dust2', 'dust index', r'$\tau$'], 
        'ispeculator_emulator': [r'$\log M_*^{\rm fib}$', r'$\beta_1^{\rm SFH}$', r'$\beta_2^{\rm SFH}$', r'$\beta_3^{\rm SFH}$', r'$\beta_4^{\rm SFH}$', r'$\gamma_1^{\rm ZH}$', r'$\gamma_2^{\rm ZH}$', r'$\tau$']
        } 


def construct_sample(sim): 
    ''' construct the mini Mock Challenge photometry, spectroscopy 
    '''
    # construct photometry and spectroscopy 
    dir_run = os.path.dirname(os.path.realpath(__file__)) 
    if sim == 'lgal': 
        fscript = os.path.join(dir_run, 'fm_lgal.py')
    elif sim == 'tng': 
        fscript = os.path.join(dir_run, 'fm_tng.py')
    os.system('python %s' % fscript)
    return None 


def validate_sample(sim): 
    ''' generate some plots to validate the mini Mock Challenge photometry
    and spectroscopy 
    '''
    # read photometry 
    photo, _ = Data.Photometry(sim=sim, noise='legacy', sample='mini_mocha') 
    photo_g = 22.5 - 2.5 * np.log10(photo['flux'][:,0])
    photo_r = 22.5 - 2.5 * np.log10(photo['flux'][:,1])
    photo_z = 22.5 - 2.5 * np.log10(photo['flux'][:,2])
        
    flux_g = photo['flux'][:,0] * 1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs())
    flux_r = photo['flux'][:,1] * 1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs())
    flux_z = photo['flux'][:,2] * 1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()) # convert to 10^-17 ergs/s/cm^2/Ang
    ivar_g = photo['ivar'][:,0] * (1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_r = photo['ivar'][:,1] * (1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_z = photo['ivar'][:,2] * (1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()))**-2. # convert to 10^-17 ergs/s/cm^2/Ang
    
    # read BGS targets from imaging 
    bgs_targets = h5py.File(os.path.join(UT.dat_dir(), 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    n_targets = len(bgs_targets['ra'][...]) 
    
    bgs_g = 22.5 - 2.5 * np.log10(bgs_targets['flux_g'][...])[::10]
    bgs_r = 22.5 - 2.5 * np.log10(bgs_targets['flux_r'][...])[::10]
    bgs_z = 22.5 - 2.5 * np.log10(bgs_targets['flux_z'][...])[::10]
    
    # photometry validation 
    fig = plt.figure(figsize=(6,6)) 
    sub = fig.add_subplot(111)
    DFM.hist2d(bgs_g - bgs_r, bgs_r - bgs_z, color='k', levels=[0.68, 0.95], 
            range=[[-1., 3.], [-1., 3.]], bins=40, smooth=0.5, 
            plot_datapoints=True, fill_contours=False, plot_density=False, linewidth=0.5, 
            ax=sub) 
    sub.scatter([-100.], [-100], c='k', s=1, label='BGS targets')
    sub.scatter(photo_g - photo_r, photo_r - photo_z, c='C0', s=1, label='LGal photometry') 
    sub.legend(loc='upper left', handletextpad=0, markerscale=10, fontsize=20) 
    sub.set_xlabel('$g-r$', fontsize=25) 
    sub.set_xlim(-1., 3.) 
    sub.set_ylabel('$r-z$', fontsize=25) 
    sub.set_ylim(-1., 3.) 
    ffig = os.path.join(UT.dat_dir(), 
            'mini_mocha', 'mini_mocha.%s.photo.png' % sim)
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 

    # read spectra
    spec_s, _ = Data.Spectra(sim=sim, noise='none', sample='mini_mocha') 
    spec_bgs0, _ = Data.Spectra(sim=sim, noise='bgs0', sample='mini_mocha') 
    spec_bgs1, _ = Data.Spectra(sim=sim, noise='bgs1', sample='mini_mocha') 
    spec_bgs2, _ = Data.Spectra(sim=sim, noise='bgs2', sample='mini_mocha') 
        
    # read sky brightness
    _fsky = os.path.join(UT.dat_dir(), 'mini_mocha', 'bgs.exposure.surveysim.150s.v0p4.sample.hdf5') 
    fsky = h5py.File(_fsky, 'r') 
    wave_sky    = fsky['wave'][...] # sky wavelength 
    sbright_sky = fsky['sky'][...]
    
    # spectra validationg 
    fig = plt.figure(figsize=(10,5))     
    sub = fig.add_subplot(111)
    
    for i, spec_bgs in enumerate([spec_bgs2, spec_bgs0]): 
        wsort = np.argsort(spec_bgs['wave']) 
        if i == 0: 
            _plt, = sub.plot(spec_bgs['wave'][wsort], spec_bgs['flux'][0,wsort], c='C%i' % i, lw=0.25) 
        else:
            sub.plot(spec_bgs['wave'][wsort], spec_bgs['flux'][0,wsort], c='C%i' % i, lw=0.25) 
    
    _plt_photo = sub.errorbar([4750, 6350, 9250], [flux_g[0], flux_r[0], flux_z[0]], 
            [ivar_g[0]**-0.5, ivar_r[0]**-0.5, ivar_z[0]**-0.5], fmt='.r') 

    _plt_sim, = sub.plot(spec_s['wave'][0,:], spec_s['flux'][0,:], c='k', ls='-', lw=1) 
    _plt_sim0, = sub.plot(spec_s['wave'][0,:], spec_s['flux_unscaled'][0,:], c='k', ls=':', lw=0.25) 

    leg = sub.legend(
            [_plt_sim0, _plt_photo, _plt_sim, _plt], 
            ['%s spectrum' % sim.upper(), '%s photometry' % sim.upper(), 
                '%s fiber spectrum' % sim.upper(), '%s BGS spectra' % sim.upper()],
            loc='upper right', fontsize=17) 
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    sub.set_xlabel('Wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3e3, 1e4) 
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$', fontsize=20) 
    if sim == 'lgal': sub.set_ylim(-2., 8.) 
    elif sim == 'tng': sub.set_ylim(0., None) 
    
    ffig = os.path.join(UT.dat_dir(), 
            'mini_mocha', 'mini_mocha.%s.spectra.png' % sim)
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    
    return None 


def fit_photometry(igal, sim='lgal', noise='legacy', method='ifsps', 
        model='emulator', nwalkers=100, burnin=100, niter=1000,
        overwrite=False, justplot=False): 
    ''' Fit simulated photometry. `noise` specifies whether to fit spectra without noise or 
    with legacy-like noise. `dust` specifies whether to if spectra w/ dust or not. 
    Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 

    :param noise: 
        If 'none', fit noiseless photometry. 
        If 'legacy', fit Legacy-like photometry. (default: 'none') 

    :param justplot: 
        If True, skip the fitting and plot the best-fit. This is mainly implemented 
        because I'm having issues plotting in NERSC. (default: False) 
    '''
    # read Lgal photometry of the mini_mocha mocks 
    photo, meta = Data.Photometry(sim=sim, noise=noise, lib='bc03', sample='mini_mocha') 

    if meta['redshift'][igal] < 0.101: 
        # current Speculator wavelength doesn't extend far enough
        # current Speculator wavelength doesn't extend far enough
        # current Speculator wavelength doesn't extend far enough
        # current Speculator wavelength doesn't extend far enough
        # current Speculator wavelength doesn't extend far enough
        return None 

    if method == 'ispeculator':  
        photo_obs   = photo['flux'][igal,:3]
        ivar_obs    = photo['ivar'][igal,:3]
    else: 
        photo_obs   = photo['flux'][igal,:5]
        ivar_obs    = photo['ivar'][igal,:5]

    labels      = theta_dict['%s_%s' % (method, model)].copy() 
    truths      = [None for _ in labels] 
    truths[0]   = meta['logM_total'][igal] 

    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    
    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', method,  
            '%s.photo.noise_%s.%s.%i.hdf5' % (sim, noise, model, igal))
    
    # initiating fit
    if method == 'ifsps': 
        ifitter = Fitters.iFSPS(model_name=model) 
    elif method == 'ispeculator': 
        ifitter = Fitters.iSpeculator(model_name=model) 

    print('--- bestfit ---') 
    if (justplot or not overwrite) and os.path.isfile(f_bf): 
        # read in best-fit file with mcmc chain
        print('    reading in %s' % f_bf) 
        fbestfit = h5py.File(f_bf, 'r')  
        bestfit = {} 
        for k in fbestfit.keys(): 
            bestfit[k] = fbestfit[k][...]
        fbestfit.close() 
    else: 
        if os.path.isfile(f_bf) and not overwrite: 
            print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
        
        print('    writing %s' % f_bf) 
        prior = ifitter._default_prior(f_fiber_prior=None)

        bestfit = ifitter.MCMC_photo(
                photo_obs, 
                ivar_obs,
                meta['redshift'][igal], 
                bands='desi', 
                prior=prior, 
                nwalkers=nwalkers, 
                burnin=burnin, 
                niter=niter, 
                writeout=f_bf,
                silent=False)

    print('log M* total = %f' % bestfit['theta_med'][0])
    print('---------------') 
    if 'logsfr.100' not in bestfit['theta_names'].astype(str): 
        print('    appending logsfr.100 to markov chain') 
        # calculate sfr_100myr for MC chain add it in 
        bestfit['mcmc_chain'] = ifitter.add_logSFR_to_chain(bestfit['mcmc_chain'],
                meta['redshift'][igal], dt=0.1) 
        bestfit['prior_range'] = np.concatenate([bestfit['prior_range'],
            np.array([[-4., 4]])], axis=0) 
        bestfit['theta_names'] = np.array(list(bestfit['theta_names'].astype(str)) + ['logsfr.100'],
                dtype='S') 
        
        fbestfit = h5py.File(f_bf, 'w')  
        for k in bestfit.keys(): 
            fbestfit[k] = bestfit[k] 
        fbestfit.close() 

    truths += [np.log10(meta['sfr_100myr'][igal])]
    labels += [r'$\log {\rm SFR}_{\rm 100 Myr}$'] 
        
    try: 
        # plotting on nersc never works.
        if os.environ['NERSC_HOST'] == 'cori': return None 
    except KeyError: 
        fig = DFM.corner(bestfit['mcmc_chain'], range=bestfit['prior_range'], quantiles=[0.16, 0.5, 0.84], 
                levels=[0.68, 0.95], nbin=40, smooth=True, 
                truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
        fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
        plt.close() 
        
        fig = plt.figure(figsize=(5,3))
        sub = fig.add_subplot(111)
        sub.errorbar(np.arange(len(photo_obs)), photo_obs,
                yerr=ivar_obs**-0.5, fmt='.k', label='data')
        sub.scatter(np.arange(len(photo_obs)), bestfit['flux_photo_model'], c='C1',
                label='model') 
        sub.legend(loc='upper left', markerscale=3, handletextpad=0.2, fontsize=15) 
        sub.set_xticks([0, 1, 2]) 
        sub.set_xticklabels(['$g$', '$r$', '$z$']) 
        sub.set_xlim(-0.5, len(photo_obs)-0.5)
        fig.savefig(f_bf.replace('.hdf5', '.bestfit.png'), bbox_inches='tight') 
        plt.close()
    return None 


def fit_spectrophotometry(igal, sim='lgal', noise='bgs0_legacy',
        method='ifsps', model='emulator', nwalkers=100, burnin=100, niter=1000,
        overwrite=False, justplot=False):  
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without noise or 
    with BGS-like noise. Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 
    :param noise: 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 
    :param justplot: 
        If True, skip the fitting and plot the best-fit. This is mainly implemented 
        because I'm having issues plotting in NERSC. (default: False) 
    '''
    noise_spec = noise.split('_')[0]
    noise_photo = noise.split('_')[1]
    # read noiseless Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim=sim, noise=noise_spec, lib='bc03', sample='mini_mocha') 
    # read Lgal photometry of the mini_mocha mocks 
    photo, _ = Data.Photometry(sim=sim, noise=noise_photo, lib='bc03', sample='mini_mocha') 
    
    if meta['redshift'][igal] < 0.101: 
        # current Speculator wavelength doesn't extend far enough
        # current Speculator wavelength doesn't extend far enough
        # current Speculator wavelength doesn't extend far enough
        # current Speculator wavelength doesn't extend far enough
        # current Speculator wavelength doesn't extend far enough
        return None 

    w_obs       = specs['wave']
    flux_obs    = specs['flux'][igal]
    ivar_obs    = specs['ivar'][igal]
    if method == 'ispeculator': 
        photo_obs       = photo['flux'][igal,:3]
        photo_ivar_obs  = photo['ivar'][igal,:3]
    else: 
        photo_obs       = photo['flux'][igal,:5]
        photo_ivar_obs  = photo['ivar'][igal,:5]

    # get fiber flux factor prior range based on measured fiber flux 
    f_fiber_true = (photo['fiberflux_r_meas'][igal]/photo['flux_r_true'][igal]) 
    f_fiber_min = (photo['fiberflux_r_meas'][igal] - 3.*photo['fiberflux_r_ivar'][igal]**-0.5)/photo['flux'][igal,1]
    f_fiber_max = (photo['fiberflux_r_meas'][igal] + 3.*photo['fiberflux_r_ivar'][igal]**-0.5)/photo['flux'][igal,1]
    f_fiber_prior = [f_fiber_min, f_fiber_max]

    labels      = theta_dict['%s_%s' % (method, model)].copy() 
    labels      += [r'$f_{\rm fiber}$']
    truths      = [None for _ in labels] 
    truths[0]   = meta['logM_total'][igal] 
    truths[-1]  = f_fiber_true
    print(truths)

    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('log M* fiber = %f' % meta['logM_fiber'][igal])
    print('log SFR = %f' % np.log10(meta['sfr_100myr'][igal])) 

    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', method, 
            '%s.specphoto.noise_%s.%s.%i.hdf5' % (sim, noise, model, igal))
    
    if method == 'ifsps': 
        ifitter = Fitters.iFSPS(model_name=model) 
    elif method == 'ispeculator': 
        ifitter = Fitters.iSpeculator(model_name=model) 

    print('--- bestfit ---') 
    if (justplot or not overwrite) and os.path.isfile(f_bf): 
        print('     reading... %s' % os.path.basename(f_bf)) 
        # read in best-fit file with mcmc chain
        fbestfit = h5py.File(f_bf, 'r')  
        bestfit = {} 
        for k in fbestfit.keys(): 
            bestfit[k] = fbestfit[k][...]
        fbestfit.close() 
    else: 
        if os.path.isfile(f_bf) and not overwrite: 
            print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
        print('     writing... %s' % os.path.basename(f_bf)) 
        # initiating fit
        prior = ifitter._default_prior(f_fiber_prior=f_fiber_prior)

        bestfit = ifitter.MCMC_spectrophoto(
                w_obs, 
                flux_obs, 
                ivar_obs, 
                photo_obs, 
                photo_ivar_obs, 
                meta['redshift'][igal], 
                mask='emline', 
                prior=prior, 
                nwalkers=nwalkers, 
                burnin=burnin, 
                niter=niter, 
                writeout=f_bf,
                silent=False)
    print('log M* total = %f' % bestfit['theta_med'][0])
    print('log M* fiber = %f' % (bestfit['theta_med'][0] + np.log10(bestfit['theta_med'][-1])))
    
    if 'logsfr.100' not in bestfit['theta_names'].astype(str): 
        print('    appending logsfr.100 to markov chain') 
        # calculate sfr_100myr for MC chain add it in 
        bestfit['mcmc_chain'] = ifitter.add_logSFR_to_chain(bestfit['mcmc_chain'],
                meta['redshift'][igal], dt=0.1) 
        bestfit['prior_range'] = np.concatenate([bestfit['prior_range'],
            np.array([[-4., 4]])], axis=0) 
        bestfit['theta_names'] = np.array(list(bestfit['theta_names'].astype(str)) + ['logsfr.100'],
                dtype='S') 
        
        fbestfit = h5py.File(f_bf, 'w')  
        for k in bestfit.keys(): 
            fbestfit[k] = bestfit[k] 
        fbestfit.close() 

    truths += [np.log10(meta['sfr_100myr'][igal])]
    labels += [r'$\log {\rm SFR}_{\rm 100 Myr}$'] 
    print('log SFR = %f' % (np.median(bestfit['mcmc_chain'][:,-1])))
    print('---------------') 
    try: 
        # plotting on nersc never works.
        if os.environ['NERSC_HOST'] == 'cori': return None 
    except KeyError: 
        # corner plot of the posteriors 
        fig = DFM.corner(bestfit['mcmc_chain'], range=bestfit['prior_range'], quantiles=[0.16, 0.5, 0.84], 
                levels=[0.68, 0.95], nbin=40, smooth=True, 
                truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
        fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
        plt.close() 
        
        fig = plt.figure(figsize=(18,5))
        gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,3], wspace=0.2) 
        sub = plt.subplot(gs[0]) 
        sub.errorbar(np.arange(len(photo_obs)), photo_obs,
                yerr=photo_ivar_obs**-0.5, fmt='.k', label='data')
        sub.scatter(np.arange(len(photo_obs)), bestfit['flux_photo_model'], c='C1',
                label='model') 
        sub.legend(loc='upper left', markerscale=3, handletextpad=0.2, fontsize=15) 
        sub.set_xticks([0, 1, 2, 3, 4]) 
        sub.set_xticklabels(['$g$', '$r$', '$z$', 'W1', 'W2']) 
        sub.set_xlim(-0.5, len(photo_obs)-0.5)

        sub = plt.subplot(gs[1]) 
        sub.plot(w_obs, flux_obs, c='k', lw=1, label='data')
        sub.plot(bestfit['wavelength_model'], bestfit['flux_spec_model'], c='C1',
                ls='--', lw=1, label=method) 
        sub.legend(loc='upper right', fontsize=15) 
        sub.set_xlabel('wavelength [$A$]', fontsize=20) 
        sub.set_xlim(3600., 9800.)

        fig.savefig(f_bf.replace('.hdf5', '.bestfit.png'), bbox_inches='tight') 
        plt.close()
    return None 


def MP_sed_fit(spec_or_photo, igals, sim='lgal', noise='none', method='ifsps', 
        model='emulator', nthreads=1, nwalkers=100, burnin=100, niter=1000,
        overwrite=False, justplot=False): 
    ''' multiprocessing wrapepr for fit_spectra and fit_photometry. This does *not* parallelize 
    the MCMC sampling of individual fits but rather runs multiple fits simultaneously. 
    
    :param spec_or_photo: 
        fit spectra or photometry 

    :param igals: 
        array/list of spectral_challenge galaxy indices

    :param noise: 
        If 'none', fit noiseless spectra. 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 

    :param dust: 
        If True, fit the spectra w/ dust using a model with dust 
        If False, fit the spectra w/o dust using a model without dust. 
        (default: False) 

    :param nthreads: 
        Number of threads. If nthreads == 1, just runs fit_spectra
    '''
    args = igals # galaxy indices 

    kwargs = {
            'sim': sim, 
            'noise': noise, 
            'method': method, 
            'model': model,
            'nwalkers': nwalkers,
            'burnin': burnin,
            'niter': niter, 
            'overwrite': overwrite, 
            'justplot': justplot
            }
    if spec_or_photo == 'spec': 
        fit_func = fit_spectra
    elif spec_or_photo == 'photo': 
        fit_func = fit_photometry
    elif spec_or_photo == 'specphoto': 
        fit_func = fit_spectrophotometry

    if nthreads > 1: 
        pool = Pool(processes=nthreads) 
        pool.map(partial(fit_func, **kwargs), args)
        pool.close()
        pool.terminate()
        pool.join()
    else: 
        # single thread, loop over 
        for igal in args: fit_func(igal, **kwargs)
    return None 

# --- pseudoFirefly --- 
def fit_pFF_spectra(igal, noise='none', iter_max=10, overwrite=False): 
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without noise or 
    with BGS-like noise.  

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 

    :param noise: 
        If 'none', fit noiseless spectra. 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 

    :param justplot: 
        If True, skip the fitting and plot the best-fit. This is mainly implemented 
        because I'm having issues plotting in NERSC. (default: False) 
    '''
    # read noiseless Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim='lgal', noise=noise, lib='bc03', sample='mini_mocha') 

    model       = 'vanilla'
    w_obs       = specs['wave']
    flux_obs    = specs['flux'][igal]
    if noise != 'none': ivar_obs = specs['ivar'][igal]
    truths      = [meta['logM_fiber'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal]]
    labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$']

    if noise == 'none': # no noise 
        ivar_obs = np.ones(len(w_obs)) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('log M* fiber = %f' % meta['logM_fiber'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 

    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'pff', 'lgal.spec.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    if os.path.isfile(f_bf): 
        if not overwrite: 
            print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
    # initiating fit
    pff = Fitters.pseudoFirefly(model_name=model, prior=None) 
    bestfit = pff.Fit_spec(
            w_obs, 
            flux_obs, 
            ivar_obs, 
            meta['redshift'][igal], 
            mask='emline', 
            fit_cap=1000, 
            iter_max=10, 
            writeout=f_bf,
            silent=False)
    print('--- bestfit ---') 
    print('written to %s' % f_bf) 
    print('log M* = %f' % bestfit['theta_med'][0])
    print('log Z = %f' % bestfit['theta_med'][1]) 
    print('---------------') 
    
    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(111)
    sub.plot(bestfit['wavelength_data'], bestfit['flux_data'], c='k', zorder=1, lw=0.5) 
    sub.plot(bestfit['wavelength_model'] * (1. + bestfit['redshift']), bestfit['flux_model'], c='C1') 
    sub.set_xlabel('Wavelength', fontsize=20) 
    sub.set_xlim(3500, 1e4) 
    sub.set_ylabel('Flux', fontsize=20) 
    sub.set_ylim(0., None) 
    fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


"""
    def fit_spectra(igal, sim='lgal', noise='bgs0', method='ifsps', model='emulator', nwalkers=100,
            burnin=100, niter=1000, overwrite=False, justplot=False):
        ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without noise or 
        with BGS-like noise. Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

        :param igal: 
            index of Lgal galaxy within the spectral_challenge 
        :param noise: 
            If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 
        :param justplot: 
            If True, skip the fitting and plot the best-fit. This is mainly implemented 
            because I'm having issues plotting in NERSC. (default: False) 
        '''
        # read noiseless Lgal spectra of the spectral_challenge mocks 
        specs, meta = Data.Spectra(sim=sim, noise=noise, lib='bc03', sample='mini_mocha') 
        
        if meta['redshift'][igal] < 0.1: 
            # current Speculator wavelength doesn't extend far enough
            # current Speculator wavelength doesn't extend far enough
            # current Speculator wavelength doesn't extend far enough
            # current Speculator wavelength doesn't extend far enough
            # current Speculator wavelength doesn't extend far enough
            return None 

        w_obs       = specs['wave']
        flux_obs    = specs['flux'][igal]
        ivar_obs    = specs['ivar'][igal]

        labels      = theta_dict['%s_%s' % (method, model)]
        truths      = [None for _ in labels] 
        truths[0]   = meta['logM_fiber'][igal] 

        print('--- input ---') 
        print('z = %f' % meta['redshift'][igal])
        print('log M* total = %f' % meta['logM_total'][igal])
        print('log M* fiber = %f' % meta['logM_fiber'][igal])

        f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', method, 
                '%s.spec.noise_%s.%s.%i.hdf5' % (sim, noise, model, igal))
        
        # initiate fitter
        if method == 'ifsps': 
            ifitter = Fitters.iFSPS(model_name=model) 
        else: 
            ifitter = Fitters.iSpeculator(model_name=model) 
        
        if (justplot or not overwrite) and os.path.isfile(f_bf): 
            print('     reading... %s' % os.path.basename(f_bf)) 
            # read in best-fit file with mcmc chain
            fbestfit = h5py.File(f_bf, 'r')  
            bestfit = {} 
            for k in fbestfit.keys(): 
                bestfit[k] = fbestfit[k][...]
            fbestfit.close() 
        else: 
            if os.path.isfile(f_bf) and not overwrite: 
                print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
            prior = ifitter._default_prior(f_fiber_prior=None)

            bestfit = ifitter.MCMC_spec(
                    w_obs, 
                    flux_obs, 
                    ivar_obs, 
                    meta['redshift'][igal], 
                    mask='emline', 
                    prior=prior, 
                    nwalkers=nwalkers, 
                    burnin=burnin, 
                    niter=niter, 
                    writeout=f_bf,
                    silent=False)

        print('--- bestfit ---') 
        print('written to %s' % f_bf) 
        print('log M* = %f' % bestfit['theta_med'][0])
        print('---------------') 

        if bestfit['mcmc_chain'].shape[1] == len(truths): 
            # calculate sfr_100myr for MC chain add it in 
            bestfit['mcmc_chain'] = ifitter.add_logSFR_to_chain(bestfit['mcmc_chain'],
                    meta['redshift'][igal], dt=0.1) 
            bestfit['prior_range'] = np.concatenate([bestfit['prior_range'],
                np.array([[-4., 4]])], axis=0) 
            
            fbestfit = h5py.File(f_bf, 'w')  
            for k in bestfit.keys(): 
                fbestfit[k] = bestfit[k] 
            fbestfit.close() 

        truths += [np.log10(meta['sfr_100myr'][igal])]
        labels += [r'$\log {\rm SFR}_{\rm 100 Myr}$'] 

        try: 
            # plotting on nersc never works.
            if os.environ['NERSC_HOST'] == 'cori': return None 
        except KeyError: 
            # corner plot of the posteriors 
            fig = DFM.corner(bestfit['mcmc_chain'], range=bestfit['prior_range'], quantiles=[0.16, 0.5, 0.84], 
                    levels=[0.68, 0.95], nbin=40, smooth=True, 
                    truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
            fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
            plt.close()
            
            fig = plt.figure(figsize=(10,5))
            sub = fig.add_subplot(111)
            sub.plot(w_obs, flux_obs, c='k', lw=1, label='data')
            sub.plot(bestfit['wavelength_model'], bestfit['flux_spec_model'], c='C1',
                    ls='--', lw=1, label='model') 
            sub.legend(loc='upper right', fontsize=15) 
            sub.set_xlabel('wavelength [$A$]', fontsize=20) 
            sub.set_xlim(3600., 9800.)
            fig.savefig(f_bf.replace('.hdf5', '.bestfit.png'), bbox_inches='tight') 
            plt.close()
        return None 
"""

if __name__=="__main__": 
    # >>> python mini_mocha.py
    spec_or_photo   = sys.argv[1]
    sim             = sys.argv[2] 

    if spec_or_photo == 'construct':  
        construct_sample(sim)
        validate_sample(sim)
        sys.exit() 

    igal0           = int(sys.argv[3]) 
    igal1           = int(sys.argv[4]) 
    noise           = sys.argv[5]
    method          = sys.argv[6]
    model           = sys.argv[7]
    nthreads        = int(sys.argv[8]) 
    nwalkers        = int(sys.argv[9]) 
    burnin          = int(sys.argv[10]) 
    niter           = int(sys.argv[11]) 
    overwrite       = sys.argv[12] == 'True'

    # if specified, it assumes the chains already exist and just makes the 
    # corner plots (implemented because I have difficult making plots on nersc)
    try: 
        justplot = sys.argv[13] == 'True'
    except IndexError: 
        justplot = False

    if method not in ['ifsps', 'ispeculator']: 
        raise NotImplementedError
    print('----------------------------------------') 
    print('%s fitting %s of mini_mocha galaxies %i to %i' % (method, spec_or_photo, igal0, igal1))
    print('using %i threads' % nthreads) 
    igals = range(igal0, igal1+1)

    MP_sed_fit(spec_or_photo, igals, sim=sim, noise=noise, method=method, model=model, nthreads=nthreads, 
                    nwalkers=nwalkers, burnin=burnin, niter=niter, overwrite=overwrite, justplot=justplot)

    '''
        elif method == 'pfirefly': 
            raise NotImplementedError('need to update prior set up') 
            for igal in range(igal0, igal1+1):
                fit_pFF_spectra(igal, noise=noise, iter_max=10, overwrite=overwrite)
    '''
