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


def construct_sample(): 
    ''' construct the mini Mock Challenge photometry, spectroscopy 
    '''
    # select the galaxies of the mini mock challenge
    #Data._mini_mocha_galid() 
    # construct photometry and spectroscopy 
    Data.make_mini_mocha()  
    return None 


def validate_sample(): 
    ''' generate some plots to validate the mini Mock Challenge photometry
    and spectroscopy 
    '''
    # read photometry 
    photo, _ = Data.Photometry(noise='legacy', sample='mini_mocha') 
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
    ffig = os.path.join(UT.dat_dir(), 'mini_mocha', 'mini_mocha.photo.png')
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 

    # read spectra
    spec_s, _ = Data.Spectra(noise='none', sample='mini_mocha') 
    spec_bgs0, _ = Data.Spectra(noise='bgs0', sample='mini_mocha') 
    spec_bgs1, _ = Data.Spectra(noise='bgs1', sample='mini_mocha') 
    spec_bgs2, _ = Data.Spectra(noise='bgs2', sample='mini_mocha') 
        
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

    _plt_lgal, = sub.plot(spec_s['wave'], spec_s['flux'][0,:], c='k', ls='-', lw=1) 
    _plt_lgal0, = sub.plot(spec_s['wave'], spec_s['flux_unscaled'][0,:], c='k', ls=':', lw=0.25) 

    leg = sub.legend(
            [_plt_lgal0, _plt_photo, _plt_lgal, _plt], 
            ['LGal spectrum', 'LGal photometry', 'LGal fiber spectrum', 'LGal BGS spectra'],
            loc='upper right', fontsize=17) 
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    sub.set_xlabel('Wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3e3, 1e4) 
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$', fontsize=20) 
    sub.set_ylim(-2., 8.) 
    
    ffig = os.path.join(UT.dat_dir(), 'mini_mocha', 'mini_mocha.spectra.png')
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    
    return None 

# --- iFSPS --- 
def fit_iFSPS_spectra(igal, noise='none', nwalkers=100, burnin=100, niter=1000, overwrite=False, justplot=False): 
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without noise or 
    with BGS-like noise. Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

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
    truths      = [meta['logM_fiber'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None, None]
    labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', 'dust2', r'$\tau$']

    if noise == 'none': # no noise 
        ivar_obs = np.ones(len(w_obs)) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('log M* fiber = %f' % meta['logM_fiber'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 

    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ifsps', 'lgal.spec.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    if not justplot: 
        if os.path.isfile(f_bf): 
            if not overwrite: 
                print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
        # initiating fit
        ifsps = Fitters.iFSPS(model_name=model) 
        
        prior = ifsps._default_prior(f_fiber_prior=None)

        bestfit = ifsps.MCMC_spec(
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
    else: 
        # read in best-fit file with mcmc chain
        fbestfit = h5py.File(f_bf, 'r')  
        bestfit = {} 
        for k in fbestfit.keys(): 
            bestfit[k] = fbestfit[k][...]

    print('--- bestfit ---') 
    print('written to %s' % f_bf) 
    print('log M* = %f' % bestfit['theta_med'][0])
    print('log Z = %f' % bestfit['theta_med'][1]) 
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
    return None 


def fit_iFSPS_photometry(igal, noise='none', nwalkers=100, burnin=100, niter=1000, overwrite=False, justplot=False): 
    ''' Fit Lgal photometry. `noise` specifies whether to fit spectra without noise or 
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
    photo, meta = Data.Photometry(sim='lgal', noise=noise, lib='bc03', sample='mini_mocha') 
    
    model       = 'vanilla'
    photo_obs   = photo['flux'][igal,:5]
    if noise != 'none': ivar_obs = photo['ivar'][igal,:5]
    truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None, None]
    labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', 'dust2', r'$\tau$']

    if noise == 'none': # no noise 
        ivar_obs = np.ones(photo_obs.shape[0]) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 
    
    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ifsps', 'lgal.photo.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    if not justplot: 
        if os.path.isfile(f_bf): 
            if not overwrite: 
                print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
                return None 
        # initiate fitting
        ifsps = Fitters.iFSPS(model_name=model) 

        prior = ifsps._default_prior(f_fiber_prior=None)

        bestfit = ifsps.MCMC_photo(
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
    else: 
        # read in best-fit file with mcmc chain
        fbestfit = h5py.File(f_bf, 'r')  
        bestfit = {} 
        for k in fbestfit.keys(): 
            bestfit[k] = fbestfit[k][...]
    print('--- bestfit ---') 
    print('written to %s ---' % f_bf)
    print('log M* = %f' % bestfit['theta_med'][0])
    print('log Z = %f' % bestfit['theta_med'][1]) 
    print('---------------') 
    
    try: 
        # plotting on nersc never works.
        if os.environ['NERSC_HOST'] == 'cori': return None 
    except KeyError: 
        fig = DFM.corner(bestfit['mcmc_chain'], range=bestfit['prior_range'], quantiles=[0.16, 0.5, 0.84], 
                levels=[0.68, 0.95], nbin=40, smooth=True, 
                truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
        fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def fit_iFSPS_spectrophotometry(igal, noise='none', nwalkers=100, burnin=100, niter=1000, overwrite=False, justplot=False): 
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without noise or 
    with BGS-like noise. Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 

    :param noise: 
        If 'none', fit noiseless spectra. 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 

    :param justplot: 
        If True, skip the fitting and plot the best-fit. This is mainly implemented 
        because I'm having issues plotting in NERSC. (default: False) 
    '''
    if noise != 'none': 
        noise_spec = noise.split('_')[0]
        noise_photo = noise.split('_')[1]
    else: 
        noise_spec = 'none'
        noise_photo = 'none'
    # read noiseless Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim='lgal', noise=noise_spec, lib='bc03', sample='mini_mocha') 
    
    # read Lgal photometry of the mini_mocha mocks 
    photo, _ = Data.Photometry(sim='lgal', noise=noise_photo, lib='bc03', sample='mini_mocha') 

    model       = 'vanilla'
    w_obs       = specs['wave']
    flux_obs    = specs['flux'][igal]
    if noise_spec != 'none': 
        ivar_obs = specs['ivar'][igal]
    else:  
        ivar_obs = np.ones(len(w_obs)) 
    photo_obs   = photo['flux'][igal,:5]
    if noise_photo != 'none': 
        photo_ivar_obs = photo['ivar'][igal,:5]
    else:  
        photo_ivar_obs = np.ones(photo_obs.shape[0]) 

    # get fiber flux factor prior range based on measured fiber flux 
    f_fiber_true = (photo['fiberflux_r_meas'][igal]/photo['flux_r_true'][igal]) 
    f_fiber_min = (photo['fiberflux_r_meas'][igal] - 3.*photo['fiberflux_r_ivar'][igal]**-0.5)/photo['flux'][igal,1]
    f_fiber_max = (photo['fiberflux_r_meas'][igal] + 3.*photo['fiberflux_r_ivar'][igal]**-0.5)/photo['flux'][igal,1]
    f_fiber_prior = [f_fiber_min, f_fiber_max]
    print(f_fiber_prior) 
    print(f_fiber_true) 
    print((photo['fiberflux_r_true'][igal]/photo['flux_r_true'][igal])) 

    truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None, None, f_fiber_true]
    labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', 'dust2', r'$\tau$', r'$f_{\rm fiber}$']

    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('log M* fiber = %f' % meta['logM_fiber'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 

    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ifsps', 'lgal.specphoto.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    if not justplot: 
        if os.path.isfile(f_bf): 
            if not overwrite: 
                print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
        # initiating fit
        ifsps = Fitters.iFSPS(model_name=model) 

        prior = ifsps._default_prior(f_fiber_prior=f_fiber_prior)

        bestfit = ifsps.MCMC_spectrophoto(
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
    else: 
        # read in best-fit file with mcmc chain
        fbestfit = h5py.File(f_bf, 'r')  
        bestfit = {} 
        for k in fbestfit.keys(): 
            bestfit[k] = fbestfit[k][...]

    print('--- bestfit ---') 
    print('written to %s' % f_bf) 
    print('log M* total = %f' % bestfit['theta_med'][0])
    print('log M* fiber = %f' % (bestfit['theta_med'][0] + np.log10(bestfit['theta_med'][-1])))
    print('log Z = %f' % bestfit['theta_med'][1]) 
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
    return None 


def MP_fit_iFSPS(spec_or_photo, igals, noise='none', nthreads=1, nwalkers=100, burnin=100, niter=1000, overwrite=False, justplot=False): 
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
            'noise': noise, 
            'nwalkers': nwalkers,
            'burnin': burnin,
            'niter': niter, 
            'overwrite': overwrite, 
            'justplot': justplot
            }
    if spec_or_photo == 'spec': 
        fit_func = fit_iFSPS_spectra
    elif spec_or_photo == 'photo': 
        fit_func = fit_iFSPS_photometry
    elif spec_or_photo == 'specphoto': 
        fit_func = fit_iFSPS_spectrophotometry

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

# --- iSpeculator --- 
def fit_iSpeculator_spectra(igal, noise='none', nwalkers=100, burnin=100, niter=1000, overwrite=False, justplot=False): 
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without noise or 
    with BGS-like noise. Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

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
    truths      = [meta['logM_fiber'][igal], None, None, None, None, None, None, meta['t_age_MW'][igal], None]
    labels      = [r'$\log M_*^{\rm fib}$', r'$\beta_1^{\rm SFH}$', r'$\beta_2^{\rm SFH}$', r'$\beta_3^{\rm SFH}$', r'$\beta_4^{\rm SFH}$', r'$\gamma_1^{\rm ZH}$', r'$\gamma_2^{\rm ZH}$', r'$t_{\rm age}$', r'$\tau$']

    if noise == 'none': # no noise 
        ivar_obs = np.ones(len(w_obs)) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('log M* fiber = %f' % meta['logM_fiber'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 

    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ispeculator', 
            'lgal.spec.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    if not justplot: 
        if os.path.isfile(f_bf): 
            if not overwrite: 
                print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
        # initiating fit
        ispeculator = Fitters.iSpeculator(model_name=model) 
        
        prior = ispeculator._default_prior(f_fiber_prior=None)

        bestfit = ispeculator.MCMC_spec(
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
    else: 
        # read in best-fit file with mcmc chain
        fbestfit = h5py.File(f_bf, 'r')  
        bestfit = {} 
        for k in fbestfit.keys(): 
            bestfit[k] = fbestfit[k][...]

    print('--- bestfit ---') 
    print('written to %s' % f_bf) 
    print('log M* = %f' % bestfit['theta_med'][0])
    print('log Z = %f' % bestfit['theta_med'][1]) 
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
    return None 


def fit_iSpeculator_photometry(igal, noise='none', nwalkers=100, burnin=100, niter=1000, overwrite=False, justplot=False): 
    ''' Fit Lgal photometry. `noise` specifies whether to fit spectra without noise or 
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
    photo, meta = Data.Photometry(sim='lgal', noise=noise, lib='bc03', sample='mini_mocha') 
    
    model       = 'vanilla'
    photo_obs   = photo['flux'][igal,:3]
    if noise != 'none': ivar_obs = photo['ivar'][igal,:3]
    truths      = [meta['logM_total'][igal], None, None, None, None, None, None, meta['t_age_MW'][igal], None]
    labels      = ['$\log M_*$', r'$\beta_1^{\rm SFH}$', r'$\beta_2^{\rm SFH}$', r'$\beta_3^{\rm SFH}$', r'$\beta_4^{\rm SFH}$', r'$\gamma_1^{\rm ZH}$', r'$\gamma_2^{\rm ZH}$', r'$t_{\rm age}$', r'$\tau$']
    if noise == 'none': # no noise 
        ivar_obs = np.ones(photo_obs.shape[0]) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 
    
    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ispeculator', 
            'lgal.photo.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    if not justplot: 
        if os.path.isfile(f_bf): 
            if not overwrite: 
                print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
                return None 
        # initiate fitting
        ispeculator = Fitters.iSpeculator(model_name=model) 

        prior = ispeculator._default_prior(f_fiber_prior=None)

        bestfit = ispeculator.MCMC_photo(
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
    else: 
        # read in best-fit file with mcmc chain
        fbestfit = h5py.File(f_bf, 'r')  
        bestfit = {} 
        for k in fbestfit.keys(): 
            bestfit[k] = fbestfit[k][...]
    print('--- bestfit ---') 
    print('written to %s ---' % f_bf)
    print('log M* = %f' % bestfit['theta_med'][0])
    print('log Z = %f' % bestfit['theta_med'][1]) 
    print('---------------') 
    
    try: 
        # plotting on nersc never works.
        if os.environ['NERSC_HOST'] == 'cori': return None 
    except KeyError: 
        fig = DFM.corner(bestfit['mcmc_chain'], range=bestfit['prior_range'], quantiles=[0.16, 0.5, 0.84], 
                levels=[0.68, 0.95], nbin=40, smooth=True, 
                truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
        fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def fit_iSpeculator_spectrophotometry(igal, noise='none', nwalkers=100, burnin=100, niter=1000, overwrite=False, justplot=False): 
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without noise or 
    with BGS-like noise. Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 

    :param noise: 
        If 'none', fit noiseless spectra. 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 

    :param justplot: 
        If True, skip the fitting and plot the best-fit. This is mainly implemented 
        because I'm having issues plotting in NERSC. (default: False) 
    '''
    if noise != 'none': 
        noise_spec = noise.split('_')[0]
        noise_photo = noise.split('_')[1]
    else: 
        noise_spec = 'none'
        noise_photo = 'none'
    # read noiseless Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim='lgal', noise=noise_spec, lib='bc03', sample='mini_mocha') 
    
    # read Lgal photometry of the mini_mocha mocks 
    photo, _ = Data.Photometry(sim='lgal', noise=noise_photo, lib='bc03', sample='mini_mocha') 

    model       = 'vanilla'
    w_obs       = specs['wave']
    flux_obs    = specs['flux'][igal]
    if noise_spec != 'none': 
        ivar_obs = specs['ivar'][igal]
    else:  
        ivar_obs = np.ones(len(w_obs)) 
    photo_obs   = photo['flux'][igal,:3]
    if noise_photo != 'none': 
        photo_ivar_obs = photo['ivar'][igal,:3]
    else:  
        photo_ivar_obs = np.ones(photo_obs.shape[0]) 

    # get fiber flux factor prior range based on measured fiber flux 
    f_fiber_true = (photo['fiberflux_r_meas'][igal]/photo['flux_r_true'][igal]) 
    f_fiber_min = (photo['fiberflux_r_meas'][igal] - 3.*photo['fiberflux_r_ivar'][igal]**-0.5)/photo['flux'][igal,1]
    f_fiber_max = (photo['fiberflux_r_meas'][igal] + 3.*photo['fiberflux_r_ivar'][igal]**-0.5)/photo['flux'][igal,1]
    f_fiber_prior = [f_fiber_min, f_fiber_max]
    print(f_fiber_prior) 
    print(f_fiber_true) 
    print((photo['fiberflux_r_true'][igal]/photo['flux_r_true'][igal])) 

    truths      = [meta['logM_total'][igal], None, None, None, None, None, None, meta['t_age_MW'][igal], None, f_fiber_true]
    labels      = ['$\log M_*$', r'$\beta_1^{\rm SFH}$', r'$\beta_2^{\rm SFH}$', r'$\beta_3^{\rm SFH}$', r'$\beta_4^{\rm SFH}$', r'$\gamma_1^{\rm ZH}$', r'$\gamma_2^{\rm ZH}$', r'$t_{\rm age}$', r'$\tau$', r'$f_{\rm fiber}$']

    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('log M* fiber = %f' % meta['logM_fiber'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 

    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ispeculator', 'lgal.specphoto.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    if not justplot: 
        if os.path.isfile(f_bf): 
            if not overwrite: 
                print("** CAUTION: %s already exists **" % os.path.basename(f_bf)) 
        # initiating fit
        ispeculator = Fitters.iSpeculator(model_name=model) 

        prior = ispeculator._default_prior(f_fiber_prior=f_fiber_prior)

        bestfit = ispeculator.MCMC_spectrophoto(
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
    else: 
        # read in best-fit file with mcmc chain
        fbestfit = h5py.File(f_bf, 'r')  
        bestfit = {} 
        for k in fbestfit.keys(): 
            bestfit[k] = fbestfit[k][...]

    print('--- bestfit ---') 
    print('written to %s' % f_bf) 
    print('log M* total = %f' % bestfit['theta_med'][0])
    print('log M* fiber = %f' % (bestfit['theta_med'][0] + np.log10(bestfit['theta_med'][-1])))
    print('log Z = %f' % bestfit['theta_med'][1]) 
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
    return None 


def MP_fit_iSpeculator(spec_or_photo, igals, noise='none', nthreads=1, nwalkers=100, burnin=100, niter=1000, overwrite=False, justplot=False): 
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
            'noise': noise, 
            'nwalkers': nwalkers,
            'burnin': burnin,
            'niter': niter, 
            'overwrite': overwrite, 
            'justplot': justplot
            }
    if spec_or_photo == 'spec': 
        fit_func = fit_iSpeculator_spectra
    elif spec_or_photo == 'photo': 
        fit_func = fit_iSpeculator_photometry
    elif spec_or_photo == 'specphoto': 
        fit_func = fit_iSpeculator_spectrophotometry

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


if __name__=="__main__": 
    # >>> python mini_mocha.py
    spec_or_photo   = sys.argv[1]

    if spec_or_photo == 'construct':  
        construct_sample()
        validate_sample()
        sys.exit() 

    igal0           = int(sys.argv[2]) 
    igal1           = int(sys.argv[3]) 
    noise           = sys.argv[4]
    method          = sys.argv[5]
    nthreads        = int(sys.argv[6]) 
    nwalkers        = int(sys.argv[7]) 
    burnin          = int(sys.argv[8]) 
    niter           = int(sys.argv[9]) 
    str_overwrite   = sys.argv[10]
    
    if str_overwrite == 'True': overwrite=True
    elif str_overwrite == 'False': overwrite=False

    # if specified, it assumes the chains already exist and just makes the 
    # corner plots (implemented because I have difficult making plots on nersc)
    try: 
        _justplot = sys.argv[11]
        if _justplot == 'True': justplot = True
        elif _justplot == 'False': justplot = False 
    except IndexError: 
        justplot = False

    if method == 'ifsps': 
        print('----------------------------------------') 
        print('iFSPS fitting %s of mini_mocha galaxies %i to %i' % (spec_or_photo, igal0, igal1))
        print('using %i threads' % nthreads) 
        igals = range(igal0, igal1+1) 
        MP_fit_iFSPS(spec_or_photo, igals, noise=noise, nthreads=nthreads, 
                nwalkers=nwalkers, burnin=burnin, niter=niter, overwrite=overwrite, justplot=justplot)
    elif method == 'ispeculator': 
        print('----------------------------------------') 
        print('iSpeculator fitting %s of mini_mocha galaxies %i to %i' % (spec_or_photo, igal0, igal1))
        print('using %i threads' % nthreads) 
        igals = range(igal0, igal1+1) 
        MP_fit_iSpeculator(spec_or_photo, igals, noise=noise, nthreads=nthreads, 
                nwalkers=nwalkers, burnin=burnin, niter=niter, overwrite=overwrite, justplot=justplot)

    elif method == 'pfirefly': 
        raise NotImplementedError('need to update prior set up') 
        for igal in range(igal0, igal1+1):
            fit_pFF_spectra(igal, noise=noise, iter_max=10, overwrite=overwrite)
