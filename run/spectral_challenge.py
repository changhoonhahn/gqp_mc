''' 
fit the spectra or photometry of the spectral_challenge mocks 

Usage::
    e.g. fit spectra of galaxies 0 to 9 with no noise, no dust, run on 10 threads, 100 walkers, 100 burnin, 1000 main chain
    >>> python spectral_challenge.py spec 0 9 none False 10 100 100 1000

    e.g. fit photometry of galaxies 0 to 9 with no noise, no dust, run on 10 threads, 100 walkers, 100 burnin, 1000 main chain
    >>> python spectral_challenge.py photo 0 9 none False 10 100 100 1000

'''
import os 
import sys
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


def fit_spectra(igal, noise='none', dust=False, nwalkers=100, burnin=100, niter=1000, justplot=False): 
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without noise or 
    with BGS-like noise. `dust` specifies whether to if spectra w/ dust or not. 
    Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 

    :param noise: 
        If 'none', fit noiseless spectra. 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 

    :param dust: 
        If True, fit the spectra w/ dust using a model with dust 
        If False, fit the spectra w/o dust using a model without dust. 
        (default: False) 

    :param justplot: 
        If True, skip the fitting and plot the best-fit. This is mainly implemented 
        because I'm having issues plotting in NERSC. (default: False) 
    '''
    # read noiseless Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim='lgal', noise=noise, lib='bc03', sample='spectral_challenge') 

    if not dust: 
        model       = 'dustless_vanilla'
        w_obs       = specs['wavelength'][igal] 
        flux_obs    = specs['flux_nodust'][igal]
        if noise != 'none': ivar_obs = specs['ivar_nodust'][igal]
        truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None]
        labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', r'$\tau$']
    else: 
        model       = 'vanilla'
        w_obs       = specs['wavelength'][igal] 
        flux_obs    = specs['flux_dust'][igal]
        if noise != 'none': ivar_obs = specs['ivar_dust'][igal]
        truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None, None]
        labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', 'dust2', r'$\tau$']

    if noise == 'none': # no noise 
        ivar_obs = np.ones(len(w_obs)) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 
    
    f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 
            'spec.noise_%s.dust_%s.%s.%i.hdf5' % (noise, ['no', 'yes'][dust], model, igal))
    if not justplot: 
        if os.path.isfile(f_bf): 
            print("** CAUTION: %s already exists and is being overwritten **" % os.path.basename(f_bf)) 
        # initiating fit
        ifsps = Fitters.iFSPS(model_name=model, prior=None) 
        bestfit = ifsps.MCMC_spec(
                w_obs, 
                flux_obs, 
                ivar_obs, 
                meta['redshift'][igal], 
                mask='emline', 
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
        fig = DFM.corner(bestfit['mcmc_chain'], range=bestfit['priors'], quantiles=[0.16, 0.5, 0.84], 
                levels=[0.68, 0.95], nbin=40, smooth=True, 
                truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
        fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def fit_photometry(igal, noise='none', dust=False, nwalkers=100, burnin=100, niter=1000, justplot=False): 
    ''' Fit Lgal photometry. `noise` specifies whether to fit spectra without noise or 
    with legacy-like noise. `dust` specifies whether to if spectra w/ dust or not. 
    Produces an MCMC chain and, if not on nersc, a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 

    :param noise: 
        If 'none', fit noiseless photometry. 
        If 'legacy', fit Legacy-like photometry. (default: 'none') 

    :param dust: 
        If True, fit photometry w/ dust using a model with dust 
        If False, fit photometry w/o dust using a model without dust. 
        (default: False) 
    
    :param justplot: 
        If True, skip the fitting and plot the best-fit. This is mainly implemented 
        because I'm having issues plotting in NERSC. (default: False) 
    '''
    # read Lgal photometry of the spectral_challenge mocks 
    photo, meta = Data.Photometry(sim='lgal', noise=noise, lib='bc03', sample='spectral_challenge') 
    
    if not dust: 
        model       = 'dustless_vanilla'
        photo_obs   = np.array([photo['flux_nodust_%s' % band][igal] for band in ['g', 'r', 'z', 'w1', 'w2']])  
        if noise != 'none': 
            ivar_obs    = np.array([photo['ivar_nodust_%s' % band][igal] for band in ['g', 'r', 'z', 'w1', 'w2']])
        truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None]
        labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', r'$\tau$']
    else: 
        model       = 'vanilla'
        photo_obs   = np.array([photo['flux_dust_%s' % band][igal] for band in ['g', 'r', 'z', 'w1', 'w2']])  
        if noise != 'none': 
            ivar_obs    = np.array([photo['ivar_dust_%s' % band][igal] for band in ['g', 'r', 'z', 'w1', 'w2']])
        truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None, None]
        labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', 'dust2', r'$\tau$']

    if noise == 'none': # no noise 
        ivar_obs = np.ones(len(photo_obs)) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 
    
    f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 
            'photo.noise_%s.dust_%s.%s.%i.hdf5' % (noise, ['no', 'yes'][dust], model, igal))
    if not justplot: 
        if os.path.isfile(f_bf): 
            print("** CAUTION: %s already exists and is being overwritten **" % os.path.basename(f_bf)) 
        # initiate fitting
        ifsps = Fitters.iFSPS(model_name=model, prior=None) 
        bestfit = ifsps.MCMC_photo(
                photo_obs, 
                ivar_obs,
                meta['redshift'][igal], 
                bands='desi', 
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
        fig = DFM.corner(bestfit['mcmc_chain'], range=bestfit['priors'], quantiles=[0.16, 0.5, 0.84], 
                levels=[0.68, 0.95], nbin=40, smooth=True, 
                truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
        fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def MP_fit(spec_or_photo, igals, noise='none', dust=False, nthreads=1, nwalkers=100, burnin=100, niter=1000, justplot=False): 
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
            'dust': dust,
            'nwalkers': nwalkers,
            'burnin': burnin,
            'niter': niter, 
            'justplot': justplot
            }
    if spec_or_photo == 'spec': 
        fit_func = fit_spectra
    elif spec_or_photo == 'photo': 
        fit_func = fit_photometry

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


if __name__=="__main__": 
    # python spectral_challenge.py
    spec_or_photo   = sys.argv[1]
    igal0           = int(sys.argv[2]) 
    igal1           = int(sys.argv[3]) 
    noise           = sys.argv[4]
    str_dust        = sys.argv[5]
    nthreads        = int(sys.argv[6]) 
    nwalkers        = int(sys.argv[7]) 
    burnin          = int(sys.argv[8]) 
    niter           = int(sys.argv[9]) 
    
    if str_dust == 'True': dust = True
    elif str_dust == 'False': dust = False 
    
    # if specified, it assumes the chains already exist and just makes the 
    # corner plots (implemented because I have difficult making plots on nersc)
    try: 
        _justplot = sys.argv[10]
        if _justplot == 'True': justplot = True
        elif _justplot == 'False': justplot = False 
    except IndexError: 
        justplot = False

    print('----------------------------------------') 
    print('fitting %s of spectral_challenge galaxies %i to %i' % (spec_or_photo, igal0, igal1))
    igals = range(igal0, igal1+1) 
    MP_fit(spec_or_photo, igals, noise=noise, dust=dust, nthreads=nthreads, 
            nwalkers=nwalkers, burnin=burnin, niter=niter, justplot=justplot)
