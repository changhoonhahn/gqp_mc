'''

fit the spectra of the spectral_challenge mocks 


'''
import os 
import sys
import h5py 
import numpy as np 
import corner as DFM
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
from gqp_mc import fitters as Fitters
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode']=True
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


def fit_spectra(igal, noise='none', dust=False, justplot=False): 
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
        ivar_obs    = specs['ivar_nodust'][igal]
        truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None]
        labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', r'$\tau$']
    else: 
        model       = 'vanilla'
        w_obs       = specs['wavelength'][igal] 
        flux_obs    = specs['flux_dust'][igal]
        ivar_obs    = specs['ivar_dust'][igal]
        truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None, None]
        labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', 'dust2', r'$\tau$']

    if noise is None: # no noise 
        ivar_obs = np.ones(len(w_obs)) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 
    
    f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 
            'spec.noise_%s.dust_%s.%s.%i.hdf5' % (noise, ['no', 'yes'][dust], model, igal))
    if not justplot: 
        # initiating fit
        ifsps = Fitters.iFSPS(model_name=model, prior=None) 
        bestfit = ifsps.MCMC_spec(
                w_obs, 
                flux_obs, 
                ivar_obs, 
                meta['redshift'][igal], 
                mask='emline', 
                nwalkers=10, 
                burnin=100, 
                niter=1000, 
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
                truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
        fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def fit_photometry(igal, noise='none', dust=False, justplot=False): 
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
        ivar_obs    = np.array([photo['ivar_nodust_%s' % band][igal] for band in ['g', 'r', 'z', 'w1', 'w2']])  
        truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None]
        labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', r'$\tau$']
    else: 
        model       = 'vanilla'
        photo_obs   = np.array([photo['flux_dust_%s' % band][igal] for band in ['g', 'r', 'z', 'w1', 'w2']])  
        ivar_obs    = np.array([photo['ivar_dust_%s' % band][igal] for band in ['g', 'r', 'z', 'w1', 'w2']])  
        truths      = [meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None, None]
        labels      = ['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', 'dust2', r'$\tau$']

    if noise is None: # no noise 
        ivar_obs = np.ones(len(photo_obs)) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 
    
    f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 
            'photo.noise_%s.dust_%s.%s.%i.hdf5' % (noise, ['no', 'yes'][dust], model, igal))
    if not justplot: 
        # initiate fitting
        ifsps = Fitters.iFSPS(model_name=model, prior=None) 
        bestfit = ifsps.MCMC_photo(
                photo_obs, 
                ivar_obs,
                meta['redshift'][igal], 
                bands='desi', 
                nwalkers=100, 
                burnin=100, 
                niter=1000, 
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
                truths=truths, labels=labels, label_kwargs={'fontsize': 20}) 
        fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def _plot_MCMCchain(igal, bestfit_file):  
    ''' 
    :param igal: 
        index of spectral challenge galaxy 

    :param bestfit_file:
        file name of best-fit file that includes the MCMC chain.
    '''
    # read meta data of the spectral_challenge mocks 
    _, meta = Data.Photometry(sim='lgal', noise='none', lib='bc03', sample='spectral_challenge') 

    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 
    
    # read in best-fit file with mcmc chain
    f_bf = bestfit_file 
    fbestfit = h5py.File(f_bf, 'r')  
    bestfit = {} 
    for k in fbestfit.keys(): 
        bestfit[k] = fbestfit[k][...]

    # **WARNING** for the future this should be replaced with bestfit['priors']!!
    priors = [(8, 13), (-3, 1), (0., 13.), (0.1, 10.)]

    fig = DFM.corner(bestfit['mcmc_chain'], range=priors, quantiles=[0.16, 0.5, 0.84], 
            truths=[meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None], 
            labels=['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', r'$\tau$'], label_kwargs={'fontsize': 20}) 
    fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    spec_or_photo   = sys.argv[1]
    igal            = int(sys.argv[2]) 
    noise           = sys.argv[3]
    str_dust        = sys.argv[4]
    if str_dust == 'True': dust = True
    elif str_dust == 'False': dust = False 
    
    try: 
        _justplot = sys.argv[5]
        if _justplot == 'True': justplot = True
        elif _justplot == 'False': justplot = False 
    except IndexError: 
        justplot = False
    
    if spec_or_photo == 'photo': 
        fit_photometry(igal, noise=noise, dust=dust, justplot=justplot)
    elif spec_or_photo == 'spec': 
        fit_spectra(igal, noise=noise, dust=dust, justplot=justplot)
