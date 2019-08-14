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


def nonoise_spectra(igal): 
    # read noiseless Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim='lgal', noise='none', lib='bc03', sample='spectral_challenge') 
    
    ifsps = Fitters.iFSPS(model_name='dustless_vanilla', prior=None) 
    
    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 

    f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 
            'spec.nonoise.nodust.dustless_vanilla.%i.hdf5' % igal)
    bestfit = ifsps.MCMC_spec(
            specs['wavelength'][igal], 
            specs['flux_nodust'][igal], 
            np.ones(len(specs['flux_nodust'][igal])), 
            meta['redshift'][igal], 
            mask='emline', 
            nwalkers=10, 
            burnin=100, 
            niter=1000, 
            writeout=f_bf,
            silent=False)
    print('--- bestfit ---') 
    print('log M* = %f' % bestfit['theta_med'][0])
    print('log Z = %f' % bestfit['theta_med'][1]) 
    print('---------------') 
    if os.environ['NERSC_HOST'] == 'cori': return None 
    fig = DFM.corner(bestfit['mcmc_chain'], range=ifsps.priors, quantiles=[0.16, 0.5, 0.84], 
            truths=[meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None], 
            labels=['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', r'$\tau$'], label_kwargs={'fontsize': 20}) 
    fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def nonoise_photometry(igal): 
    # read Lgal photometry of the spectral_challenge mocks 
    photo, meta = Data.Photometry(sim='lgal', noise='none', lib='bc03', sample='spectral_challenge') 
    
    ifsps = Fitters.iFSPS(model_name='dustless_vanilla', prior=None) 

    print('--- input ---') 
    print('z = %f' % meta['redshift'][igal])
    print('log M* total = %f' % meta['logM_total'][igal])
    print('MW Z = %f' % meta['Z_MW'][igal]) 
    print('MW tage = %f' % meta['t_age_MW'][igal]) 

    photo_i = np.array([
        photo['flux_nodust_g'][igal], 
        photo['flux_nodust_r'][igal], 
        photo['flux_nodust_z'][igal], 
        photo['flux_nodust_w1'][igal], 
        photo['flux_nodust_w2'][igal] 
        ])  
    print(photo_i) 

    f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 
            'photo.nonoise.nodust.dustless_vanilla.%i.WRONG.hdf5' % igal)
    print('--- writing to %s ---' % f_bf)
    bestfit = ifsps.MCMC_photo(
            photo_i, 
            np.ones(len(photo_i)), 
            meta['redshift'][igal], 
            bands='desi', 
            nwalkers=100, 
            burnin=100, 
            niter=1000, 
            writeout=f_bf,
            silent=False)
    print('--- bestfit ---') 
    print('log M* = %f' % bestfit['theta_med'][0])
    print('log Z = %f' % bestfit['theta_med'][1]) 
    print('---------------') 

    if os.environ['NERSC_HOST'] == 'cori': return None 

    fig = DFM.corner(bestfit['mcmc_chain'], range=ifsps.priors, quantiles=[0.16, 0.5, 0.84], 
            truths=[meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None], 
            labels=['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', r'$\tau$'], label_kwargs={'fontsize': 20}) 
    fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


def plot_MCMCchain(igal, bestfit_file):  
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
    spec_or_photo = sys.argv[1]
    igal = int(sys.argv[2]) 

    if spec_or_photo == 'photo': 
        nonoise_photometry(igal)
    elif spec_or_photo == 'spec': 
        nonoise_spectra(igal)
    elif spec_or_photo == 'plot': 
        # quick hacks to plot the MCMC chains
        f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 
            'spec.nonoise.nodust.dustless_vanilla.%i.hdf5' % igal)
        plot_MCMCchain(igal, f_bf)
        f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 
                'photo.nonoise.nodust.dustless_vanilla.%i.WRONG.hdf5' % igal)
        plot_MCMCchain(igal, f_bf)
