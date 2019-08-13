'''

fit the spectra of the spectral_challenge mocks 


'''
import os 
import sys
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


def nonoise_spectra(): 
    # read noiseless Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim='lgal', noise='none', lib='bc03', sample='spectral_challenge') 
    
    ifsps = Fitters.iFSPS(model_name='dustless_vanilla', prior=None) 

    for igal in [0]: 
        print('--- input ---') 
        print('z = %f' % meta['redshift'][igal])
        print('log M* total = %f' % meta['logM_total'][igal])
        print('M* disk = %f' % np.sum(meta['sfh_disk'][igal]))
        print('M* bulge = %f' % np.sum(meta['sfh_bulge'][igal]))
    
        f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', 'ifsps', 'spec.nonoise.nodust.dustless_vanilla.%i.hdf5' % igal)
        bestfit = ifsps.MCMC_spec(
                specs['wavelength'][igal], 
                specs['flux_nodust'][igal], 
                np.ones(len(specs['flux_nodust'][igal])), 
                meta['redshift'][igal], 
                mask='emline', 
                nwalkers=10, 
                burnin=100, 
                niter=1000, 
                threads=1,
                writeout=f_bf,
                silent=False)
        print('--- bestfit ---') 
        print('log M* = %f' % bestfit['theta_med'][0])
        print('log Z = %f' % bestfit['theta_med'][1]) 
        print('---------------') 
        fig = DFM.corner(bestfit['mcmc_chain'], range=ifsps.prior) 
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
    fig = DFM.corner(bestfit['mcmc_chain'], range=ifsps.priors, quantiles=[0.16, 0.5, 0.84], 
            truths=[meta['logM_total'][igal], np.log10(meta['Z_MW'][igal]), meta['t_age_MW'][igal], None], 
            labels=['$\log M_*$', '$\log Z$', r'$t_{\rm age}$', r'$\tau$'], label_kwargs={'fontsize': 20}) 
    fig.savefig(f_bf.replace('.hdf5', '.png'), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    spec_or_photo = sys.argv[1]
    igal = int(sys.argv[2]) 

    if spec_or_photo == 'photo': 
        nonoise_photometry(igal)
    else: 
        raise NotImplementedError
        #nonoise_spectra()
