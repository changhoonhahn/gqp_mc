'''

validate that Speculator is sufficiently accurate 

'''
import os, sys 
import pickle
import numpy as np
# -- gqp_mc --
from gqp_mc import util as UT
# --- speculator ---
from speculator import Speculator
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



def validate_desi_simpledust(batch0, batch1, seed=0): 
    ''' validate Speculator model for simple dust set up 
    '''
    assert os.environ['NERSC_HOST'] == 'cori' # meant to be run on cori 

    # fsps wavelength 
    fwave   = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy') 
    wave    = np.load(fwave)

    # batches of fsps spectra
    batches = range(batch0, batch1+1)

    fthetas = [os.path.join(UT.dat_dir(), 'speculator',
        'DESI_simpledust.theta_train.%i.seed%i.npy' % (i, seed)) for i in batches]
    fspecs  = [os.path.join(UT.dat_dir(), 'speculator', 
        'DESI_simpledust.logspectrum_fsps_train.%i.seed%i.npy' % (i, seed)) for i in batches]
    
    # read in training theta and spectrum 
    training_theta = np.concatenate([np.load(fthetas[i]) for i in range(len(batches))])
    training_spectrum = np.concatenate([np.load(fspecs[i]) for i in range(len(batches))])
    # training selection 
    trainig_selection = np.load('DESI_simpledust_model.trainig_selection.npy') 

    # load Speculator object
    speculator = Speculator(restore=True, restore_filename='_DESI_simpledust_model.log')

    theta_test = training_theta[~training_selection]
    spectrum_test = training_spectrum[~training_selection]
    spectrum_spec = speculator.log_spectrum(theta_test)

    # figure comparing Speculator log spectrum to FSPS log spectrum  
    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    for i in np.random.choice(len(theta_test), size=5): 
        sub.plot(wave, spectrum_spec[i], c='C%i' % i, ls='-', label='Speculator')
        sub.plot(wave, spectrum_test[i], c='C%i' % i, ls=':', label='FSPS')

    sub.set_xlabel('wavelength ($A$)', fontsize=25) 
    sub.set_xlim(3e3, 1e4)
    sub.set_ylabel('log flux', fontsize=25) 
    ffig = os.path.join(UT.dat_dir(), 'speculator', 'validate_desi_simpledust0.png') 
    fig.savefig(ffig, bbox_inches='tight') 

    # more quantitative accuracy test of the Speculator model 
    frac_dspectrum = (spectrum_spec - spectrum_test) / spectrum_test 
    frac_dspectrum_quantiles = np.quantile(frac_dspectrum, 
            [0.0005, 0.005, 0.025, 0.5, 0.975, 0.995, 0.9995], axis=0)

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111)
    sub.fill_between(wave, frac_dspectrum_quantiles[0], frac_dspectrum_quantiles[6], color='C0', alpha=0.1)
    sub.fill_between(wave, frac_dspectrum_quantiles[1], frac_dspectrum_quantiles[5], color='C0', alpha=0.2)
    sub.fill_between(wave, frac_dspectrum_quantiles[2], frac_dspectrum_quantiles[4], color='C0', alpha=0.3)
    sub.plot(wave, frac_dspectrum_quantiles[3], c='C0', ls='-') 
    sub.plot(wave, np.ones(len(wave)), c='k', ls=':') 

    sub.set_xlabel('wavelength ($A$)', fontsize=25) 
    sub.set_xlim(3e3, 1e4)
    #sub.set_ylabel('log flux', fontsize=25) 
    ffig = os.path.join(UT.dat_dir(), 'speculator', 'validate_desi_simpledust1.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 
