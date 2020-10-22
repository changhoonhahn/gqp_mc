import os, sys
import numpy as np 
import dask
# -- gqp_mc --
from gqp_mc import util as UT
from gqp_mc import fitters as Fitters
# -- plotting --
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


# wavelength range set to cover the DESI spectra and photometric filter
# wavelength range 
wmin, wmax = 2300., 11030.


def sample_simpledust_prior(n_sample):
    ''' sample a padded uniform prior
    '''
    prior_min = np.array([0.0, 0.0, 0.0, 0.0, 6.5e-5, 6.5e-5, 0.0, 8.6])
    prior_max = np.array([1.1, 1.1, 1.1, 1.1, 7.5e-3, 7.5e-3, 3.5, 13.8])
    return prior_min + (prior_max - prior_min) * np.random.uniform(size=(n_sample, len(prior_min)))


def sample_complexdust_prior(n_sample):
    ''' sample a padded uniform prior
    '''
    prior_min = np.array([0.0, 0.0, 0.0, 0.0, 6.5e-5, 6.5e-5, 0.0, 0.0, -2.5, 8.6])
    prior_max = np.array([1.1, 1.1, 1.1, 1.1, 7.5e-3, 7.5e-3, 3.5, 4.0, 0.5, 13.8])

    return prior_min + (prior_max - prior_min) * np.random.uniform(size=(n_sample, len(prior_min)))


def train_desi_seds(model, ibatch, seed=0, ncpu=1): 
    ''' generate FSPS training set for DESI with either the simple calzetti
    dust model or a more complex Noll+2009 dust model 
    '''
    # set random seed 
    np.random.seed(seed * ibatch)

    nspec = 10000 # per batch
    
    if model == 'simpledust': 
        speculate = Fitters.iSpeculator(model_name='fsps')
        theta_train = sample_simpledust_prior(nspec)
    elif model == 'complexdust': 
        speculate = Fitters.iSpeculator(model_name='fsps_complexdust')
        theta_train = sample_complexdust_prior(nspec)

    theta_train[:,:4] = speculate._transform_to_SFH_basis(np.random.uniform(size=(nspec,4)))
    
    w_fsps, _ = speculate._fsps_model(theta_train[0])
    wlim = (w_fsps >= wmin) & (w_fsps <= wmax)

    fwave = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy')
    if not os.path.isfile(fwave):
        # save FSPS wavelength if not saved 
        np.save(fwave, w_fsps[wlim])

    ftheta = os.path.join(UT.dat_dir(), 'speculator',
            'DESI_%s.theta_train.%i.seed%i.npy' % (model, ibatch, seed))
    fspectrum = os.path.join(UT.dat_dir(), 'speculator',
            'DESI_%s.logspectrum_fsps_train.%i.seed%i.npy' % (model, ibatch, seed))

    if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
        print('') 
        print('--- batch %i already exists ---' % ibatch)
        print('--- do not overwrite ---' % ibatch)
        print('')  
        return None 
    else: 
        print('--- batch %i ---' % ibatch)
        if ncpu == 1: 
            logspectra_train = []
            for _theta in theta_train:
                _, _spectrum = speculate._fsps_model(_theta)
                logspectra_train.append(np.log(_spectrum[wlim]))
        else: 
            def _fsps_model_wrapper(theta):
                _, _spectrum = speculate._fsps_model(theta)
                return np.log(_spectrum[wlim]) 

            lazys = [] 
            for _theta in theta_train: 
                lazy = dask.delayed(_fsps_model_wrapper)(_theta)
                lazys.append(lazy) 
            logspectra_train = dask.compute(*lazys) 

        np.save(ftheta, theta_train)
        np.save(fspectrum, np.array(logspectra_train))

    return None 


def test_desi_seds(model): 
    ''' generate FSPS test set for DESI with either simple calzetti dust model
    or the more complex Noll+(2009) model
    '''
    # set random seed
    np.random.seed(1234)

    nspec = 10000
    
    if model == 'simpledust': 
        speculate = Fitters.iSpeculator(model_name='fsps')
        theta_train = sample_simpledust_prior(nspec)
    elif model == 'complexdust': 
        speculate = Fitters.iSpeculator(model_name='fsps_complexdust')
        theta_train = sample_complexdust_prior(nspec)

    theta_train[:,:4] =\
            speculate._transform_to_SFH_basis(np.random.uniform(size=(nspec,4)))
    
    w_fsps, _ = speculate._fsps_model(theta_train[0])
    wlim = (w_fsps >= wmin) & (w_fsps <= wmax)

    fwave = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy')
    if not os.path.isfile(fwave):
        # save FSPS wavelength if not saved 
        np.save(fwave, w_fsps[wlim])

    ftheta = os.path.join(UT.dat_dir(), 'speculator', 'DESI_%s.theta_test.npy' % model)
    fspectrum = os.path.join(UT.dat_dir(), 'speculator', 'DESI_%s.logspectrum_fsps_test.npy' % model)

    if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
        print('--- test set already exists ---')
        return None 
    else: 
        print('--- test set ---')

        logspectra_train = []
        for _theta in theta_train:
            _, _spectrum = speculate._fsps_model(_theta)
            logspectra_train.append(np.log(_spectrum[wlim]))

        np.save(ftheta, theta_train)
        np.save(fspectrum, np.array(logspectra_train))
    return None 


if __name__=='__main__': 
    # e.g. 
    # >>> speculator_training.py train simpledust 0
    train_or_test   = sys.argv[1] 
    model           = sys.argv[2]

    assert model in ['simpledust', 'complexdust']

    if train_or_test == 'train': 
        ibatch  = int(sys.argv[3])
        ncpu    = int(sys.argv[4]) 
    
        train_desi_seds(model, ibatch, seed=0, ncpu=ncpu)
    elif train_or_test == 'test' :
        test_desi_seds(model)
    else:
        raise ValueError 
