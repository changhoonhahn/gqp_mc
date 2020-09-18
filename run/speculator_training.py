import os, sys
import numpy as np 
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


def train_desi_simpledust(ibatch, seed=0): 
    ''' generate FSPS training set for DESI with simple calzetti dust model.
    '''
    # set random seed
    np.random.seed(seed)

    batch = 100
    nspec = 6000

    assert ibatch < batch
    
    # wavelength range set to cover the DESI spectra and photometric filter
    # wavelength range 
    wmin, wmax = 2300., 11030.

    speculate = Fitters.iSpeculator(model_name='fsps')

    theta_train = sample_simpledust_prior(batch*nspec)
    theta_train[10000:,:4] =\
            speculate._transform_to_SFH_basis(np.random.uniform(size=(batch*nspec-10000,4)))
    
    w_fsps, _ = speculate._fsps_model(theta_train[0])
    wlim = (w_fsps >= wmin) & (w_fsps <= wmax)

    fwave = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy')
    if not os.path.isfile(fwave):
        # save FSPS wavelength if not saved 
        np.save(fwave, w_fsps[wlim])

    ftheta = os.path.join(UT.dat_dir(), 'speculator',
            'DESI_simpledust.theta_train.%i.seed%i.npy' % (ibatch, seed))
    fspectrum = os.path.join(UT.dat_dir(), 'speculator',
            'DESI_simpledust.logspectrum_fsps_train.%i.seed%i.npy' % 
            (ibatch, seed))

    if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
        print('--- batch %i already exists ---' % ibatch)
        return None 
    else: 
        print('--- batch %i ---' % ibatch)

        i_batch = range(nspec*ibatch,nspec*(ibatch+1))

        logspectra_train = []
        for _theta in theta_train[i_batch]:
            _, _spectrum = speculate._fsps_model(_theta)
            logspectra_train.append(np.log(_spectrum[wlim]))

        np.save(ftheta, theta_train[i_batch])
        np.save(fspectrum, np.array(logspectra_train))

    return None 


def train_desi_complexdust(ibatch, seed=0): 
    ''' generate FSPS training set for DESI with complex noll dust model.
    '''
    # set random seed
    np.random.seed(seed)
    
    # we may need more batches... 
    batch = 100
    nspec = 10000 

    assert ibatch < batch
    
    # wavelength range set to cover the DESI spectra and photometric filter
    # wavelength range 
    wmin, wmax = 2300., 11030.

    speculate = Fitters.iSpeculator(model_name='fsps_complexdust')

    theta_train = sample_complexdust_prior(batch*nspec)
    # I'll keep 10000 outside of dirichlet priors as padding. Not sure if this
    # is a good idea or not. 
    theta_train[10000:,:4] =\
            speculate._transform_to_SFH_basis(np.random.uniform(size=(batch*nspec-10000,4)))
    
    w_fsps, _ = speculate._fsps_model(theta_train[0])
    wlim = (w_fsps >= wmin) & (w_fsps <= wmax)

    fwave = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy')
    if not os.path.isfile(fwave):
        # save FSPS wavelength if not saved 
        np.save(fwave, w_fsps[wlim])

    ftheta = os.path.join(UT.dat_dir(), 'speculator',
            'DESI_complexdust.theta_train.%i.seed%i.npy' % (ibatch, seed))
    fspectrum = os.path.join(UT.dat_dir(), 'speculator',
            'DESI_complexdust.logspectrum_fsps_train.%i.seed%i.npy' % 
            (ibatch, seed))

    if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
        print('--- batch %i already exists ---' % ibatch)
        return None 
    else: 
        print('--- batch %i ---' % ibatch)

        i_batch = range(nspec*ibatch,nspec*(ibatch+1))

        logspectra_train = []
        for _theta in theta_train[i_batch]:
            _, _spectrum = speculate._fsps_model(_theta)
            logspectra_train.append(np.log(_spectrum[wlim]))

        np.save(ftheta, theta_train[i_batch])
        np.save(fspectrum, np.array(logspectra_train))

    return None 



if __name__=='__main__': 
    # e.g. 
    # >>> speculator_training.py simpledust 0
    
    model = sys.argv[1]
    ibatch = int(sys.argv[2])
    if model == 'simpledust':
        train_desi_simpledust(ibatch, seed=0)
    elif model == 'complexdust': 
        train_desi_complexdust(ibatch, seed=0)
    else:
        raise ValueError 
