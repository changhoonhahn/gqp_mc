import os
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


def sample_prior(n_sample):
    ''' sample a padded uniform prior
    '''
    prior_min = np.array([0.0, 0.0, 0.0, 0.0, 6.5e-5, 6.5e-5, 0.0, 8.6])
    prior_max = np.array([1.1, 1.1, 1.1, 1.1, 7.5e-3, 7.5e-3, 3.5, 13.8])
    return prior_min + (prior_max - prior_min) * np.random.uniform(size=(n_sample, len(prior_min)))


def train_desi_simpledust(seed=0): 
    ''' generate FSPS training set for DESI with simple calzetti dust model.
    '''
    # set random seed
    np.random.seed(seed)

    batch = 100
    nspec = 6000
    
    # wavelength range set to cover the DESI spectra and photometric filter
    # wavelength range 
    wmin, wmax = 2300., 11030.

    speculate = Fitters.iSpeculator(model_name='fsps')

    theta_train = sample_prior(batch*nspec)
    theta_train[10000:,:4] =\
            speculate._transform_to_SFH_basis(np.random.uniform(size=(batch*nspec,4)))
    
    # save FSPS wavelength 
    w_fsps, _ = speculate._fsps_model(theta_train[0])
    wlim = (w_fsps >= wmin) & (w_fsps <= wmax)

    fwave = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy')
    np.save(fwave, w_fsps[wlim])

    for i in range(batch):
        ftheta = os.path.join(UT.dat_dir(), 'speculator', 'DESI_simpledust.theta_train.%i.npy' % i)
        fspectrum = os.path.join(UT.dat_dir(), 'speculator', 'DESI_simpledust.logspectrum_fsps_train.%i.npy' % i)
    
        if os.path.isfile(ftheta) and os.path.isfile(fspectrum): 
            pass
        else: 
            print('--- batch %i ---' % i)

            i_batch = range(nspec*i,nspec*(i+1))

            logspectra_train = []
            for _theta in theta_train[i_batch]:
                _, _spectrum = speculate._fsps_model(_theta)
                logspectra_train.append(np.log(_spectrum[wlim]))

            np.save(ftheta, theta_train[i_batch])
            np.save(fspectrum, np.array(logspectra_train))

    return None 


if __name__=='__main__': 
    train_desi_simpledust(seed=0)
