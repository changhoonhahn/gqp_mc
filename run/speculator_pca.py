'''

train PCA for speculator training data sets and save to pickle file 

'''
import os, sys 
import pickle
import numpy as np
# -- gqp_mc --
from gqp_mc import util as UT
# --- speculator ---
from speculator import SpectrumPCA


def train_desi_simpledust_pca(batch0, batch1, seed=0): 
    ''' train PCA for DESI simple dust training set 
    '''
    # fsps wavelength 
    fwave   = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy') 
    wave    = np.load(fwave)

    # batches of fsps spectra
    batches = range(batch0, batch1+1)

    fthetas = [os.path.join(UT.dat_dir(), 'speculator',
        'DESI_simpledust.theta_train.%i.seed%i.npy' % (i, seed)) for i in batches]
    fspecs  = [os.path.join(UT.dat_dir(), 'speculator', 
        'DESI_simpledust.logspectrum_fsps_train.%i.seed%i.npy' % (i, seed)) for i in batches]

    # theta = [b1, b2, b3, b4, g1, g2, tau, tage]
    n_param = 8 
    n_wave  = len(wave)
    n_pcas  = 20 
    
    # train PCA basis 
    PCABasis = SpectrumPCA(
            n_parameters=n_param,       # number of parameters
            n_wavelengths=n_wave,       # number of wavelength values
            n_pcas=n_pcas,              # number of pca coefficients to include in the basis 
            spectrum_filenames=fspecs,  # list of filenames containing the (un-normalized) log spectra for training the PCA
            parameter_filenames=fthetas, # list of filenames containing the corresponding parameter values
            parameter_selection=None) # pass an optional function that takes in parameter vector(s) and returns True/False for any extra parameter cuts we want to impose on the training sample (eg we may want to restrict the parameter ranges)

    PCABasis.compute_spectrum_parameters_shift_and_scale() # computes shifts and scales for (log) spectra and parameters
    PCABasis.train_pca()
    PCABasis.transform_and_stack_training_data(
            os.path.join(UT.dat_dir(), 'speculator', 
                'DESI_simpledust.%i_%i.seed%i' % (batch0, batch1, seed)), 
            retain=True) 

    #  save to file 
    PCABasis._save_to_file(
            os.path.join(UT.dat_dir(), 'speculator', 
                'DESI_simpledust.%i_%i.seed%i.pca.hdf5' % (batch0, batch1, seed))
            )
    return None 


def train_desi_complexdust_pca(batch0, batch1, seed=0): 
    ''' train PCA for DESI complex dust training set 
    '''
    # fsps wavelength 
    fwave   = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy') 
    wave    = np.load(fwave)

    # batches of fsps spectra
    batches = range(batch0, batch1+1)

    fthetas = [os.path.join(UT.dat_dir(), 'speculator',
        'DESI_complexdust.theta_train.%i.seed%i.npy' % (i, seed)) for i in batches]
    fspecs  = [os.path.join(UT.dat_dir(), 'speculator', 
        'DESI_complexdust.logspectrum_fsps_train.%i.seed%i.npy' % (i, seed)) for i in batches]

    # theta = [b1, b2, b3, b4, g1, g2, tau, dust1, dust2, tage]
    n_param = 10 
    n_wave  = len(wave)
    n_pcas  = 30 
    
    # train PCA basis 
    PCABasis = SpectrumPCA(
            n_parameters=n_param,       # number of parameters
            n_wavelengths=n_wave,       # number of wavelength values
            n_pcas=n_pcas,              # number of pca coefficients to include in the basis 
            spectrum_filenames=fspecs,  # list of filenames containing the (un-normalized) log spectra for training the PCA
            parameter_filenames=fthetas, # list of filenames containing the corresponding parameter values
            parameter_selection=None) # pass an optional function that takes in parameter vector(s) and returns True/False for any extra parameter cuts we want to impose on the training sample (eg we may want to restrict the parameter ranges)

    PCABasis.compute_spectrum_parameters_shift_and_scale() # computes shifts and scales for (log) spectra and parameters
    PCABasis.train_pca()
    PCABasis.transform_and_stack_training_data(
            os.path.join(UT.dat_dir(), 'speculator', 
                'DESI_complexdust.%i_%i.seed%i' % (batch0, batch1, seed)), 
            retain=True) 

    #  save to file 
    PCABasis._save_to_file(
            os.path.join(UT.dat_dir(), 'speculator', 
                'DESI_complexdust.%i_%i.seed%i.pca.hdf5' % (batch0, batch1, seed))
            )
    return None 


if __name__=="__main__": 
    model   = sys.argv[1]
    ibatch0 = int(sys.argv[2])
    ibatch1 = int(sys.argv[3])
    
    if model == 'simpledust': 
        train_desi_simpledust_pca(ibatch0, ibatch1, seed=0)
    else: 
        train_desi_complexdust_pca(ibatch0, ibatch1, seed=0)
