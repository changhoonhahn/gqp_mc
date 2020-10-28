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


def train_desi_pca(model, batch0, batch1, n_pcas, seed=0): 
    ''' train PCA for DESI simpledust or complexdust training sets
    '''
    # fsps wavelength 
    fwave   = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy') 
    wave    = np.load(fwave)

    # batches of fsps spectra
    batches = range(batch0, batch1+1)

    fthetas = [os.path.join(UT.dat_dir(), 'speculator',
        'DESI_%s.theta_train.%i.seed%i.npy' % (model, i, seed)) for i in batches]
    fspecs  = [os.path.join(UT.dat_dir(), 'speculator', 
        'DESI_%s.logspectrum_fsps_train.%i.seed%i.npy' % (model, i, seed)) for i in batches]
    
    if model == 'simpledust': 
        # theta = [b1, b2, b3, b4, g1, g2, tau, tage]
        n_param = 8 
    elif model == 'complexdust': 
        # theta = [b1, b2, b3, b4, g1, g2, tau, dust1, dust2, tage]
        n_param = 10 
    n_wave  = len(wave)
    
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
                'DESI_%s.%i_%i.seed%i.pca%i' % (model, batch0, batch1, seed, n_pcas)), 
            retain=True) 

    #  save to file 
    PCABasis._save_to_file(
            os.path.join(UT.dat_dir(), 'speculator', 
                'DESI_%s.%i_%i.seed%i.pca%i.hdf5' % (model, batch0, batch1, seed, n_pcas))
            )
    return None 


def _divide_desi_trainingset_3intervals(model, batch0, batch1, seed=0): 
    ''' divide DESI training set into 3 wavelength intervals: 
      1. wave < 4500
      2. 4500 < wave < 6500
      3. 6500 < wave 
    '''
    # fsps wavelength 
    fwave   = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy') 
    wave    = np.load(fwave)

    wave_bin0 = (wave < 4500) 
    wave_bin1 = (wave >= 4500) & (wave < 6500) 
    wave_bin2 = (wave >= 6500) 

    # batches of fsps spectra
    batches = range(batch0, batch1+1)

    fspecs  = [os.path.join(UT.dat_dir(), 'speculator', 
        'DESI_%s.logspectrum_fsps_train.%i.seed%i.npy' % (model, i, seed)) for i in batches]

    for fspec in fspecs: 
        np.save(fspec.replace('.npy', '.wave_bin0.npy'), np.load(fspec)[:,wave_bin0])
        np.save(fspec.replace('.npy', '.wave_bin1.npy'), np.load(fspec)[:,wave_bin1])
        np.save(fspec.replace('.npy', '.wave_bin2.npy'), np.load(fspec)[:,wave_bin2])

    return None 


def train_desi_pca_3intervals(model, batch0, batch1, n_pcas, seed=0): 
    ''' train PCA for DESI simpledust or complexdust training sets for 3 wavelength
    intervals: 
      1. wave < 4500
      2. 4500 < wave < 6500
      3. 6500 < wave 

    '''
    # fsps wavelength 
    fwave   = os.path.join(UT.dat_dir(), 'speculator', 'wave_fsps.npy') 
    wave    = np.load(fwave)

    wave_bin0 = (wave < 4500) 
    wave_bin1 = (wave >= 4500) & (wave < 6500) 
    wave_bin2 = (wave >= 6500) 

    # batches of fsps spectra
    batches = range(batch0, batch1+1)
    
    # parameters 
    fthetas = [os.path.join(UT.dat_dir(), 'speculator',
        'DESI_%s.theta_train.%i.seed%i.npy' % (model, i, seed)) for i in batches]

    for i_bin, wave_bin, n_pca in zip(range(3), [wave_bin0, wave_bin1, wave_bin2], n_pcas): 
        # log(spectra) over wavelength bin 
        fspecs  = [os.path.join(UT.dat_dir(), 'speculator', 
            'DESI_%s.logspectrum_fsps_train.%i.seed%i.wave_bin%i.npy' % (model, i, seed, i_bin)) for i in batches]

        if model == 'simpledust': 
            # theta = [b1, b2, b3, b4, g1, g2, tau, tage]
            n_param = 8 
        elif model == 'complexdust': 
            # theta = [b1, b2, b3, b4, g1, g2, tau, dust1, dust2, tage]
            n_param = 10 
        n_wave = np.sum(wave_bin) 
        
        # train PCA basis 
        PCABasis = SpectrumPCA(
                n_parameters=n_param,       # number of parameters
                n_wavelengths=n_wave,       # number of wavelength values
                n_pcas=n_pca,               # number of pca coefficients to include in the basis 
                spectrum_filenames=fspecs,  # list of filenames containing the (un-normalized) log spectra for training the PCA
                parameter_filenames=fthetas, # list of filenames containing the corresponding parameter values
                parameter_selection=None) 

        PCABasis.compute_spectrum_parameters_shift_and_scale() # computes shifts and scales for (log) spectra and parameters
        PCABasis.train_pca()
        PCABasis.transform_and_stack_training_data(
                os.path.join(UT.dat_dir(), 'speculator', 
                    'DESI_%s.%i_%i.seed%i.wave_bin%i.pca%i' % (model, batch0, batch1, seed, i_bin, n_pca)), 
                retain=True) 
        # save to file 
        PCABasis._save_to_file(
                os.path.join(UT.dat_dir(), 'speculator', 
                    'DESI_%s.%i_%i.seed%i.wave_bin%i.pca%i.hdf5' % (model, batch0, batch1, seed, i_bin, n_pca))
                )
    return None 



if __name__=="__main__": 
    train_or_transform = sys.argv[1]
    model   = sys.argv[2]
    ibatch0 = int(sys.argv[3])
    ibatch1 = int(sys.argv[4])
    
    assert model in ['simpledust', 'complexdust'] 
    
    if train_or_transform == 'train': 
        n_pca0  = int(sys.argv[5]) 
        n_pca1  = int(sys.argv[6]) 
        n_pca2  = int(sys.argv[7]) 
        n_pcas = [n_pca0, n_pca1, n_pca2] 
        train_desi_pca_3intervals(model, ibatch0, ibatch1, n_pcas, seed=0)
    elif train_or_transform == 'divide': 
        _divide_desi_trainingset_3intervals(model, ibatch0, ibatch1, seed=0)
