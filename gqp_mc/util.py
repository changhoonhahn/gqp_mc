'''

some utility functions 

'''
import os
from astropy.io import fits 


def readDESIspec(ffits): 
    ''' read DESI spectra fits file

    :params ffits: 
        name of fits file  
    
    :returns spec:
        dictionary of spectra
    '''
    fitobj = fits.open(ffits)
    
    spec = {} 
    for i_k, k in enumerate(['wave', 'flux', 'ivar']): 
        spec[k+'_b'] = fitobj[2+i_k].data
        spec[k+'_r'] = fitobj[7+i_k].data
        spec[k+'_z'] = fitobj[12+i_k].data
    return spec 


def check_env(): 
    if os.environ.get('GQPMC_DIR') is None: 
        raise ValueError("set $GQPMC_DIR in bashrc file!") 
    return None


def dat_dir(): 
    return os.environ.get('GQPMC_DIR') 


def lgal_dir(): 
    return os.path.join(dat_dir(), 'Lgal')

