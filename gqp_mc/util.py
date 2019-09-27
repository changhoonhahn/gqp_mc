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

# --- units ---- 
def Lsun(): 
    return 3.846e33  # erg/s


def parsec(): 
    return 3.085677581467192e18  # in cm


def to_cgs(): # at 10pc 
    lsun = Lsun()
    pc = parsec()
    return lsun/(4.0 * np.pi * (10 * pc)**2) 


def c_light(): # AA/s
    return 2.998e18


def jansky_cgs(): 
    return 1e-23


def fig_tex(ffig, pdf=False): 
    ''' given filename of figure return a latex friendly file name
    '''
    path, ffig_base = os.path.split(ffig) 
    ext = ffig_base.rsplit('.', 1)[-1] 
    ffig_name = ffig_base.rsplit('.', 1)[0]

    _ffig_name = ffig_name.replace('.', '_') 
    if pdf: ext = 'pdf' 
    return os.path.join(path, '.'.join([_ffig_name, ext])) 

