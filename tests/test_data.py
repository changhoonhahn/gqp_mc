__all__ = ['test_spectra', 'test_photometry'] 

import pytest
from itertools import product
# --- gqp_mc --- 
from gqp_mc import data as Data


@pytest.mark.parametrize("noise", ('none', 'bgs1', 'bgs2'))
def test_spectra(noise, sim='lgal', lib='bc03', sample='spectral_challenge'): 
    print(noise)
    specs, meta = Data.Spectra(sim=sim, noise=noise, lib=lib, sample=sample)

    n_meta = len(meta['galid'])
    assert n_meta == len(meta['redshift']) 

    if noise == 'none': 
        n_spec = specs['flux_dust'].shape[0]
    elif 'bgs' in noise: 
        n_spec = specs['flux_dust_b'].shape[0]

    assert n_meta == n_spec
    

def test_photometry(sim='lgal', lib='bc03', sample='spectral_challenge'): 
    photo, meta = Data.Photometry(sim=sim, lib=lib, sample=sample) 

    n_meta = len(meta['galid'])
    assert n_meta == len(meta['redshift']) 

    n_photo = len(photo['flux_nodust_g']) 
    assert n_meta == n_photo
