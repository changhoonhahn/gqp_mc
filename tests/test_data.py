__all__ = ['test_spectra', 'test_photometry'] 

import pytest
# --- gqp_mc --- 
from gqp_mc import data as Data


@pytest.mark.parametrize("noise", ('none', 'bgs1', 'bgs2'))
def test_spectra(noise, sim='lgal', lib='bc03', sample='mini_mocha'): 
    print(noise)
    specs, meta = Data.Spectra(sim=sim, noise=noise, lib=lib, sample=sample)

    n_meta = len(meta['galid'])
    assert n_meta == len(meta['redshift']) 

    n_spec = specs['flux'].shape[0]
    assert n_meta == n_spec
    

@pytest.mark.parametrize("noise", ('none', 'legacy'))
def test_photometry(noise, sim='lgal', lib='bc03', sample='mini_mocha'): 
    photo, meta = Data.Photometry(sim=sim, noise=noise, lib=lib, sample=sample) 

    n_meta = len(meta['galid'])
    assert n_meta == len(meta['redshift']) 

    n_photo = photo['flux'].shape[0]
    assert n_meta == n_photo
