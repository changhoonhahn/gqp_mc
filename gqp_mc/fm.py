'''

submodule for forward modeling spectrophotometry  


'''
import os 
import numpy as np 
from speclite import filters as specFilter


def Photo_DESI(wave, spectra): 
    ''' generate photometry by convolving the input spectrum with DECAM and WISE 
    bandpasses: g, r, z, W1, W2, W3, W4 filters. 

    :param wave: 
        wavelength of input spectra in Angstroms. 2D array Nspec x Nwave.

    :param fluxes: 
        fluxes of input spectra. This should be noiseless source spectra. 
        2D array Nspec x Nwave. In units of 10e-17 erg/s/cm2/A 
    '''
    assert wave.shape[0] == spectra.shape[1] 
    n_spec = spectra.shape[0] # number of spectra 

    from astropy import units as U
    
    # load DECAM g, r, z and WISE W1-4
    filter_response = specFilter.load_filters(
            'decam2014-g', 'decam2014-r', 'decam2014-z',
            'wise2010-W1', 'wise2010-W2', 'wise2010-W3', 'wise2010-W4')
   
    # apply filters
    fluxes = np.zeros((n_spec, 7)) # photometric flux in nanomaggies 
    for i in range(n_spec): 
        spectrum = spectra[i] 

        # apply filters
        flux = np.array(filter_response.get_ab_maggies(
                np.atleast_2d(spectrum) * 1e-17 * U.erg/U.s/U.cm**2/U.Angstrom, 
                wave*U.Angstrom))
        # convert to nanomaggies 
        fluxes[i,:] = 1e9 * np.array([flux[0][0], flux[0][1], flux[0][2], flux[0][3], flux[0][4], flux[0][5], flux[0][6]]) 
    
    # calculate magnitudes (not advised due to NaNs) 
    mags = 22.5 - 2.5 * np.log10(fluxes) 
    return fluxes, mags 


def Spec_BGS(wave, flux, exptime, airmass, Isky, filename=None):
    ''' Given noiseless spectra, simulate noisy BGS spectra with Isky 
    sky brightness, exptime sec exposure time, and airmass. Wrapper for 
    FM.fakeDESIspec().simExposure  
    
    :param wave: 
        wavelength of spectra. Nwave
    :param flux: 
        noiseless spectra in units of 1e-17 erg/s/cm2/A. Nspec x Nwave
    :param exptime: 
        exposure time 
    :param airmass: 
        airmass 
    :param Isky: 
        [wave_sky, sky_brightness]. sky brightness is in units of 
        1e-17 erg / Ang / arcsec^2 / cm^2 / sec
    :param filename: 
        If specified, the output fits file. (default: None) 

    :return bgs_spec: 
        data structure with all BGS data from the DESI spectrographs: 
        bgs.wave['b'], bgs.wave['r'], bgs.wave['z'] 
        bgs.flux['b'], bgs.flux['r'], bgs.flux['z'] 
        bgs.ivar['b'], bgs.ivar['r'], bgs.ivar['z'] 
    '''
    # requires desiutil, desimodel, desisim, desispec, desitarget,
    # also requires numba, fitsio, healpy, pandas, astroplan... shoot me in the face!
    from feasibgs import forwardmodel as FM 

    fdesi = FM.fakeDESIspec()
    bgs_spec = fdesi.simExposure(wave, flux, exptime=exptime, airmass=airmass, Isky=Isky, filename=filename) 
    return bgs_spec 
