import os 
import h5py 
import fsps
import pickle
import numpy as np 
from scipy.stats import sigmaclip
from scipy.special import gammainc
import scipy.interpolate as Interp
# --- astropy --- 
from astropy import units as U
from astropy.cosmology import Planck13 as cosmo
# --- speclite ---
from speclite import filters as specFilter
# --- gqp_mc --- 
from . import util as UT
from .firefly._firefly import hpf, curve_smoother, calculate_averages_pdf, convert_chis_to_probs


class Fitter(object): 
    def __init__(self): 
        self.prior = None   # prior object 

    def _check_mask(self, mask, wave_obs, flux_ivar_obs, zred): 
        ''' check that mask is sensible and mask out any parts of the 
        spectra where the ivar doesn't make sense. 
        '''
        if mask is None: 
            _mask = np.zeros(len(wave_obs)).astype(bool) 

        elif mask == 'emline': 
            # mask 20As around emission lines.
            w_lines = np.array([3728., 4861., 5007., 6564.]) * (1. + zred) 
            
            _mask = np.zeros(len(wave_obs)).astype(bool) 
            for wl in w_lines: 
                inemline = ((wave_obs > wl - 20) & (wave_obs < wl + 20))
                _mask = (_mask | inemline)

        elif isinstance(mask, np.ndarray): 
            # user input array 
            assert np.array_equal(mask, mask.astype(bool))
            assert len(mask) == len(wave_obs) 
            _mask = mask 
        else: 
            raise ValueError("mask = None, 'emline', or boolean array") 

        zero_err = ~np.isfinite(flux_ivar_obs)
        _mask = _mask | zero_err 
        return _mask 

    def _lnPrior(self, tt_arr, prior=None): 
        ''' log prior(theta). 
        '''
        assert prior is not None 

        _prior = prior(tt_arr) 
        if _prior == 0: return -np.inf 
        else: return np.log(_prior) 

    def _default_prior(self): 
        ''' set self.prior to be default prior 
        '''
        # thetas: mass, Z, dust2, tau
        prior_min = np.array([8., -3., 0., 0.1])
        prior_max = np.array([13., 1., 10., 10.])
        prior = UniformPrior(prior_min, prior_max)
        return prior  


class iFSPS(Fitter): 
    ''' inference wrapper that uses FSPS as its model --- essentially a light weight 
    version of prospector 
    
    Usage:: 
    
    for fitting spectra 
        >>> ifsps = iFSPS(model_name='vanilla')
        >>> bestfit = ifsps.MCMC_spec(wave_obs, flux_obs, flux_ivar_obs, zred, writeout='output_file', silent=False) 

    for fitting photometry 
        >>> ifsps = iFSPS(model_name='vanilla')
        >>> bestfit = ifsps.MCMC_photo(flux_obs, flux_ivar_obs, zred, writeout='output_file', silent=False) 

    :param model_name: (optional) 
        name of the model to use. This specifies the free parameters. For model_name == 'vanilla', 
        the free parameters are mass, Z, dust2, tau. (default: vanilla)

    :param cosmo: (optional) 
        astropy.cosmology object that specifies the cosmology.(default: astropy.cosmology.Planck13) 

    '''
    def __init__(self, model_name='vanilla', cosmo=cosmo): 
        self._init_model(model_name)
        self.cosmo      = cosmo # cosmology  
        self.ssp        = self._ssp_initiate() # initial ssp
        
    def MCMC_spectrophoto(self, wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs, zred, prior=None, 
            mask=None, bands='desi', nwalkers=100, burnin=100, niter=1000, maxiter=200000, opt_maxiter=100, 
            writeout=None, overwrite=False, silent=True): 
        ''' infer the posterior distribution of the free parameters given spectroscopy and photometry:
        observed wavelength, spectra flux, inverse variance flux, photometry, inv. variance photometry
        using MCMC. The function outputs a dictionary with the median theta of the posterior as well as 
        the 1sigma and 2sigma errors on the parameters (see below).

        :param wave_obs: 
            array of the observed wavelength
        
        :param flux_obs: 
            array of the observed flux __in units of ergs/s/cm^2/Ang__

        :param flux_ivar_obs: 
            array of the observed flux **inverse variance**. Not uncertainty!
        
        :param photo_obs: 
            array of the observed photometric flux __in units of nanomaggies__

        :param photo_ivar_obs: 
            array of the observed photometric flux **inverse variance**. Not uncertainty!

        :param zred: 
            float specifying the redshift of the observations  
    
        :param prior: 
             callable prior object (e.g. prior(theta)). See priors below.

        :param mask: (optional) 
            boolean array specifying where to mask the spectra. If mask == 'emline' the spectra
            is masked around emission lines at 3728., 4861., 5007., 6564. Angstroms. (default: None) 

        :param nwalkers: (optional) 
            number of walkers. (default: 100) 
        
        :param burnin: (optional) 
            int specifying the burnin. (default: 100) 
        
        :param nwalkers: (optional) 
            int specifying the number of iterations. (default: 1000) 

        :param maxiter: (default: 200000) 
            maximum number of iterations if `niter=adaptive`. 

        :param opt_maxiter: (default: 100)
            maximum number of iterations for initial optimizer before MCMC is
            run. 
        
        :param writeout: (optional) 
            string specifying the output file. If specified, everything in the output dictionary 
            is written out as well as the entire MCMC chain. (default: None) 

        :param silent: (optional) 
            If False, a bunch of messages will be shown 

        :return output: 
            dictionary that with keys: 
            - output['redshift'] : redshift 
            - output['theta_med'] : parameter value of the median posterior
            - output['theta_1sig_plus'] : 1sigma above median 
            - output['theta_2sig_plus'] : 2sigma above median 
            - output['theta_1sig_minus'] : 1sigma below median 
            - output['theta_2sig_minus'] : 2sigma below median 
            - output['wavelength_model'] : wavelength of best-fit model 
            - output['flux_model'] : flux of best-fit model 
            - output['wavelength_data'] : wavelength of observations 
            - output['flux_data'] : flux of observations 
            - output['flux_ivar_data'] = inverse variance of the observed flux. 
        '''
        # check mask for spectra 
        _mask = self._check_mask(mask, wave_obs, flux_ivar_obs, zred) 
        
        # get photometric bands  
        bands_list = self._get_bands(bands)
        assert len(bands_list) == len(photo_obs) 
        # get filters
        filters = specFilter.load_filters(*tuple(bands_list))

        # posterior function args and kwargs
        lnpost_args = (wave_obs, 
                flux_obs,               # 10^-17 ergs/s/cm^2/Ang
                flux_ivar_obs,          # 1/(10^-17 ergs/s/cm^2/Ang)^2
                photo_obs,              # nanomaggies
                photo_ivar_obs,         # 1/nanomaggies^2
                zred) 
        lnpost_kwargs = {
                'mask': _mask,          # emission line mask 
                'filters': filters,
                'prior': prior          # prior
                }
        self.data_type = 'specphoto'
        self.theta_names += ['f_fiber']
        
        # run emcee and get MCMC chains 
        output = self._emcee(
                self._lnPost_spectrophoto, 
                lnpost_args, 
                lnpost_kwargs, 
                nwalkers=nwalkers, 
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter,
                silent=silent,
                writeout=writeout, 
                overwrite=overwrite)
        return output  

    def MCMC_spec(self, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None, prior=None,
            nwalkers=100, burnin=100, niter=1000, maxiter=200000,
            opt_maxiter=100, writeout=None, overwrite=False, silent=True): 
        ''' infer the posterior distribution of the free parameters given observed
        wavelength, spectra flux, and inverse variance using MCMC. The function 
        outputs a dictionary with the median theta of the posterior as well as the 
        1sigma and 2sigma errors on the parameters (see below).

        :param wave_obs: 
            array of the observed wavelength
        
        :param flux_obs: 
            array of the observed flux __in units of ergs/s/cm^2/Ang__

        :param flux_ivar_obs: 
            array of the observed flux **inverse variance**. Not uncertainty!

        :param zred: 
            float specifying the redshift of the observations  

        :param mask: (optional) 
            boolean array specifying where to mask the spectra. If mask == 'emline' the spectra
            is masked around emission lines at 3728., 4861., 5007., 6564. Angstroms. (default: None) 

        :param prior: (optional) 
             callable prior object (e.g. prior(theta)). See priors below. (default: None) 

        :param nwalkers: (optional) 
            number of walkers. (default: 100) 
        
        :param burnin: (optional) 
            int specifying the burnin. (default: 100) 
        
        :param nwalkers: (optional) 
            int specifying the number of iterations. (default: 1000) 

        :param maxiter: (default: 200000) 
            maximum number of iterations if `niter=adaptive`. 

        :param opt_maxiter: (default: 100)
            maximum number of iterations for initial optimizer before MCMC is
            run. 
        
        :param writeout: (optional) 
            string specifying the output file. If specified, everything in the output dictionary 
            is written out as well as the entire MCMC chain. (default: None) 

        :param silent: (optional) 
            If False, a bunch of messages will be shown 

        :return output: 
            dictionary that with keys: 
            - output['redshift'] : redshift 
            - output['theta_med'] : parameter value of the median posterior
            - output['theta_1sig_plus'] : 1sigma above median 
            - output['theta_2sig_plus'] : 2sigma above median 
            - output['theta_1sig_minus'] : 1sigma below median 
            - output['theta_2sig_minus'] : 2sigma below median 
            - output['wavelength_model'] : wavelength of best-fit model 
            - output['flux_model'] : flux of best-fit model 
            - output['wavelength_data'] : wavelength of observations 
            - output['flux_data'] : flux of observations 
            - output['flux_ivar_data'] = inverse variance of the observed flux. 
        '''
        # check mask 
        _mask = self._check_mask(mask, wave_obs, flux_ivar_obs, zred) 

        # posterior function args and kwargs
        lnpost_args = (wave_obs, 
                flux_obs,        # 10^-17 ergs/s/cm^2/Ang
                flux_ivar_obs,   # 1/(10^-17 ergs/s/cm^2/Ang)^2
                zred) 
        lnpost_kwargs = {
                'mask': _mask,          # emission line mask 
                'prior': prior          # prior 
                }
        self.data_type = 'spec'

        # run emcee and get MCMC chains 
        output = self._emcee(
                self._lnPost, 
                lnpost_args, 
                lnpost_kwargs, 
                nwalkers=nwalkers,
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter,
                silent=silent,
                writeout=writeout, 
                overwrite=overwrite)
        return output  
    
    def MCMC_photo(self, photo_obs, photo_ivar_obs, zred, bands='desi', prior=None,
            nwalkers=100, burnin=100, niter=1000, maxiter=200000,
            opt_maxiter=100, writeout=None, overwrite=False, silent=True): 
        ''' infer the posterior distribution of the free parameters given observed
        photometric flux, and inverse variance using MCMC. The function 
        outputs a dictionary with the median theta of the posterior as well as the 
        1sigma and 2sigma errors on the parameters (see below).
        
        :param photo_obs: 
            array of the observed photometric flux __in units of nanomaggies__

        :param photo_ivar_obs: 
            array of the observed photometric flux **inverse variance**. Not uncertainty!

        :param zred: 
            float specifying the redshift of the observations  

        :param bands: 
            specify the photometric bands. Some pre-programmed bands available. e.g. 
            if bands == 'desi' then 
            bands_list = ['decam2014-g', 'decam2014-r', 'decam2014-z','wise2010-W1', 'wise2010-W2', 'wise2010-W3', 'wise2010-W4']. 
            (default: 'desi') 

        :param nwalkers: (optional) 
            number of walkers. (default: 100) 
        
        :param burnin: (optional) 
            int specifying the burnin. (default: 100) 
        
        :param nwalkers: (optional) 
            int specifying the number of iterations. (default: 1000) 
        
        :param maxiter: (default: 200000) 
            maximum number of iterations if `niter=adaptive`. 

        :param opt_maxiter: (default: 100)
            maximum number of iterations for initial optimizer before MCMC is
            run. 
        
        :param writeout: (optional) 
            string specifying the output file. If specified, everything in the output dictionary 
            is written out as well as the entire MCMC chain. (default: None) 

        :param silent: (optional) 
            If False, a bunch of messages will be shown 

        :return output: 
            dictionary that with keys: 
            - output['redshift'] : redshift 
            - output['theta_med'] : parameter value of the median posterior
            - output['theta_1sig_plus'] : 1sigma above median 
            - output['theta_2sig_plus'] : 2sigma above median 
            - output['theta_1sig_minus'] : 1sigma below median 
            - output['theta_2sig_minus'] : 2sigma below median 
            - output['wavelength_model'] : wavelength of best-fit model 
            - output['flux_model'] : flux of best-fit model 
            - output['wavelength_data'] : wavelength of observations 
            - output['flux_data'] : flux of observations 
            - output['flux_ivar_data'] = inverse variance of the observed flux. 
        '''
        ndim = prior.ndim
    
        # get photometric bands  
        bands_list = self._get_bands(bands)
        assert len(bands_list) == len(photo_obs) 
        # get filters
        filters = specFilter.load_filters(*tuple(bands_list))

        # posterior function args and kwargs
        lnpost_args = (
                photo_obs,               # nanomaggies
                photo_ivar_obs,         # 1/nanomaggies^2
                zred
                ) 
        lnpost_kwargs = {
                'filters': filters,
                'prior': prior  # prior object
                }
        self.data_type = 'photo'
    
        # run emcee and get MCMC chains 
        output = self._emcee(
                self._lnPost_photo, 
                lnpost_args,
                lnpost_kwargs, 
                nwalkers=nwalkers, 
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter,
                silent=silent,
                writeout=writeout,
                overwrite=overwrite)

        return output  

    def model(self, tt_arr, zred=0.1, wavelength=None): 
        ''' very simple wrapper for a fsps model with minimal overhead. Generates a
        spectra given the free parameters. This will be called by the inference method. 

        parameters
        ----------
        tt_arr : array 
            array of free parameters
        zred : float,array (default: 0.1) 
            The output wavelength and spectra are redshifted.
        wavelength : (default: None)  
            If specified, the model will interpolate the spectra to the specified 
            wavelengths.

        returns
        -------
        outwave : array
            output wavelength (angstroms) 
        outspec : array 
            spectra generated from FSPS model(theta) in units of 1e-17 * erg/s/cm^2/Angstrom
        '''
        tage    = self.cosmo.age(zred).value # age of the universe at z=zred in Gyr
        theta   = self._theta(tt_arr) 

        if self.model_name in ['vanilla', 'vanilla_kroupa']: 
            self.ssp.params['logzsol']  = np.log10(theta['Z']/0.0190) # log(z/zsun) 
            self.ssp.params['dust2']    = theta['dust2'] # dust2 parameter in fsps 
            self.ssp.params['tau']      = theta['tau'] # sfh parameter 

            w, ssp_lum = self.ssp.get_spectrum(tage=tage, peraa=True) 
        elif self.model_name == 'vanilla_complexdust': 
            self.ssp.params['logzsol']  = np.log10(theta['Z']/0.0190) # log(z/zsun) 
            self.ssp.params['dust1']    = theta['dust1'] # dust1 parameter in fsps 
            self.ssp.params['dust2']    = theta['dust2'] # dust2 parameter in fsps 
            self.ssp.params['dust_index'] = theta['dust_index'] # dust2 parameter in fsps 
            self.ssp.params['tau']      = theta['tau'] # sfh parameter 

            w, ssp_lum = self.ssp.get_spectrum(tage=tage, peraa=True) 

        # mass normalization
        lum_ssp = theta['mass'] * ssp_lum

        # redshift the spectra
        w_z = w * (1. + zred)
        d_lum = self.cosmo.luminosity_distance(zred).to(U.cm).value # luminosity distance in cm
        flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + zred) * 1e17 # 10^-17 ergs/s/cm^2/Ang

        if wavelength is None: 
            outwave = w_z
            outspec = flux_z
        else: 
            outwave = wavelength
            outspec = np.zeros(outwave.shape)
            outspec = np.interp(outwave, w_z, flux_z, left=0, right=0)

        return outwave, outspec 
   
    def model_photo(self, tt_arr, zred=0.1, filters=None, bands=None): 
        ''' very simple wrapper for a fsps model with minimal overhead. Generates photometry 
        in specified photometric bands 

        :param tt_arr:
            array of free parameters

        :param zred:
            redshift (default: 0.1) 

        :param filters:             
            speclite.filters filter object. Either filters or bands has to be specified. (default: None) 

        :param bands: (optional) 
            photometric bands to generate the photometry. Either bands or filters has to be 
            specified. (default: None)  

        :return outphoto:
            array of photometric fluxes in nanomaggies in the specified bands 
        '''
        if filters is None: 
            if bands is not None: 
                bands_list = self._get_bands(bands) # get list of bands 
                filters = specFilter.load_filters(*tuple(bands_list))
            else: 
                raise ValueError("specify either filters or bands") 

        w, spec = self.model(tt_arr, zred=zred) # get spectra  
    
        maggies = filters.get_ab_maggies(np.atleast_2d(spec) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom, 
            wavelength=w.flatten()*U.Angstrom) # maggies 

        return np.array(list(maggies[0])) * 1e9
   
    def _model_spectrophoto(self, tt_arr, zred=0.1, wavelength=None, filters=None, bands=None): 
        ''' very simple wrapper for a fsps model with minimal overhead. Generates photometry 
        in specified photometric bands 

        :param tt_arr:
            array of free parameters

        :param zred:
            redshift (default: 0.1) 

        :param filters:             
            speclite.filters filter object. Either filters or bands has to be specified. (default: None) 

        :param bands: (optional) 
            photometric bands to generate the photometry. Either bands or filters has to be 
            specified. (default: None)  

        :return outphoto:
            array of photometric fluxes in nanomaggies in the specified bands 
        '''
        if filters is None: 
            if bands is not None: 
                bands_list = self._get_bands(bands) # get list of bands 
                filters = specFilter.load_filters(*tuple(bands_list))
            else: 
                raise ValueError("specify either filters or bands") 

        w, spec = self.model(tt_arr, zred=zred) # get spectra  

        if wavelength is not None: 
            outspec = np.zeros(wavelength.shape)
            outspec = np.interp(wavelength, w, spec, left=0, right=0)
        
        try: 
            maggies = filters.get_ab_maggies(np.atleast_2d(spec) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom,
                wavelength=w.flatten()*U.Angstrom) # maggies 
        except ValueError: 
            print('redshift = %f' % zred)
            raise ValueError

        return outspec, np.array(list(maggies[0])) * 1e9
    
    def postprocess(self, mcmc=None, f_mcmc=None, thin=1,
            writeout=None, silent=True): 
        ''' postprocess MC chain and calculate SFR and Z for the chain using
        NMF bases 

        :param mcmc:
            output dictionary from MCMC_specphoto, MCMC_spec, or MCMC_photo
            that contains all the information from the MCMC sampling. 
            (default: None) 
        :param f_mcmc: 
            alternatively you can specify the hdf5 file name where the MCMC
            dictionary is saved. 
            (default: None) 
        :param thin: 
            Thin out MCMC chains by thin factor.
            (default: 1) 
        :param writeout: 
            optional file name you can specify to write out the post processed
            mcmc chain to file. 
        :param silent: 
            If False, print stuff
        :return mcmc_output: 
            postprocessed mcmc chain dictionary 
        '''
        if mcmc is None and f_mcmc is None: 
            raise ValueError
        if mcmc is not None and f_mcmc is not None:
            raise ValueError
        
        if f_mcmc is not None: 
            mcmc = self.read_chain(f_mcmc, silent=silent)

        # check the model names agree with one another 
        assert self.model_name == mcmc['model'] 

        theta_names = list(mcmc['theta_names'].astype(str))

        for prop in ['logsfr.100myr', 'logsfr.1gyr', 'logz.mw']: 
            if prop in theta_names: 
                raise ValueError('there are already SFR or Z for the chain')
    
        chain   = self._flatten_chain(mcmc['mcmc_chain'])[::thin] # flattened and thined chain
        zred    = mcmc['redshift']
        prior_ranges = mcmc['prior_range']

        # calculate 100 Myr and 1Gyr logSFRs from the posteriors 
        theta_names += ['logsfr.100myr']
        logsfr100myr = np.log10(self.get_SFR(chain.T, zred, dt=0.1))

        theta_names += ['logsfr.1gyr']
        logsfr1gyr = np.log10(self.get_SFR(chain.T, zred, dt=1.)) 

        theta_names += ['logz.mw'] # log10(mass weighted metallicity) 
        logzmw = np.log10(self.get_Z_MW(chain.T, zred)) 

        # concatenate SFRs to the markov chain  
        chain = np.concatenate([
            chain, 
            np.atleast_2d(logsfr100myr).T,
            np.atleast_2d(logsfr1gyr).T, 
            np.atleast_2d(logzmw).T], 
            axis=1) 
        prior_ranges = np.concatenate([prior_ranges, np.array([[-4., 4]]),
            np.array([[-4., 4]]), np.array([[-3., 1.]])], axis=0)  

        # get quanitles of the chain 
        lowlow, low, med, high, highhigh = np.percentile(chain, [2.5, 16, 50, 84, 97.5], axis=0)

        # update the mcmc_output with SFR, Z, and thinned chain 
        mcmc['theta_names']      = np.array(theta_names, dtype='S') 
        mcmc['theta_med']        = med 
        mcmc['theta_1sig_plus']  = high
        mcmc['theta_2sig_plus']  = highhigh
        mcmc['theta_1sig_minus'] = low
        mcmc['theta_2sig_minus'] = lowlow

        mcmc['prior_range']      = prior_ranges 

        mcmc['mcmc_chain']       = chain 

        if writeout is not None: 
            assert writeout != f_mcmc, "don't overwrite the MCMC chain file!"

            fh5  = h5py.File(writeout, 'w') 
            for k in mcmc.keys(): 
                fh5.create_dataset(k, data=mcmc[k]) 
            fh5.close() 
        return mcmc  

    def get_SFR(self, tt, zred, dt=1.):
        ''' given theta calculate SFR averaged over dt Gyr

        :param tt: 
            single paramter value. For model == 'vanilla', tt = [mass, Z, dust2, tau]. 
        :param zred: 
            redshift of galaxy 
        :param dt: 
            timescale of SFR in Gyr (default: 1.)
        :return sfr: 
            average SFR over dt Gyrs
        '''
        tage = self.cosmo.age(zred).value 
        assert tage > dt

        theta = self._theta(tt)
        if 'vanilla' in self.model_name:
            sf_trunc = 0.0
            sf_start = 0.0 
            sfh = 4
            tburst = 0.0
            fburst = 0.0 
            const = 0.0 
            # indices in theta 
        else: 
            raise NotImplementedError 

        if sfh == 1: power = 1
        elif sfh == 4: power = 2
        else: raise ValueError("get_SFR not supported for this SFH type.")
    
        tau = theta['tau']

        tb = (tburst - sf_start) / tau
        tmax = (tage - sf_start) / tau
        normalized_t0 = (tage - sf_start) / tau
        # clip at 0. This means that if tage < dt then we're not actually getting a fair average.
        normalized_t1 = np.clip(((tage - dt) - sf_start) / tau, 0, np.inf) 

        mass0 = gammainc(power, normalized_t0) / gammainc(power, tmax)
        mass1 = gammainc(power, normalized_t1) / gammainc(power, tmax)

        avsfr = (mass0 - mass1) / dt / 1e9  # Msun/yr
 
        #normalized_times = (np.array([tage, tage - dt]).T - sf_start) / tau
        #mass = gammainc(power, normalized_times) / gammainc(power, tmax)
        #avsfr = (mass[..., 0] - mass[..., 1]) / dt / 1e9  # Msun/yr
    
        # These lines change behavior when you request sfrs outside the range (sf_start + dt, tage)
        #avsfr[times > tage] = np.nan  # does not work for scalars
        #avsfr *= times <= tage
        #avsfr[np.isfinite(avsfr)] = 0.0 # does not work for scalars
        avsfr *= theta['mass']
        return np.clip(avsfr, 0, np.inf)

    def get_Z_MW(self, tt, zred):
        ''' given theta calculate mass weighted metallicity

        for iFSPS this is super simple since we assume single metallicty
        '''
        theta = self._theta(tt)
        return theta['Z'] 

    def _emcee(self, lnpost_fn, lnpost_args, lnpost_kwargs, nwalkers=100,
            burnin=100, niter='adaptive', maxiter=200000, opt_maxiter=1000, 
            silent=True, writeout=None, overwrite=False, **kwargs): 
        ''' Runs MCMC (using emcee) for a given log posterior function.

        :param lnpost_fn: 
            log(posterior) function 

        :param lnpost_args: 
            arguments for the lnpost_fn function

        :param lnpost_kwargs: 
            keyward arguments for lnpost_fn function

        :param nwalkers: (default: 100) 
            number of mcmc walkers

        :param burnin: (default: 100) 
            number of iterations for burnin. If using ACM, this is not terribly
            important. 

        :param niter: (default: 'adaptive') 
            number of MCMC iterations. If `niter=adaptive`, MCMC will use an
            adpative method based on periodic evaluations of the Gelman-Rubin
            diagnostic to assess convergences (recommended). 

        :param maxiter: (default: 100000) 
            maximum number of MCMC iterations for adaptive method. MCMC can
            always be restarted so, keep this at some sensible number.  

        :param opt_maxiter: (default: 1000) 
            maximum number of iterations for initial optimizer. 

        :param silent: (default: True) 
            If `False`, there will be periodic print statements with run details

        :param writeout (default: None)
            name of the writeout files that will be passed into temporary saving function

        :param overwrite (default: False) 
            If True, overwrite mcmc file. Otherwise, append to MCMC file  

        notes:
        -----
        * `skip_initial_state_check` included in emcee sampler. This *assumes*
          that the chain you're appending to is correct. Might be worth
          revisiting.
        '''
        import emcee
        
        self.nwalkers = nwalkers

        prior = lnpost_kwargs['prior']
        ndim = prior.ndim
        dprior = prior.max - prior.min

        _lnpost = lambda *args: -2. * lnpost_fn(*args, **lnpost_kwargs) 

        
        if (writeout is None) or (not os.path.isfile(writeout)) or (overwrite): 
            # if mcmc chain file does not exist or we want to overwrite
            import scipy.optimize as op

            # get initial theta by minimization 
            if not silent: print('getting initial theta') 
        
            min_result = op.minimize(
                    _lnpost, 
                    0.5*(prior.max + prior.min), # guess the middle of the prior 
                    args=lnpost_args, 
                    method='Nelder-Mead', 
                    options={'maxiter': opt_maxiter}
                    ) 
            tt0 = min_result['x'] 
            if not silent: print('initial theta = [%s]' % ', '.join([str(_t) for _t in tt0])) 
        
            # initial sampler 
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_fn, 
                    args=lnpost_args, kwargs=lnpost_kwargs)
            # initial walker positions 
            p0 = [tt0 + 1.e-4 * dprior * np.random.randn(ndim) for i in range(nwalkers)]

            # burn in 
            if not silent: print('running burn-in') 
            pos, prob, state = self.sampler.run_mcmc(p0, burnin)
            self.sampler.reset()
        else: 
            # file exists and we are appending to it. check that priors and
            # parameters agree. 
            if not silent: print('appending chain to ... %s' % writeout) 
            mcmc = self.read_chain(writeout, flat=False, silent=silent) 
            assert np.array_equal(mcmc['prior_range'].T[0], prior.min), 'prior range does not agree with existing chain'
            assert np.array_equal(mcmc['prior_range'].T[1], prior.max), 'prior range does not agree with existing chain'

            # check that theta names agree 
            for theta in self.theta_names: 
                assert theta in mcmc['theta_names'][...].astype(str), 'parameters are different than existing chain' 

            # get walker position from MCMC file 
            pos = mcmc['mcmc_chain'][-1,:,:] 
            
            # initial sampler 
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_fn, 
                    args=lnpost_args, kwargs=lnpost_kwargs)

        if not silent: print('running main chain') 
        if niter == 'adaptive': # adaptive MCMC 
            # ACM interval
            STEP = 1000
            index = 0
            _niter = 0 
            autocorr = np.empty(maxiter)
            
            old_tau = np.inf

            for sample in self.sampler.sample(pos, iterations=maxiter,
                    progress=False, skip_initial_state_check=True):
                if self.sampler.iteration % STEP:
                    continue
                if not silent: print(f'chain #{index+1}')
                tau = self.sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                convergent = np.all(tau * 11 < self.sampler.iteration)
                convergent &= np.all(np.abs(old_tau - tau) / tau < 0.05)

                if convergent: 
                    if not silent: print('converged!') 
                    break

                old_tau = tau
                
                _chain = self.sampler.get_chain()

                # write out incrementally
                if overwrite and index == 1: 
                    output = self._save_chains(_chain[_niter:,:,:],
                            lnpost_args, lnpost_kwargs,
                            writeout=writeout, overwrite=overwrite,
                            silent=silent, **kwargs) 
                else: 
                    output = self._save_chains(_chain[_niter:,:,:],
                            lnpost_args, lnpost_kwargs,
                            writeout=writeout, overwrite=False,
                            silent=silent, **kwargs)  
                _niter = _chain.shape[0]
        else:
            # run standard mcmc with niter iterations 
            assert isinstance(niter, int) 
            self.sampler.run_mcmc(pos, niter)
            _chain = self.sampler.get_chain()
                    
            output = self._save_chains(_chain,
                    lnpost_args, lnpost_kwargs, 
                    writeout=writeout, overwrite=overwrite, silent=silent) 
        
        return output 

    def _save_chains(self, chain, lnpost_args, lnpost_kwargs, writeout=None,
            overwrite=False, silent=False):
        ''' save MCMC chains to file. If file exists, it will append it to the
        hdf5 file. 
        '''
        if self.data_type == 'specphoto': 
            wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs, zred = lnpost_args
        elif self.data_type == 'spec': 
            wave_obs, flux_obs, flux_ivar_obs, zred = lnpost_args 
        elif self.data_type == 'photo': 
            photo_obs, photo_ivar_obs, zred = lnpost_args
        prior = lnpost_kwargs['prior'] 
    
        if not overwrite and writeout is not None and os.path.isfile(writeout): 
            if not silent: print('  appending to ... %s' % writeout)
            # if file exists and we don't want to overwrite it check that
            # priors and theta_names are consistent
            _mcmc = self.read_chain(writeout)  
            old_chain = _mcmc['mcmc_chain']
            
            # append chain to existing mcmc
            mcmc = h5py.File(writeout, 'a')  #  append 
            mcmc.create_dataset('mcmc_chain%i' % _mcmc['nchain'], data=chain)
            chain = np.concatenate([old_chain, chain], axis=0) 
            newfile = False
        else:   
            if writeout is not None: 
                if not silent: print('  writing to ... %s' % writeout)
                mcmc = h5py.File(writeout, 'w')  # write 
                mcmc.create_dataset('mcmc_chain0', data=chain) # first chain 
                newfile = True
    
        # get quanitles of the posterior
        flat_chain = self._flatten_chain(chain)
        lowlow, low, med, high, highhigh = np.percentile(flat_chain, [2.5, 16, 50, 84, 97.5], axis=0)
    
        output = {} 
        output['redshift'] = zred 
        output['model'] = self.model_name
        output['theta_names'] = np.array(self.theta_names, dtype='S') 
        output['theta_med'] = med 
        output['theta_1sig_plus'] = high
        output['theta_2sig_plus'] = highhigh
        output['theta_1sig_minus'] = low
        output['theta_2sig_minus'] = lowlow

        if self.data_type == 'specphoto': # spectrophotometric data 
            w_model, flux_model = self.model(
                    med[:-1], 
                    zred=zred, 
                    wavelength=wave_obs)
            photo_model = self.model_photo(
                    med[:-1], 
                    zred=zred,
                    filters=lnpost_kwargs['filters'])

            output['wavelength_model']  = w_model
            output['flux_spec_model']   = med[-1] * flux_model
            output['flux_photo_model']  = photo_model 
       
            output['wavelength_data']       = wave_obs
            output['flux_spec_data']        = flux_obs
            output['flux_spec_ivar_data']   = flux_ivar_obs
            output['flux_photo_data']       = photo_obs
            output['flux_photo_ivar_data']  = photo_ivar_obs
        elif self.data_type == 'spec': 
            w_model, flux_model = self.model(med, zred=zred, wavelength=wave_obs)

            output['wavelength_model']  = w_model
            output['flux_spec_model']   = flux_model
       
            output['wavelength_data']       = wave_obs
            output['flux_spec_data']        = flux_obs
            output['flux_spec_ivar_data']   = flux_ivar_obs
        elif self.data_type == 'photo': 
            photo_model = self.model_photo(
                    med, 
                    zred=zred, 
                    filters=lnpost_kwargs['filters'])

            output['flux_photo_model']      = photo_model
            output['flux_photo_data']       = photo_obs
            output['flux_photo_ivar_data']  = photo_ivar_obs
        
        # save prior range 
        output['prior_range'] = np.vstack([prior.min, prior.max]).T
        
        if writeout is None: 
            output['mcmc_chain'] = chain 
            return output 

        if not newfile: 
            # update these columns
            for k in output.keys(): 
                mcmc[k][...] = output[k]
        else: 
            # writeout these columns
            for k in output.keys(): 
                mcmc.create_dataset(k, data=output[k]) 
        mcmc.close() 
        output['mcmc_chain'] = chain 
        return output  

    def read_chain(self, fchain, flat=False, silent=True): 
        ''' read MCMC chain file. MCMC chains will be saved in chunks. This
        method reads in all the chunks and appends them together into one big
        chain 
        '''
        if not silent: print('reading %s' % fchain) 

        chains = h5py.File(fchain, 'r') 
    
        mcmc = {} 
        i_chains = []
        for k in chains.keys(): 
            if 'mcmc_chain' in k: 
                i_chains.append(int(k.replace('mcmc_chain', '')))
            else: 
                mcmc[k] = chains[k][...]
         
        nchain = np.max(i_chains)+1 # number of chains 
        mcmc['nchain'] = nchain

        chain_dsets = []
        for i in range(nchain):
            chain_dsets.append(chains['mcmc_chain%i' % i]) 
        if not silent: print('%i chains read' % nchain) 

        if not flat: 
            mcmc['mcmc_chain'] = np.concatenate(chain_dsets, axis=0) 
        else:
            mcmc['mcmc_chain'] = self._flatten_chain(np.concatenate(chain_dsets, axis=0)) 
        chains.close() 
        return mcmc 

    def _flatten_chain(self, chain): 
        ''' flatten mcmc chain. If chain object is 2D then it assumes it's
        already flattened. 
        '''
        if len(chain.shape) == 2: return chain # already flat 

        s = list(chain.shape[1:])
        s[0] = np.prod(chain.shape[:2]) 
        return chain.reshape(s)

    def _lnPost_spectrophoto(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs, zred, 
            mask=None, filters=None, bands=None, prior=None, debug=False): 
        ''' calculate the log posterior 

        :param tt_arr: 
            array of free parameters. last element is fspectrophoto factor 
        :param wave_obs:
            wavelength of 'observations'
        :param flux_obs: 
            flux of the observed spectra
        :param flux_ivar_obs :
            inverse variance of of the observed spectra
        :param photo_obs: 
            flux of the observed photometry maggies  
        :param photo_ivar_obs :
            inverse variance of of the observed photometry  
        :param zred:
            redshift of the 'observations'
        :param mask: (optional) 
            A boolean array that specifies what part of the spectra to mask 
            out. (default: None) 
        :param prior: (optional) 
            callable prior object 
        '''
        lp = self._lnPrior(tt_arr, prior=prior) # log prior
        if debug: 
            print('--- iFSPS._lnPost_spectrophoto ---') 
            print('  log Prior = %f' % lp) 
        if not np.isfinite(lp): 
            return -np.inf

        chi_tot = self._Chi2_spectrophoto(tt_arr[:-1], wave_obs, flux_obs,
                flux_ivar_obs, photo_obs, photo_ivar_obs, zred, mask=mask,
                f_fiber=tt_arr[-1], filters=filters, bands=bands, debug=debug) 
        if debug: print('  total Chi2 = %f' % chi_tot) 

        return lp - 0.5 * chi_tot

    def _Chi2_spectrophoto(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, photo_obs,
            photo_ivar_obs, zred, mask=None, f_fiber=1., filters=None,
            bands=None, debug=False): 
        ''' calculated the chi-squared between the data and model spectra. 
        '''
        # model(theta) 
        flux, photo = self._model_spectrophoto(tt_arr, zred=zred,
                wavelength=wave_obs, filters=filters, bands=bands, debug=debug) 
        # data - model(theta) with masking 
        dflux = (f_fiber * flux[~mask] - flux_obs[~mask]) 
        # calculate chi-squared for spectra
        _chi2_spec = np.sum(dflux**2 * flux_ivar_obs[~mask]) 
        if debug: 
            print('--- iFSPS._Chi2_spectrophoto ---') 
            print('  Spectroscopic Chi2 = %f' % _chi2_spec)
        # data - model(theta) for photometry  
        dphoto = (photo - photo_obs) 
        # calculate chi-squared for photometry 
        _chi2_photo = np.sum(dphoto**2 * photo_ivar_obs) 
        if debug: 
            print('  Photometric Chi2 = %f' % _chi2_photo)

        return _chi2_spec + _chi2_photo 

    def _lnPost(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None, prior=None): 
        ''' calculate the log posterior 

        :param tt_arr: 
            array of free parameters

        :param wave_obs:
            wavelength of 'observations'

        :param flux_obs: 
            flux of the observed spectra

        :param flux_ivar_obs :
            inverse variance of of the observed spectra

        :param zred:
            redshift of the 'observations'

        :param mask: (optional) 
            A boolean array that specifies what part of the spectra to mask 
            out. (default: None) 

        :param prior: (optional) 
            callable prior object. (default: None) 
        '''
        lp = self._lnPrior(tt_arr, prior=prior) # log prior
        if not np.isfinite(lp): 
            return -np.inf
        return lp - 0.5 * self._Chi2(tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred, mask=mask)

    def _Chi2(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None, f_fiber=1.): 
        ''' calculated the chi-squared between the data and model spectra. 
        '''
        # model(theta) 
        _, flux = self.model(tt_arr, zred=zred, wavelength=wave_obs) 
        # data - model(theta) with masking 
        dflux = (f_fiber * flux[~mask] - flux_obs[~mask]) 
        # calculate chi-squared
        _chi2 = np.sum(dflux**2 * flux_ivar_obs[~mask]) 
        return _chi2

    def _lnPost_spec(self, *args, **kwargs): 
        return self._lnPost(*args, **kwargs)
    
    def _lnPost_photo(self, tt_arr, flux_obs, flux_ivar_obs, zred, filters=None, bands=None, prior=None): 
        ''' calculate the log posterior for photometry

        :param tt_arr: 
            array of free parameters
        :param flux_obs: 
            flux of the observed photometry maggies  
        :param flux_ivar_obs :
            inverse variance of of the observed spectra

        :param zred:
            redshift of the 'observations'

        :param filters: 
            speclite.filters filter object. Either filters or bands has to be specified. (default: None) 

        :param bands: 
            photometric bands to generate the photometry. Either bands or filters has to be 
            specified. (default: None)  

        :param prior: (optional) 
            callable prior object. 
        '''
        lp = self._lnPrior(tt_arr, prior=prior) # log prior
        if not np.isfinite(lp): 
            return -np.inf
        return lp - 0.5 * self._Chi2_photo(tt_arr, flux_obs, flux_ivar_obs, zred, filters=filters, bands=bands)

    def _Chi2_photo(self, tt_arr, flux_obs, flux_ivar_obs, zred, filters=None, bands=None): 
        ''' calculated the chi-squared between the data and model photometry
        '''
        # model(theta) 
        flux = self.model_photo(tt_arr, zred=zred, filters=filters, bands=bands) 
        # data - model(theta) with masking 
        dflux = (flux - flux_obs) 
        # calculate chi-squared
        _chi2 = np.sum(dflux**2 * flux_ivar_obs) 
        #print(flux, _chi2) 
        return _chi2

    def _ssp_initiate(self): 
        ''' initialize sps (FSPS StellarPopulaiton object) 
        '''
        if self.model_name == 'vanilla': 
            ssp = fsps.StellarPopulation(
                    zcontinuous=1,          # interpolate metallicities
                    sfh=4,                  # sfh type 
                    dust_type=2,            # Calzetti et al. (2000) attenuation curve. 
                    imf_type=1)             # chabrier 
        elif self.model_name == 'vanilla_complexdust': 
            ssp = fsps.StellarPopulation(
                    zcontinuous=1,          # interpolate metallicities
                    sfh=4,                  # sfh type 
                    dust_type=4,            # Kriek & Conroy attenuation curve. 
                    imf_type=1)             # chabrier 
        elif self.model_name == 'vanilla_kroupa': 
            ssp = fsps.StellarPopulation(
                    zcontinuous=1,          # interpolate metallicities
                    sfh=4,                  # sfh type 
                    dust_type=2,            # Calzetti et al. (2000) attenuation curve. 
                    imf_type=2)             # chabrier 
        else: 
            raise NotImplementedError
        return ssp 
    
    def _theta(self, tt_arr): 
        ''' Given some theta 1D array return dictionary of parameter values. 
        This is synchronized with self.model_name
        '''
        theta = {} 
        if self.model_name in ['vanilla', 'vanilla_kroupa']: 
            # tt_arr columns: mass, Z, dust2, tau
            theta['mass']   = 10**tt_arr[0]
            theta['Z']      = 10**tt_arr[1]
            theta['dust2']  = tt_arr[2]
            theta['tau']    = tt_arr[3]
        elif self.model_name == 'vanilla_complexdust': 
            # tt_arr columns: mass, Z, dust1, dust2, dust_index, tau
            theta['mass']   = 10**tt_arr[0]
            theta['Z']      = 10**tt_arr[1]
            theta['dust1']  = tt_arr[2]
            theta['dust2']  = tt_arr[3]
            theta['dust_index']  = tt_arr[4]
            theta['tau']    = tt_arr[5]
        else: 
            raise NotImplementedError
        return theta

    def _get_bands(self, bands): 
        ''' given bands
        '''
        if isinstance(bands, str): 
            if bands == 'desi': 
                bands_list = ['decam2014-g', 'decam2014-r', 'decam2014-z', 'wise2010-W1', 'wise2010-W2']#, 'wise2010-W3', 'wise2010-W4']
            else: 
                raise NotImplementedError("specified bands not implemented") 
        elif isinstance(bands, list): 
            bands_list = bands
        else: 
            raise NotImplementedError("specified bands not implemented") 
        return bands_list 

    def _init_model(self, model_name): 
        ''' initialize theta values 
        '''
        self.model_name = model_name # store model name 
        if self.model_name in ['vanilla', 'vanilla_kroupa']: 
            names = ['logmstar', 'z_metal', 'dust2', 'tau']
        elif self.model_name == 'vanilla_complexdust': 
            names = ['logmstar', 'z_metal', 'dust1', 'dust2', 'dust_index', 'tau']
        self.theta_names = names 
        return None 

    def _default_prior(self, f_fiber_prior=None): 
        ''' return default prior object 
        '''
        if self.model_name in ['vanilla', 'vanilla_kroupa']: 
            # thetas: mass, Z, dust2, tau
            prior_min = [8., -3., 0., 0.1]
            prior_max = [13., 1., 10., 10.]
        elif self.model_name == 'vanilla_complexdust': 
            # thetas: mass, Z, dust1, dust2, dust_index, tau
            names = ['logmstar', 'z_metal', 'dust1', 'dust2', 'dust_index', 'tau']
            prior_min = [8., -3., 0., 0., -2.2, 0.1]
            prior_max = [13., 1., 4., 4., 0.4, 10.]

        if f_fiber_prior is not None: 
            prior_min.append(f_fiber_prior[0])
            prior_max.append(f_fiber_prior[1])

        return UniformPrior(np.array(prior_min), np.array(prior_max)) 


class iSpeculator(iFSPS):
    ''' inference that uses Speculator Alsing+(2020)  https://arxiv.org/abs/1911.11778 as its model. 
    This is a child class of iFSPS and consequently shares many of its methods. Usage is essentially 
    the same. 

    The MCMC is run on in the parameter space with transformed SFH basis coefficients rather than 
    the original SFH basis cofficient. This transformation is detailed in Betancourt2013 
    (https://arxiv.org/abs/1010.3436). **The output chain however is transformed back to the original
    SFH coefficients.**
    '''
    def __init__(self, model_name='emulator', cosmo=cosmo): 
        self._init_model(model_name)
        self.cosmo = cosmo # cosmology  
        self._load_model_params() # load emulator parameters
        self._load_NMF_bases() # read SFH and ZH basis 
        self._ssp_initiate() 

        # interpolators for speeding up cosmological calculations 
        _z = np.linspace(0, 0.4, 100)
        _tage = self.cosmo.age(_z).value
        _d_lum_cm = self.cosmo.luminosity_distance(_z).to(U.cm).value # luminosity distance in cm

        self._tage_z_interp = \
                Interp.InterpolatedUnivariateSpline(_z, _tage, k=3)
        self._d_lum_z_interp = \
                Interp.InterpolatedUnivariateSpline(_z, _d_lum_cm, k=3)
        
        # initiate p(SSFR) 
        self.ssfr_prior = None 
    
    def MCMC_spectrophoto(self, wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs, zred, prior=None, 
            mask=None, bands='desi', dirichlet_transform=False, nwalkers=100, burnin=100, niter=1000,
            maxiter=200000, opt_maxiter=100, writeout=None, overwrite=False, silent=True): 
        ''' infer the posterior distribution of the free parameters given spectroscopy and photometry:
        observed wavelength, spectra flux, inverse variance flux, photometry, inv. variance photometry
        using MCMC. The function outputs a dictionary with the median theta of the posterior as well as 
        the 1sigma and 2sigma errors on the parameters (see below).

        :param wave_obs: 
            array of the observed wavelength
        
        :param flux_obs: 
            array of the observed flux __in units of ergs/s/cm^2/Ang__

        :param flux_ivar_obs: 
            array of the observed flux **inverse variance**. Not uncertainty!
        
        :param photo_obs: 
            array of the observed photometric flux __in units of nanomaggies__

        :param photo_ivar_obs: 
            array of the observed photometric flux **inverse variance**. Not uncertainty!

        :param zred: 
            float specifying the redshift of the observations  
    
        :param prior: 
             callable prior object (e.g. prior(theta)). See priors below.

        :param mask: (optional) 
            boolean array specifying where to mask the spectra. If mask == 'emline' the spectra
            is masked around emission lines at 3728., 4861., 5007., 6564. Angstroms. (default: None) 

        :param bands: 
            photometric bands to generate the photometry. Either bands or filters has to be 
            specified. (default: None)  

        :param dirichlet_transform: 
            If True, apply warped_manifold_transform. Don't use this unless you
            explicitly know what you're doing. (default: False) 

        :param nwalkers: (optional) 
            number of walkers. (default: 100) 
        
        :param burnin: (optional) 
            int specifying the burnin. (default: 100) 
        
        :param nwalkers: (optional) 
            int specifying the number of iterations. (default: 1000) 
        
        :param maxiter: (default: 100000) 
            maximum number of MCMC iterations for adaptive method. MCMC can
            always be restarted so, keep this at some sensible number.  

        :param opt_maxiter: (default: 1000) 
            maximum number of iterations for initial optimizer. 
        
        :param writeout: (optional) 
            string specifying the output file. If specified, everything in the output dictionary 
            is written out as well as the entire MCMC chain. (default: None) 

        :param silent: (optional) 
            If False, a bunch of messages will be shown 

        :return output: 
            dictionary that with keys: 
            - output['redshift'] : redshift 
            - output['theta_med'] : parameter value of the median posterior
            - output['theta_1sig_plus'] : 1sigma above median 
            - output['theta_2sig_plus'] : 2sigma above median 
            - output['theta_1sig_minus'] : 1sigma below median 
            - output['theta_2sig_minus'] : 2sigma below median 
            - output['wavelength_model'] : wavelength of best-fit model 
            - output['flux_spec_model'] : flux of best-fit model spectrum
            - output['flux_photo_model'] : flux of best-fit model photometry 
            - output['wavelength_data'] : wavelength of observations 
            - output['flux_spec_data'] : flux of observed spectrum 
            - output['flux_spec_ivar_data'] = inverse variance of the observed flux. 
            - output['flux_photo_data'] : flux of observed photometry 
            - output['flux_photo_viar_data'] :  inverse variance of observed photometry 

        notes: 
        -----
        *   Because the priors for the SFH basis parameters have Dirichlet priors, we have to do a 
            transformation to get it to work
        '''
        # check mask for spectra 
        _mask = self._check_mask(mask, wave_obs, flux_ivar_obs, zred) 
        
        # get photometric bands  
        bands_list = self._get_bands(bands)
        assert len(bands_list) == len(photo_obs) 
        # get filters
        filters = specFilter.load_filters(*tuple(bands_list))

        if dirichlet_transform: 
            print('WARNING: you are applying a warped manifold transform') 
            print('  which mean you the SFH basis coefficients are sampled')
            print('  from a Dirichlet distribution!') 
            # check that the priors are sensible for the warped manifold
            # transform 
            assert np.min(prior.min) >= 0.
            assert np.max(prior.max) <= 1.

        # posterior function args and kwargs
        lnpost_args = (wave_obs, 
                flux_obs,               # 10^-17 ergs/s/cm^2/Ang
                flux_ivar_obs,          # 1/(10^-17 ergs/s/cm^2/Ang)^2
                photo_obs,              # nanomaggies
                photo_ivar_obs,         # 1/nanomaggies^2
                zred) 
        lnpost_kwargs = {
                'mask': _mask,          # emission line mask 
                'filters': filters,
                'prior': prior,         # prior
                'dirichlet_transform': dirichlet_transform
                }
        self.data_type = 'specphoto'
        self.theta_names += ['f_fiber']
        
        # run emcee and get MCMC chains 
        output = self._emcee(
                self._lnPost_spectrophoto, 
                lnpost_args, 
                lnpost_kwargs, 
                nwalkers=nwalkers,
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter, 
                silent=silent,
                writeout=writeout, 
                overwrite=overwrite, 
                dirichlet_transform=dirichlet_transform)

        return output  

    def MCMC_spec(self, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None, prior=None,
            dirichlet_transform=False, nwalkers=100, burnin=100, niter=1000, maxiter=200000, opt_maxiter=100, 
            writeout=None, overwrite=False, silent=True): 
        ''' infer the posterior distribution of the free parameters given observed
        wavelength, spectra flux, and inverse variance using MCMC. The function 
        outputs a dictionary with the median theta of the posterior as well as the 
        1sigma and 2sigma errors on the parameters (see below).

        :param wave_obs: 
            array of the observed wavelength
        
        :param flux_obs: 
            array of the observed flux __in units of ergs/s/cm^2/Ang__

        :param flux_ivar_obs: 
            array of the observed flux **inverse variance**. Not uncertainty!

        :param zred: 
            float specifying the redshift of the observations  

        :param mask: (optional) 
            boolean array specifying where to mask the spectra. If mask == 'emline' the spectra
            is masked around emission lines at 3728., 4861., 5007., 6564. Angstroms. (default: None) 

        :param prior: (optional) 
             callable prior object (e.g. prior(theta)). See priors below. (default: None) 
        
        :param dirichlet_transform: 
            If True, apply warped_manifold_transform. Don't use this unless you
            explicitly know what you're doing. (default: False) 

        :param nwalkers: (optional) 
            number of walkers. (default: 100) 
        
        :param burnin: (optional) 
            int specifying the burnin. (default: 100) 
        
        :param nwalkers: (optional) 
            int specifying the number of iterations. (default: 1000) 
        
        :param maxiter: (default: 100000) 
            maximum number of MCMC iterations for adaptive method. MCMC can
            always be restarted so, keep this at some sensible number.  

        :param opt_maxiter: (default: 1000) 
            maximum number of iterations for initial optimizer. 
        
        
        :param writeout: (optional) 
            string specifying the output file. If specified, everything in the output dictionary 
            is written out as well as the entire MCMC chain. (default: None) 

        :param silent: (optional) 
            If False, a bunch of messages will be shown 

        :return output: 
            dictionary that with keys: 
            - output['redshift'] : redshift 
            - output['theta_med'] : parameter value of the median posterior
            - output['theta_1sig_plus'] : 1sigma above median 
            - output['theta_2sig_plus'] : 2sigma above median 
            - output['theta_1sig_minus'] : 1sigma below median 
            - output['theta_2sig_minus'] : 2sigma below median 
            - output['wavelength_model'] : wavelength of best-fit model 
            - output['flux_spec_model'] : flux of best-fit model spectrum
            - output['wavelength_data'] : wavelength of observations 
            - output['flux_spec_data'] : flux of observed spectrum 
            - output['flux_spec_ivar_data'] = inverse variance of the observed flux. 
        '''
        # check mask 
        _mask = self._check_mask(mask, wave_obs, flux_ivar_obs, zred) 
        
        if dirichlet_transform: 
            print('WARNING: you are applying a warped manifold transform') 
            print('  which mean you the SFH basis coefficients are sampled')
            print('  from a Dirichlet distribution!') 
            # check that the priors are sensible for the warped manifold
            # transform 
            assert np.min(prior.min) >= 0.
            assert np.max(prior.max) <= 1.

        # posterior function args and kwargs
        lnpost_args = (wave_obs, 
                flux_obs,        # 10^-17 ergs/s/cm^2/Ang
                flux_ivar_obs,   # 1/(10^-17 ergs/s/cm^2/Ang)^2
                zred) 
        lnpost_kwargs = {
                'mask': _mask,          # emission line mask 
                'prior': prior,          # prior 
                'dirichlet_transform': dirichlet_transform
                }
        self.data_type = 'spec'

        # run emcee and get MCMC chains 
        output = self._emcee(
                self._lnPost, 
                lnpost_args, 
                lnpost_kwargs, 
                nwalkers=nwalkers, 
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter,
                opt_maxiter=opt_maxiter,
                silent=silent,
                writeout=writeout, 
                overwrite=overwrite, 
                dirichlet_transform=dirichlet_transform)
        return output  
    
    def MCMC_photo(self, photo_obs, photo_ivar_obs, zred, bands='desi', prior=None,
            dirichlet_transform=False, nwalkers=100, burnin=100, niter=1000,
            maxiter=200000, opt_maxiter=100, writeout=None, overwrite=False,
            silent=True): 
        ''' infer the posterior distribution of the free parameters given observed
        photometric flux, and inverse variance using MCMC. The function 
        outputs a dictionary with the median theta of the posterior as well as the 
        1sigma and 2sigma errors on the parameters (see below).
        
        :param photo_obs: 
            array of the observed photometric flux __in units of nanomaggies__

        :param photo_ivar_obs: 
            array of the observed photometric flux **inverse variance**. Not uncertainty!

        :param zred: 
            float specifying the redshift of the observations  

        :param bands: 
            specify the photometric bands. Some pre-programmed bands available. e.g. 
            if bands == 'desi' then 
            bands_list = ['decam2014-g', 'decam2014-r', 'decam2014-z','wise2010-W1', 'wise2010-W2', 'wise2010-W3', 'wise2010-W4']. 
            (default: 'desi') 
        
        :param dirichlet_transform: 
            If True, apply warped_manifold_transform. Don't use this unless you
            explicitly know what you're doing. (default: False) 

        :param nwalkers: (optional) 
            number of walkers. (default: 100) 
        
        :param burnin: (optional) 
            int specifying the burnin. (default: 100) 
        
        :param nwalkers: (optional) 
            int specifying the number of iterations. (default: 1000) 
        
        :param maxiter: (default: 100000) 
            maximum number of MCMC iterations for adaptive method. MCMC can
            always be restarted so, keep this at some sensible number.  

        :param opt_maxiter: (default: 1000) 
            maximum number of iterations for initial optimizer. 
        
        :param writeout: (optional) 
            string specifying the output file. If specified, everything in the output dictionary 
            is written out as well as the entire MCMC chain. (default: None) 

        :param silent: (optional) 
            If False, a bunch of messages will be shown 

        :return output: 
            dictionary that with keys: 
            - output['redshift'] : redshift 
            - output['theta_med'] : parameter value of the median posterior
            - output['theta_1sig_plus'] : 1sigma above median 
            - output['theta_2sig_plus'] : 2sigma above median 
            - output['theta_1sig_minus'] : 1sigma below median 
            - output['theta_2sig_minus'] : 2sigma below median 
            - output['wavelength_model'] : wavelength of best-fit model 
            - output['flux_photo_model'] : flux of best-fit model photometry 
            - output['flux_photo_data'] : flux of observed photometry 
            - output['flux_photo_ivar_data'] = inverse variance of the observed
              photometry
        '''
        # get photometric bands  
        bands_list = self._get_bands(bands)
        assert len(bands_list) == len(photo_obs) 
        # get filters
        filters = specFilter.load_filters(*tuple(bands_list))
        
        if dirichlet_transform: 
            print('WARNING: you are applying a warped manifold transform') 
            print('  which mean you the SFH basis coefficients are sampled')
            print('  from a Dirichlet distribution!') 
            # check that the priors are sensible for the warped manifold
            # transform 
            assert np.min(prior.min) >= 0.
            assert np.max(prior.max) <= 1.

        # posterior function args and kwargs
        lnpost_args = (
                photo_obs,               # nanomaggies
                photo_ivar_obs,         # 1/nanomaggies^2
                zred
                ) 
        lnpost_kwargs = {
                'filters': filters,
                'prior': prior,   # prior object
                'dirichlet_transform': dirichlet_transform
                }
        self.data_type = 'photo'
    
        # run emcee and get MCMC chains 
        output = self._emcee(
                self._lnPost_photo, 
                lnpost_args, 
                lnpost_kwargs, 
                nwalkers=nwalkers, 
                burnin=burnin, 
                niter=niter, 
                maxiter=maxiter, 
                opt_maxiter=opt_maxiter, 
                silent=silent,
                writeout=writeout,
                overwrite=overwrite,
                dirichlet_transform=dirichlet_transform)
        return output  

    def _lnPost_spectrophoto(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs, zred, 
            mask=None, filters=None, bands=None, prior=None, dirichlet_transform=False, debug=False): 
        ''' calculate the log posterior for spectrum and photometry 

        :param tt_arr: 
            array of free parameters. last element is fspectrophoto factor 
        :param wave_obs:
            wavelength of 'observations'
        :param flux_obs: 
            flux of the observed spectra
        :param flux_ivar_obs :
            inverse variance of of the observed spectra
        :param photo_obs: 
            flux of the observed photometry maggies  
        :param photo_ivar_obs :
            inverse variance of of the observed photometry  
        :param zred:
            redshift of the 'observations'
        :param mask: (optional) 
            A boolean array that specifies what part of the spectra to mask 
            out. (default: None) 
        :param prior: (optional) 
            callable prior object 
        :param dirichlet_transform: (optional) 
            If True, apply warped_manifold_transform so that the SFH basis
            coefficient is sampled from a Dirichlet prior
        '''
        lp = self._lnPrior(tt_arr, prior=prior) # log prior
        if debug: 
            print('iSpeculator._lnPost_spectrophoto: log Prior = %f' % lp) 
        if not np.isfinite(lp): 
            return -np.inf

        chi_tot = self._Chi2_spectrophoto(tt_arr[:-1], wave_obs, flux_obs,
                flux_ivar_obs, photo_obs, photo_ivar_obs, zred, mask=mask,
                f_fiber=tt_arr[-1], filters=filters, bands=bands, 
                dirichlet_transform=dirichlet_transform, debug=debug) 

        return lp - 0.5 * chi_tot

    def _Chi2_spectrophoto(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, photo_obs,
            photo_ivar_obs, zred, mask=None, f_fiber=1., filters=None,
            bands=None, dirichlet_transform=False, debug=False): 
        ''' calculated the chi-squared between the data and model spectra. 
        '''
        # model(theta) 
        flux, photo = self._model_spectrophoto(tt_arr, zred=zred,
                wavelength=wave_obs, filters=filters, bands=bands, 
                dirichlet_transform=dirichlet_transform, debug=debug) 
        # data - model(theta) with masking 
        dflux = (f_fiber * flux[~mask] - flux_obs[~mask]) 
        # calculate chi-squared for spectra
        _chi2_spec = np.sum(dflux**2 * flux_ivar_obs[~mask]) 
        if debug: 
            print('iSpeculator._Chi2_spectrophoto: Spectroscopic Chi2 = %f' % _chi2_spec)
        # data - model(theta) for photometry  
        dphoto = (photo - photo_obs) 
        # calculate chi-squared for photometry 
        _chi2_photo = np.sum(dphoto**2 * photo_ivar_obs) 
        if debug: 
            print('iSpeculator._Chi2_spectrophoto: Photometric Chi2 = %f' % _chi2_photo)

        if debug: print('iSpeculator._Chi2_spectrophoto: total Chi2 = %f' %
                (_chi2_spec + _chi2_photo))
        return _chi2_spec + _chi2_photo 

    def _lnPost(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred,
            mask=None, prior=None, dirichlet_transform=False, debug=False): 
        ''' calculate the log posterior for a spectrum

        :param tt_arr: 
            array of free parameters

        :param wave_obs:
            wavelength of 'observations'

        :param flux_obs: 
            flux of the observed spectra

        :param flux_ivar_obs :
            inverse variance of of the observed spectra

        :param zred:
            redshift of the 'observations'

        :param mask: (optional) 
            A boolean array that specifies what part of the spectra to mask 
            out. (default: None) 

        :param prior: (optional) 
            callable prior object. (default: None) 

        :param dirichlet_transform: (optional) 
            If True, apply warped_manifold_transform so that the SFH basis
            coefficient is sampled from a Dirichlet prior (default: False) 
    
        :param debug: (optional)
            If True, print debug statements (default: False) 
        '''
        lp = self._lnPrior(tt_arr, prior=prior) # log prior
        if not np.isfinite(lp): 
            return -np.inf
        
        if debug: print('iSpeculator._lnPost: log Prior = %f' % lp) 

        chi_tot = self._Chi2(tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred,
                mask=mask, dirichlet_transform=dirichlet_transform, debug=debug)

        return lp - 0.5 * chi_tot

    def _Chi2(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None,
            f_fiber=1., dirichlet_transform=False, debug=False): 
        ''' calculated the chi-squared between the data and model spectra. 

        :param dirichlet_transform: (optional) 
            If True, apply warped_manifold_transform so that the SFH basis
            coefficient is sampled from a Dirichlet prior (default: False) 
    
        :param debug: (optional)
            If True, print debug statements (default: False) 
        '''
        # model(theta) 
        _, flux = self.model(tt_arr, zred=zred, wavelength=wave_obs,
                dirichlet_transform=dirichlet_transform) 
        # data - model(theta) with masking 
        dflux = (f_fiber * flux[~mask] - flux_obs[~mask]) 
        # calculate chi-squared
        _chi2 = np.sum(dflux**2 * flux_ivar_obs[~mask]) 

        if debug: print('iSpeculator._Chi2: total Chi2 = %f' % _chi2) 

        return _chi2

    def _lnPost_photo(self, tt_arr, flux_obs, flux_ivar_obs, zred, filters=None, bands=None, 
            prior=None, dirichlet_transform=False, debug=False): 
        ''' calculate the log posterior for photometry

        :param tt_arr: 
            array of free parameters

        :param flux_obs: 
            flux of the observed photometry maggies  

        :param flux_ivar_obs :
            inverse variance of of the observed spectra

        :param zred:
            redshift of the 'observations'

        :param filters: 
            speclite.filters filter object. Either filters or bands has to be specified. (default: None) 

        :param bands: 
            photometric bands to generate the photometry. Either bands or filters has to be 
            specified. (default: None)  

        :param prior: (optional) 
            callable prior object. 
        
        :param dirichlet_transform: (optional) 
            If True, apply warped_manifold_transform so that the SFH basis
            coefficient is sampled from a Dirichlet prior (default: False) 
    
        :param debug: (optional)
            If True, print debug statements (default: False) 
        '''
        lp = self._lnPrior(tt_arr, prior=prior) # log prior
        if not np.isfinite(lp): 
            return -np.inf
        
        if debug: print('iSpeculator._lnPost_photo: log Prior = %f' % lp) 

        return lp - 0.5 * self._Chi2_photo(tt_arr, flux_obs, flux_ivar_obs,
                zred, filters=filters, bands=bands,
                dirichlet_transform=dirichlet_transform, debug=debug)

    def _Chi2_photo(self, tt_arr, flux_obs, flux_ivar_obs, zred, filters=None, bands=None,
            dirichlet_transform=False, debug=False): 
        ''' calculated the chi-squared between the data and model photometry
        
        :param dirichlet_transform: (optional) 
            If True, apply warped_manifold_transform so that the SFH basis
            coefficient is sampled from a Dirichlet prior (default: False) 
    
        :param debug: (optional)
            If True, print debug statements (default: False) 
        '''
        # model(theta) 
        flux = self.model_photo(tt_arr, zred=zred, filters=filters,
                bands=bands, dirichlet_transform=dirichlet_transform,
                debug=debug) 
        # data - model(theta) with masking 
        dflux = (flux - flux_obs) 
        # calculate chi-squared
        _chi2 = np.sum(dflux**2 * flux_ivar_obs) 
        
        if debug: print('iSpeculator._chi2_photo: total Chi2 = %f' % _chi2) 
        return _chi2

    def model(self, zz_arr, zred=0.1, wavelength=None, dirichlet_transform=False,
            debug=False): 
        ''' calls Speculator to computee SED given theta. theta[1:4] are the **transformed** SFH basis coefficients, 
        not the actual coefficients! This method is called by the inference method. 

        :param zz_arr:
            array of parameters: [logmstar, bSFH1, bSFH2, bSFH3, bSFH4, bZ1,
            bZ2, tau]. If you want the SFH basis coefficients bSFH1, bSFH2,
            bSFH3, bSFH4 to be sampled from a Dirichlet distribution see kwarg
            `dirichlet_transform`.

        :param zred: float,array (default: 0.1) 
            The output wavelength and spectra are redshifted.

        :param wavelength: (default: None)  
            If specified, the model will interpolate the spectra to the specified 
            wavelengths.

        :param dirichlet_transform:
            If True, transforms SFH basis coefficients to be within a Dirichlet
            distribution. (default: False) 

        returns
        -------
        outwave : array
            output wavelength (angstroms) 
        outspec : array 
            spectra generated from FSPS model(theta) in units of 1e-17 * erg/s/cm^2/Angstrom
        '''
        tage = self._tage_z_interp(zred)
        if debug: 
            print('iSpeculator.model: redshift = %f' % zred)
            print('iSpeculator.model: tage = %f' % tage) 

        # logmstar, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH, tau, tage
        ntheta = zz_arr.shape[0]
        tt_arr = np.zeros(ntheta+1) 
        tt_arr[:ntheta] = zz_arr
        tt_arr[-1] = tage

        if debug: print('iSpeculator.model: theta', tt_arr)
        if dirichlet_transform: 
            # transform SFH basis coefficients to Dirichlet distribution
            tt_arr[1:5] = self._transform_to_SFH_basis(zz_arr[1:5]) 
            if debug: print('iSpeculator.model: transformed theta', tt_arr)
        
        # input: b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH, tau, tage
        if self.model_name == 'emulator': 
            ssp_lum = self._emulator(tt_arr[1:]) 
            w = self._emu_wave
        elif 'fsps' in self.model_name: 
            w, ssp_lum = self._fsps_model(tt_arr[1:]) 
        if debug: print('iSpeculator.model: ssp lum', ssp_lum)

        # mass normalization
        lum_ssp = (10**tt_arr[0]) * ssp_lum

        # redshift the spectra
        w_z = w * (1. + zred)
        d_lum = self._d_lum_z_interp(zred) 
        flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum**2) / (1. + zred) * 1e17 # 10^-17 ergs/s/cm^2/Ang

        if wavelength is None: 
            outwave = w_z
            outspec = flux_z
        else: 
            outwave = wavelength
            outspec = np.interp(outwave, w_z, flux_z, left=0, right=0)
        return outwave, outspec 
    
    def model_photo(self, zz_arr, zred=0.1, filters=None, bands=None,
            dirichlet_transform=False, debug=False): 
        ''' very simple wrapper for a fsps model with minimal overhead. Generates photometry 
        in specified photometric bands 

        :param zz_arr:
            array of free parameters

        :param zred:
            redshift (default: 0.1) 

        :param filters:             
            speclite.filters filter object. Either filters or bands has to be specified. (default: None) 

        :param bands: (optional) 
            photometric bands to generate the photometry. Either bands or filters has to be 
            specified. (default: None)  

        :param dirichlet_transform:
            If True, transforms SFH basis coefficients to be within a Dirichlet
            distribution. (default: False) 

        :param debug: 
            If True, print debug statements. (default: False)  

        :return outphoto:
            array of photometric fluxes in nanomaggies in the specified bands 
        '''
        if filters is None: 
            if bands is not None: 
                bands_list = self._get_bands(bands) # get list of bands 
                filters = specFilter.load_filters(*tuple(bands_list))
            else: 
                raise ValueError("specify either filters or bands") 

        w, spec = self.model(zz_arr, zred=zred,
                dirichlet_transform=dirichlet_transform, debug=debug) # get SED  
         
        try: 
            maggies = filters.get_ab_maggies(np.atleast_2d(spec) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w*U.Angstrom) # maggies 
        except ValueError: 
            # this is a duct tape fix for the limited wavelength range of
            # speculator. In the future we need to retrain speculator to have a
            # wide wavelength range!
            # we *carelessly* zero pad the wavelengths assuming that the edges
            # of the transmission curve don't contribute to the photometry
            w_min, w_max = w.min(), w.max() 
            n_below = int(w_min - 1e3) + 1 # 1A resolution padding
            n_above = int(2e4 - w_max) + 1
            print('************************************************************') 
            print('WARNING: wavelength range of speculator does not cover the wavelength range of the bandpass filter!!!') 
            print('WARNING: wavelength range of speculator does not cover the wavelength range of the bandpass filter!!!') 
            print('WARNING: we currently zero pad it, which may result in incorrect photometry!') 

            w_pad = np.concatenate([
                np.linspace(1e3, w_min, n_below)[:-1], 
                w,
                np.linspace(w_max, 2e4, n_above)[1:]]) 
            spec_pad = np.concatenate([
                np.zeros(n_below-1), 
                spec, 
                np.zeros(n_above-1)])
            maggies = filters.get_ab_maggies(np.atleast_2d(spec_pad) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w_pad*U.Angstrom) # maggies 

        if debug: print('iSpeculator.model_photo: maggies', maggies) 

        return np.array(list(maggies[0])) * 1e9
    
    def _model_spectrophoto(self, tt_arr, zred=0.1, wavelength=None,
            filters=None, bands=None, dirichlet_transform=False, debug=False): 
        ''' very simple wrapper for a fsps model with minimal overhead. Generates photometry 
        in specified photometric bands 

        :param tt_arr:
            array of free parameters

        :param zred:
            redshift (default: 0.1) 

        :param filters:             
            speclite.filters filter object. Either filters or bands has to be specified. (default: None) 

        :param bands: (optional) 
            photometric bands to generate the photometry. Either bands or filters has to be 
            specified. (default: None)  

        :param dirichlet_transform: (optional) 
            If True, apply warped manifold transform so that the SFH basis
            coefficients are sampled from a Dirichlet distribution. 
            (default: False)  

        :return outphoto:
            array of photometric fluxes in nanomaggies in the specified bands 
        '''
        if filters is None: 
            if bands is not None: 
                bands_list = self._get_bands(bands) # get list of bands 
                filters = specFilter.load_filters(*tuple(bands_list))
            else: 
                raise ValueError("specify either filters or bands") 

        if debug: print('iSpeculator._model_spectrophoto: theta', tt_arr)

        w, spec = self.model(tt_arr, zred=zred,
                dirichlet_transform=dirichlet_transform, debug=debug) # get spectra  


        if wavelength is not None: 
            outspec = np.zeros(wavelength.shape)
            outspec = np.interp(wavelength, w, spec, left=0, right=0)
            if debug: print('iSpeculator._model_spectrophoto: out wave', wavelength)
        else: 
            outspec = spec 
        if debug: print('iSpeculator._model_spectrophoto: out flux', outspec)
    
        try: 
            maggies = filters.get_ab_maggies(np.atleast_2d(spec) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom,
                wavelength=w.flatten()*U.Angstrom) # maggies 
        except ValueError: 
            # this is a duct tape fix for the limited wavelength range of
            # speculator. In the future we need to retrain speculator to have a
            # wide wavelength range!
            # we *carelessly* zero pad the wavelengths assuming that the edges
            # of the transmission curve don't contribute to the photometry
            w_min, w_max = w.min(), w.max() 
            n_below = int(w_min - 1e3) + 1 # 1A resolution padding
            n_above = int(2e4 - w_max) + 1
            print('************************************************************') 
            print('WARNING: wavelength range of speculator does not cover the wavelength range of the bandpass filter!!!') 
            print('WARNING: wavelength range of speculator does not cover the wavelength range of the bandpass filter!!!') 
            print('WARNING: we currently zero pad it, which may result in incorrect photometry!') 

            w_pad = np.concatenate([
                np.linspace(1e3, w_min, n_below)[:-1], 
                w,
                np.linspace(w_max, 2e4, n_above)[1:]]) 
            spec_pad = np.concatenate([
                np.zeros(n_below-1), 
                spec, 
                np.zeros(n_above-1)])
            maggies = filters.get_ab_maggies(np.atleast_2d(spec_pad) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w_pad*U.Angstrom) # maggies 

        if debug: print('iSpeculator.model_spectrophoto: maggies', maggies) 

        return outspec, np.array(list(maggies[0])) * 1e9
   
    def _save_chains(self, _chain, lnpost_args, lnpost_kwargs, writeout=None, 
            dirichlet_transform=False, overwrite=False, silent=True):
        ''' save MCMC chains to file. If file exists, it will append it to the
        hdf5 file. 
        '''
        if self.data_type == 'specphoto': 
            wave_obs, flux_obs, flux_ivar_obs, photo_obs, photo_ivar_obs, zred = lnpost_args
        elif self.data_type == 'spec': 
            wave_obs, flux_obs, flux_ivar_obs, zred = lnpost_args 
        elif self.data_type == 'photo': 
            photo_obs, photo_ivar_obs, zred = lnpost_args
        prior = lnpost_kwargs['prior']
         
        # transform chain back to original SFH basis 
        niter, nwalker, nparam = _chain.shape
        if dirichlet_transform: 
            # flatten chain for transformation
            chain = self._flatten_chain(_chain).copy() 
            chain[:,1:5] = self._transform_to_SFH_basis(self._flatten_chain(_chain)[:,1:5]) 
            chain = chain.reshape(niter, nwalker, nparam) 
        else: 
            chain = _chain.copy() 

        if not overwrite and writeout is not None and os.path.isfile(writeout): 
            if not silent: print('  appending to ... %s' % writeout)
            # if file exists and we don't want to overwrite it check that
            # priors and theta_names are consistent
            _mcmc = self.read_chain(writeout, silent=silent)  
            old_chain = _mcmc['mcmc_chain']
            
            # append chain to existing mcmc
            mcmc = h5py.File(writeout, 'a')  #  append 
            mcmc.create_dataset('mcmc_chain%i' % _mcmc['nchain'], data=chain)

            chain = np.concatenate([old_chain, chain], axis=0) 
            newfile = False
        else:   
            if writeout is not None: 
                if not silent: print('  writing to ... %s' % writeout)
                mcmc = h5py.File(writeout, 'w')  # write 
                mcmc.create_dataset('mcmc_chain0', data=chain) # first chain 
                newfile = True
    
        # get quanitles of the posterior
        flat_chain = self._flatten_chain(chain)
        lowlow, low, med, high, highhigh = np.percentile(flat_chain, [2.5, 16, 50, 84, 97.5], axis=0)
    
        output = {} 
        output['redshift'] = zred 
        output['model'] = self.model_name
        output['theta_names'] = np.array(self.theta_names, dtype='S') 
        output['theta_med'] = med 
        output['theta_1sig_plus'] = high
        output['theta_2sig_plus'] = highhigh
        output['theta_1sig_minus'] = low
        output['theta_2sig_minus'] = lowlow

        if self.data_type == 'specphoto': # spectrophotometric data 
            w_model, flux_model = self.model(
                    med[:-1], 
                    zred=zred,
                    wavelength=wave_obs, 
                    dirichlet_transform=dirichlet_transform)
            photo_model = self.model_photo(
                    med[:-1], 
                    zred=zred, 
                    filters=lnpost_kwargs['filters'],
                    dirichlet_transform=dirichlet_transform)

            output['wavelength_model']  = w_model
            output['flux_spec_model']   = med[-1] * flux_model
            output['flux_photo_model']  = photo_model 
       
            output['wavelength_data']       = wave_obs
            output['flux_spec_data']        = flux_obs
            output['flux_spec_ivar_data']   = flux_ivar_obs
            output['flux_photo_data']       = photo_obs
            output['flux_photo_ivar_data']  = photo_ivar_obs
        elif self.data_type == 'spec': 
            w_model, flux_model = self.model(med, zred=zred, wavelength=wave_obs, 
                    dirichlet_transform=dirichlet_transform)
            output['wavelength_model'] = w_model
            output['flux_spec_model'] = flux_model
           
            output['wavelength_data'] = wave_obs
            output['flux_spec_data'] = flux_obs
            output['flux_spec_ivar_data'] = flux_ivar_obs
        elif self.data_type == 'photo': 
            photo_model = self.model_photo(
                    med, 
                    zred=zred,
                    filters=lnpost_kwargs['filters'], 
                    dirichlet_transform=dirichlet_transform)
            output['flux_photo_model'] = photo_model 
            output['flux_photo_data'] = photo_obs
            output['flux_photo_ivar_data'] = photo_ivar_obs

        # save prior range 
        output['prior_range'] = np.vstack([prior.min, prior.max]).T
        
        if writeout is None: 
            output['mcmc_chain'] = chain 
            return output 

        if not newfile: 
            # update these columns
            for k in output.keys(): 
                mcmc[k][...] = output[k]
        else: 
            # writeout these columns
            for k in output.keys(): 
                mcmc.create_dataset(k, data=output[k]) 
        mcmc.close() 
        output['mcmc_chain'] = chain 
        return output  

    def get_SFR(self, tt, zred, dt=1.):
        ''' given theta calculate SFR averaged over dt Gyr. 

        :param tt: 
           [log M*, b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH, tau]  b's here are the original
           SFH basis coefficients
        '''
        tage = self.cosmo.age(zred).value # age in Gyr
        assert tage > dt 
        t = np.linspace(0, tage, 50)

        tt_sfh = tt[1:5] # sfh bases 

        
        # normalized basis 
        _basis = np.array([self._sfh_basis[i](t)/np.trapz(self._sfh_basis[i](t), t) for i in range(4)])

        # caluclate normalized SFH
        if len(tt_sfh.shape) == 1: 
            sfh = np.sum(np.array([tt_sfh[i] * _basis[i] for i in range(4)]), axis=0)
        else: 
            sfh = np.sum(np.array([tt_sfh[i][:,None] * _basis[i][None,:] for i in range(4)]), axis=0)

        # add up the stellar mass formed during the dt time period 
        i_low = np.clip(np.argmin(np.abs(t - (tage - dt)), axis=0), None, 48) 
        avsfr = np.trapz(sfh[:,i_low:], t[i_low:]) / (tage - t[i_low]) / 1e9
        avsfr *= 10**tt[0]
        return np.clip(np.atleast_1d(avsfr), 0, np.inf)
    
    def get_Z_MW(self, tt, zred):
        ''' given theta calculate mass weighted metallicity using the ZH NMF
        bases. 
        '''
        tage = self.cosmo.age(zred).value # age in Gyr
        t = np.linspace(0, tage, 50)

        tt_sfh = tt[1:5] # sfh bases 
        tt_zh = tt[5:7] # zh bases 
        
        # normalized basis 
        _basis = np.array([self._sfh_basis[i](t)/np.trapz(self._sfh_basis[i](t), t) for i in range(4)])

        # caluclate normalized SFH
        if len(tt_sfh.shape) == 1: 
            sfh = np.sum(np.array([tt_sfh[i] * _basis[i] for i in range(4)]), axis=0)
        else: 
            sfh = np.sum(np.array([tt_sfh[i][:,None] * _basis[i][None,:] for i in range(4)]), axis=0)
        
        _z_basis = np.array([self._zh_basis[i](t) for i in range(2)]) 
        zh = np.sum(np.array([tt_zh[i][:,None] * _z_basis[i][None,:] for i in range(2)]), axis=0)
        
        # mass weighted average
        z_mw = np.trapz(zh * sfh, t) / np.trapz(sfh, t)
        return np.clip(np.atleast_1d(z_mw), 0, np.inf)
   
    def prior_correction(self, mcmc=None, f_mcmc=None, thin=1,
            writeout=None, dirichlet_transform=False, silent=True): 
        ''' postprocess importance weight to impose uniformative prior on SSFR  
        for different timescales. The method appends two set of importance
        weights, that will impose uniform priors on 1Gyr SSFR and 100Myr SSFR

        :param mcmc:
            output dictionary from MCMC_specphoto, MCMC_spec, or MCMC_photo
            that contains all the information from the MCMC sampling. 
            (default: None) 
        :param f_mcmc: 
            alternatively you can specify the hdf5 file name where the MCMC
            dictionary is saved. 
            (default: None) 
        :param thin: 
            Thin out MCMC chains by thin factor.
            (default: 1) 
        :param writeout: 
            optional file name you can specify to write out the post processed
            mcmc chain to file. 
            (default: None) 
        :param dirichlet_transform: 
            If True, the SFH basis coefficients were sampled from a Dirichlet
            prior; if False from a flat prior.
            (default: False) 
        :param silent: 
            If False, print stuff

        :return mcmc_output: 
            postprocessed mcmc chain dictionary 
        '''
        if mcmc is None and f_mcmc is None: 
            raise ValueError
        if mcmc is not None and f_mcmc is not None:
            raise ValueError
        
        if f_mcmc is not None: 
            mcmc = self.read_chain(f_mcmc, silent=silent)

        # check the model names agree with one another 
        assert self.model_name == mcmc['model'] 

        chain = self._flatten_chain(mcmc['mcmc_chain'])[::thin] # flattened and thined chain
        
        # importance weights for 1Gyr SSFR
        mcmc['w_uniform_ssfr1gyr'] = self.w_maxentropy_ssfrprior(
                chain, 
                dt_sfr=1., 
                log=False,
                redshift=mcmc['redshift'], 
                Nkde=10000,
                dirichlet_transform=dirichlet_transform)
        # importance weights for 100Myr SSFR
        mcmc['w_uniform_ssfr100myr'] = self.w_maxentropy_ssfrprior(
                chain, 
                dt_sfr=0.1, 
                log=False,
                redshift=mcmc['redshift'], 
                Nkde=10000,
                dirichlet_transform=dirichlet_transform)

        if writeout is not None: 
            assert writeout != f_mcmc, "don't overwrite the MCMC chain file!"

            fh5  = h5py.File(writeout, 'w') 
            for k in mcmc.keys(): 
                fh5.create_dataset(k, data=mcmc[k]) 
            fh5.close() 
        return mcmc  

    def w_maxentropy_ssfrprior(self, chain, dt_sfr=1., log=False,
            redshift=None, Nkde=10000, dirichlet_transform=False): 
        ''' calculate the importance sampling weights to correct for the MCMC
        chain so that we have an uninformative prior on SSFR averaged over
        `dt_sfr`. 
    
        The input chain are samples drawn from the inferred posteriors for our
        parameters: M*, b1, b2, b3, b4, g1, g2, etc. Regardless of whether we 
        use a Dirichlet or Uniform priors for b1, b2, b3, b4 (the SFH basis
        coefficients), we impose an undesirable prior on SSFR. We correct for
        this using importance weighting with weights derived from maximum
        entropy prior correction of Handley & Millea (2019). 

        :param chain: 
            mcmc chain. Assumes parameter order [M*, b1, b2, b3, b4, ... ]. I
            would suggest thinning out the chain here otherwise this method
            will take a while. (Nsteps x Nparam) 

        :param dt_sfr: 
            timescale of SFR in Gyrs. (default: 1)

        :param log: 
            If True, the weights will impose a flat prior on log(SSFR); if
            False on SSFR. (default: False) 

        :param redshift: 
            redshift of the galaxy. The prior on SSFR changes as a function of
            redshift. (default: None) 

        :param Nkde: 
            number of samples used to fit the p(SSFR) KDE

        :param dirichlet_transform: 
            If True, the SFH basis coefficients were sampled from a Dirichlet
            prior; if False from a flat prior.
        '''
        from scipy.stats import gaussian_kde as gkde
        
        if self.ssfr_prior is not None and np.all([
            (self.ssfr_prior['redshift'] == redshift), 
            (self.ssfr_prior['dt_sfr'] == dt_sfr), 
            (self.ssfr_prior['log'] == log), 
            (self.ssfr_prior['dirichlet_transform'] == dirichlet_transform)]):
            pass # use stored self.ssfr_prior 
        else: 
            self.ssfr_prior = {} 
            self.ssfr_prior['redshift'] = redshift
            self.ssfr_prior['dt_sfr'] = dt_sfr
            self.ssfr_prior['log'] = log 
            self.ssfr_prior['dirichlet_transform'] = dirichlet_transform
            
            # draw SFH basis coefficients from prior 
            if dirichlet_transform: 
                _b_prior = self._transform_to_SFH_basis(np.random.uniform(size=(Nkde,4)))
            else: 
                _b_prior = np.random.uniform(size=(Nkde,4))

            # calculate SSFR for the prior draws
            _ssfr_prior = self.get_SFR(
                    np.array([np.ones(Nkde), _b_prior[:,0], _b_prior[:,1], _b_prior[:,2], _b_prior[:,3]]),
                    redshift, dt=dt_sfr)

            if log: _ssfr_prior = np.log10(_ssfr_prior)
            else: _ssfr_prior *= 1e10 # this is so that we don't deal with 1e-11

            # fit KDE to p(SSFR(prior draws))
            self.ssfr_prior['kde'] = gkde(_ssfr_prior) 

        b1, b2, b3, b4 = chain[:,1:5].T 

        ssfr = self.get_SFR(
                np.array([np.ones(len(b1)), b1, b2, b3, b4]), redshift,
                dt=dt_sfr)
        if log: ssfr = np.log10(ssfr)
        else: ssfr *= 1e10 

        return 1./self.ssfr_prior['kde'].pdf(ssfr) 

    def _emulator(self, tt):
        ''' emulator for FSPS 

        :param tt: 
            array [b1SFH, b2SFH, b3SFH, b4SFH, g1ZH, g2ZH, tau, tage]
        :return flux: 
            FSPS SSP flux in units of Lsun/A
        '''
        # forward pass through the network
        act = []
        offset = np.log(np.sum(tt[0:4]))
        layers = [(self._transform_theta(tt) - self._emu_theta_mean)/self._emu_theta_std]
        for i in range(self._emu_n_layers-1):
       
            # linear network operation
            act.append(np.dot(layers[-1], self._emu_W[i]) + self._emu_b[i])

            # pass through activation function
            layers.append((self._emu_beta[i] + (1.-self._emu_beta[i])*1./(1.+np.exp(-self._emu_alpha[i]*act[-1])))*act[-1])

        # final (linear) layer -> (normalized) PCA coefficients
        layers.append(np.dot(layers[-1], self._emu_W[-1]) + self._emu_b[-1])

        # rescale PCA coefficients, multiply out PCA basis -> normalized spectrum, shift and re-scale spectrum -> output spectrum
        logflux = np.dot(layers[-1]*self._emu_pca_std + self._emu_pca_mean, self._emu_pcas)*self._emu_spec_std + self._emu_spec_mean + offset
        flux = np.exp(logflux)

        # normalization for the SSP SED because the SED is not normalized by
        # the integral of the SFH. This normalization should be incorporated
        # into the training set of Speculator rather than here, but for now
        # hacked together. 
        _t = np.linspace(0, tt[7], 50)
        norm_sfh = np.sum([tt[:4][i] * 
            self._sfh_basis[i](_t)/np.trapz(self._sfh_basis[i](_t), _t) for i
            in range(4)])
        flux /= norm_sfh
        return flux 
    
    def _transform_theta(self, theta):
        ''' initial transform applied to input parameters (network is trained over a 
        transformed parameter set)
        '''
        transformed_theta = np.copy(theta)
        transformed_theta[0] = np.sqrt(theta[0])
        transformed_theta[2] = np.sqrt(theta[2])
        return transformed_theta

    def _fsps_model(self, tt): 
        ''' same model as emulator but using fsps 

        :return flux: 
            FSPS SSP flux in units of Lsun/A
        '''
        if self.model_name == 'fsps': 
            tt_sfh  = tt[:4] 
            tt_zh   = tt[4:6]
            tt_dust = tt[6]
            tage    = tt[7] 
        elif self.model_name == 'fsps_complexdust': 
            tt_sfh      = tt[:4] 
            tt_zh       = tt[4:6]
            tt_dust1    = tt[6]
            tt_dust2    = tt[7]
            tt_dust_index = tt[8]
            tage        = tt[9] 

        _t = np.linspace(0, tage, 50)
        tages   = max(_t) - _t + 1e-8 

        # Compute SFH and ZH
        sfh = np.sum(np.array([
            tt_sfh[i] *
            self._sfh_basis[i](_t)/np.trapz(self._sfh_basis[i](_t), _t) 
            for i in range(4)]), 
            axis=0)
        zh = np.sum(np.array([
            tt_zh[i] * self._zh_basis[i](_t) 
            for i in range(2)]), 
            axis=0)
        
        for i, tage, m, z in zip(range(len(tages)), tages, sfh, zh): 
            if m <= 0 and i != 0: # no star formation in this bin 
                continue
            self._ssp.params['logzsol'] = np.log10(z/0.0190) # log(Z/Zsun)
            if self.model_name == 'fsps': 
                self._ssp.params['dust2'] = tt_dust 
            elif self.model_name == 'fsps_complexdust': 
                self._ssp.params['dust1'] = tt_dust1
                self._ssp.params['dust2'] = tt_dust2 
                self._ssp.params['dust_index'] = tt_dust_index
            wave_rest, lum_i = self._ssp.get_spectrum(tage=tage, peraa=True) # in units of Lsun/AA

            if i == 0: lum_ssp = np.zeros(len(wave_rest))
            lum_ssp += m * lum_i 

        lum_ssp /= np.sum(sfh)
        return wave_rest, lum_ssp
    
    def _load_model_params(self, filename=None): 
        ''' read in pickle file that contains all the parameters required for the emulator
        model

        :param filename: 
            name of model pickle file 
        '''
        if filename is None:
            fpkl = open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat', 
                'model_summary64.pkl'), 'rb')
        else: 
            fpkl = open(filename , 'rb')
        params = pickle.load(fpkl)
        fpkl.close()

        self._emu_W             = params[0] 
        self._emu_b             = params[1] 
        self._emu_alpha         = params[2]
        self._emu_beta          = params[3]
        self._emu_pcas          = params[4] 
        self._emu_pca_mean      = params[5]
        self._emu_pca_std       = params[6]
        self._emu_spec_mean     = params[7]
        self._emu_spec_std      = params[8]
        self._emu_theta_mean    = params[9]
        self._emu_theta_std     = params[10]
        self._emu_wave          = params[11]

        self._emu_n_layers = len(self._emu_W) # number of network layers
        return None 
        
    def _transform_to_SFH_basis(self, zarr):
        ''' MCMC is sampled in a warped manifold transformation of the original basis manifold
        as specified in Betancourt(2013). This function transforms back to the original manifold
    

        x_i = (\prod\limits_{k=1}^{i-1} z_k) * f 

        f = 1 - z_i         for i < m
        f = 1               for i = m 

        :param zarr: 
            N x m array 

        reference
        ---------
        * Betancourt(2013) - https://arxiv.org/pdf/1010.3436.pdf
        '''
        zarr    = np.atleast_2d(zarr)
        m       = zarr.shape[1]
        xarr    = np.empty(zarr.shape) 
    
        xarr[:,0] = 1. - zarr[:,0]
        for i in range(1,m-1): 
            xarr[:,i] = np.prod(zarr[:,:i], axis=1) * (1. - zarr[:,i]) 
        xarr[:,-1] = np.prod(zarr[:,:-1], axis=1) 

        return xarr 

    def _get_bands(self, bands): 
        ''' given bands
        '''
        if isinstance(bands, str): 
            if bands == 'desi': 
                bands_list = ['decam2014-g', 'decam2014-r', 'decam2014-z']#, 'wise2010-W1', 'wise2010-W2']#, 'wise2010-W3', 'wise2010-W4']
            else: 
                raise NotImplementedError("specified bands not implemented") 
        elif isinstance(bands, list): 
            bands_list = bands
        else: 
            raise NotImplementedError("specified bands not implemented") 
        return bands_list 
   
    def _init_model(self, model_name): 
        ''' initialize theta values 
        '''
        self.model_name = model_name # store model name 
        if self.model_name in ['emulator', 'fsps']: 
            names = ['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh',
                    'beta4_sfh', 'gamma1_zh', 'gamma2_zh', 'tau']
        elif self.model_name == 'fsps_complexdust': 
            names = ['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh',
                    'beta4_sfh', 'gamma1_zh', 'gamma2_zh', 'dust1', 'dust2',
                    'dust_index']
        else: 
            raise NotImplementedError 
        self.theta_names = names 
        return None 

    def _default_prior(self, f_fiber_prior=None): 
        ''' return default prior object. this prior spans the *transformed* SFH basis coefficients. 
        Because of this transformation, we use uniform priors. 

        :param f_fiber_prior: 
            [min, max] of f_fiber prior. if specified, f_fiber is added as an extra parameter.
            This is for spectrophotometric fitting 
        '''
        # M*, beta1', beta2', beta3', beta4', gamma1, gamma2, tau (dust2) 
        prior_min = [8., 0., 0., 0., 0., 6.9e-5, 6.9e-5, 0.]
        prior_max = [13., 1., 1., 1., 1., 7.3e-3, 7.3e-3, 3.]

        if self.model_name == 'fsps_complexdust': 
            # M*, beta1', beta2', beta3', beta4', gamma1, gamma2, dust1, dust2,
            # dust_index 
            prior_min = [8., 0., 0., 0., 0., 6.9e-5, 6.9e-5, 0., 0., -2.2]
            prior_max = [13., 1., 1., 1., 1., 7.3e-3, 7.3e-3, 3., 4., 0.4]

        if f_fiber_prior is not None: 
            prior_min.append(f_fiber_prior[0]) 
            prior_max.append(f_fiber_prior[1]) 

        return UniformPrior(np.array(prior_min), np.array(prior_max))

    def _load_NMF_bases(self): 
        ''' read NMF SFH and ZH bases and store it to object
        '''
        fsfh = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat',
                'NMF_2basis_SFH_components_nowgt_lin_Nc4.txt')
        fzh = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat',
                'NMF_2basis_Z_components_nowgt_lin_Nc2.txt') 
        ft = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat',
                'sfh_t_int.txt') 

        nmf_sfh = np.loadtxt(fsfh) 
        nmf_zh  = np.loadtxt(fzh) 
        nmf_t   = np.loadtxt(ft) # look back time 

        self._nmf_t_lookback    = nmf_t
        self._nmf_sfh_basis     = nmf_sfh 
        self._nmf_zh_basis      = nmf_zh

        Ncomp_sfh = self._nmf_sfh_basis.shape[0]
        Ncomp_zh = self._nmf_zh_basis.shape[0]
    
        self._sfh_basis = [
                Interp.InterpolatedUnivariateSpline(
                    max(self._nmf_t_lookback) - self._nmf_t_lookback, 
                    self._nmf_sfh_basis[i], k=1) 
                for i in range(Ncomp_sfh)
                ]
        self._zh_basis = [
                Interp.InterpolatedUnivariateSpline(
                    max(self._nmf_t_lookback) - self._nmf_t_lookback, 
                    self._nmf_zh_basis[i], k=1) 
                for i in range(Ncomp_zh)]
        return None 
    
    def _ssp_initiate(self):
        ''' for models that use fsps, initiate fsps.StellarPopulation object 
        '''
        if 'fsps' not in self.model_name: 
            self._ssp = None
        elif self.model_name == 'fsps': 
            # initalize fsps object
            self._ssp = fsps.StellarPopulation(
                zcontinuous=1, # SSPs are interpolated to the value of logzsol before the spectra and magnitudes are computed
                sfh=0, # single SSP
                imf_type=1, # chabrier
                dust_type=2 # Calzetti (2000) 
                )
        elif self.model_name == 'fsps_complexdust': 
            self._ssp = fsps.StellarPopulation(
                    zcontinuous=1,          # interpolate metallicities
                    sfh=0,                  # sfh type 
                    dust_type=4,            # Kriek & Conroy attenuation curve. 
                    imf_type=1)             # chabrier 
        return None  
    

class pseudoFirefly(Fitter): 
    ''' modified version of Firefly
    '''
    def __init__(self, model_name='vanilla', prior=None, cosmo=cosmo, downgraded=False, silent=True): 
        from estimations_3d import estimation

        self._model_init(model_name)
        self.cosmo = cosmo # cosmology  
        self._set_prior(prior) # set prior 

        # get models within prior  
        model_wave, model_flux, model_age, model_metal = self._get_model(downgraded=downgraded, silent=silent) 
        self.n_model        = model_flux.shape[0]
        self.model_wave     = model_wave
        self.model_flux     = model_flux
        self.model_age      = model_age
        self.model_metal    = model_metal

    def Fit_spec(self, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None, flux_unit=1e-17, fit_cap=1000, iter_max=10, writeout=None, silent=True): 
        ''' fit parameters given observed wavelength, spectra flux, and inverse variance using Firefly. 
        The function outputs a dictionary with the median theta as well as the 
        1sigma and 2sigma errors on the parameters (see below).

        :param wave_obs: 
            array of the observed wavelength
        
        :param flux_obs: 
            array of the observed flux __in units of ergs/s/cm^2/Ang__

        :param flux_ivar_obs: 
            array of the observed flux **inverse variance**. Not uncertainty!

        :param zred: 
            float specifying the redshift of the observations  

        :param mask: (optional) 
            boolean array specifying where to mask the spectra. If mask == 'emline' the spectra
            is masked around emission lines at 3728., 4861., 5007., 6564. Angstroms. (default: None) 

        :param writeout: (optional) 
            string specifying the output file. If specified, everything in the output dictionary 
            is written out as well as the entire MCMC chain. (default: None) 

        :param silent: (optional) 
            If False, a bunch of messages will be shown 

        :return output: 
            dictionary that with keys: 
            - output['redshift'] : redshift 
            - output['theta_med'] : parameter value of the median posterior
            - output['theta_1sig_plus'] : 1sigma above median 
            - output['theta_2sig_plus'] : 2sigma above median 
            - output['theta_1sig_minus'] : 1sigma below median 
            - output['theta_2sig_minus'] : 2sigma below median 
            - output['wavelength_model'] : wavelength of best-fit model 
            - output['flux_model'] : flux of best-fit model 
            - output['wavelength_data'] : wavelength of observations 
            - output['flux_data'] : flux of observations 
            - output['flux_ivar_data'] = inverse variance of the observed flux. 
        '''
        ndim = len(self.priors) 
    
        wave_obs_rest = wave_obs / (1. + zred) 

        d_lum = self.cosmo.luminosity_distance(zred).to(U.cm).value # luminosity distance 

        # check mask 
        _mask = self._check_mask(mask, wave_obs, flux_ivar_obs, zred) 

        if not silent: print("2. match model and data resolution")
        wave_obs_match, flux_obs_match, flux_ivar_obs_match, model_flux_raw = self._match_data_models(
                wave_obs_rest, 
                flux_obs, 
                flux_ivar_obs, 
                self.model_wave, 
                self.model_flux, 
                mask=_mask, 
                silent=silent)
        
        if not silent: print("Normalize the models to the median data flux")
        data_median     = np.median(flux_obs_match)
        model_median    = np.median(model_flux_raw, axis=1) 
        
        mass_factor = data_median/model_median
        model_flux = (model_flux_raw.T / model_median).T * data_median
        if self.dust_corr:
            if not silent: print("Dust correction")
            # determining attenuation curve through HPF fitting, 
            # apply attenuation curve to models and renormalise spectra
            best_ebv, attenuation_curve = self._determine_attenuation(
                    wave_obs_match, 
                    flux_obs_match, 
                    flux_ivar_obs_match, 
                    self.model_flux,
                    self.model_age, 
                    self.model_metal)

            model_flux_atten = np.zeros(model_flux_raw.shape)
            for m in range(len(model_flux_raw)):
                model_flux_atten[m] = attenuation_curve * model_flux_raw[m]

            data_median     = np.median(data_flux)
            model_median    = np.median(model_flux_atten, axis=1) 
            
            mass_factor     = data_median/model_median
            model_flux      = (model_flux_atten.T / model_median).T * data_median

        if not silent: print("Fit model to the data")
        # this is supposed to be the essential ingredient of firefly 
        # the code is a giant mess with nested functions and classes 
        # and global variables everywhere. 
        light_weights, chis, branch = self._fitter(wave_obs_match, flux_obs_match, flux_ivar_obs_match, model_flux, fit_cap=fit_cap, iter_max=iter_max)

        if not silent: print("mass weighted SSP contributions")
        # 5. Get mass-weighted SSP contributions using saved M/L ratio.
        unnorm_mass = light_weights * mass_factor
        mass_weights = (unnorm_mass.T / np.sum(unnorm_mass, axis=1)).T

        if not silent: print("chis into probabilities")
        dof = len(wave_obs_match)
        probs = convert_chis_to_probs(chis, dof)
                        
        if not silent: print("Calculating average properties and outputting")
        averages = calculate_averages_pdf(probs, 
                light_weights, 
                mass_weights, 
                unnorm_mass, 
                self.model_age, 
                self.model_metal, 
                self.pdf_sampling, 
                d_lum, 
                flux_unit)
    
        best_fit_index = np.argmin(chis)
        best_fit = np.dot(light_weights[best_fit_index], model_flux)
        
        # output dictionary 
        # theta =  total stellar mass, metallicity, age [Gyr]
        output = {} 
        output['redshift'] = zred
        output['theta_med'] = np.array([np.log10(averages['stellar_mass']), np.log10(averages['mass_metal']),  averages['mass_age']]) 
        for i_sig in [1, 2]: 
            output['theta_%isig_plus' % i_sig] = np.array([
                np.log10(averages['stellar_mass_%i_sig_plus' % i_sig]), 
                np.log10(averages['mass_metal_%i_sig_plus' % i_sig]), 
                averages['mass_age_%i_sig_plus' % i_sig]
                ]) 
            output['theta_%isig_minus' % i_sig] = np.array([
                np.log10(averages['stellar_mass_%i_sig_minus' % i_sig]),  
                np.log10(averages['mass_metal_%i_sig_minus' % i_sig]),
                averages['mass_age_%i_sig_minus' % i_sig] # Gyr
                ]) 

        output['wavelength_model'] = wave_obs_match
        output['flux_model'] = best_fit 
       
        output['wavelength_data'] = wave_obs
        output['flux_data'] = flux_obs
        output['flux_ivar_data'] = flux_ivar_obs
        
        if writeout is not None: 
            fh5  = h5py.File(writeout, 'w') 
            for k in output.keys(): 
                fh5.create_dataset(k, data=output[k]) 
            fh5.close() 
        return output  

    def _get_model(self, downgraded=False, silent=True):
        ''' Retrieves all relevant model files, in their downgraded format.
        If they aren't downgraded to the correct resolution / velocity dispersion,
        takes the base models in their native form and converts to downgraded files.

        descriptions of SSP in http://www-astro.physics.ox.ac.uk/~maraston/SSPn/SED/README_Mar05
        ''' 
        # directory with the stellar population models 
        dir_sp = os.path.join(UT.dat_dir(), 'firefly_sp_models', 'data') 

        model_files = []
        if self.ssp_model == 'm11': # Marastron & Stromback (2011) SP model 
            imf_dict = {'kroupa': 'kr', 'chabrier': 'cha', 'salpeter': 'ss'} 
            str_model = self.emp_stellar_lib+'.'+imf_dict[self.imf] 

            if downgraded :
                if self.emp_stellar_lib in ['MILES_UVextended', 'MILES_revisedIRslope']:
                    model_path = os.path.join(dir_sp, 'SSP_M11_MILES_downgraded','ssp_M11_'+str_model)
                else: 
                    model_path = os.path.join(dir_sp, 'SSP_M11_%s_downgraded' % self.emp_stellar_lib, 'ssp_M11_'+str_model)
            else:
                if self.emp_stellar_lib in ['MILES_UVextended', 'MILES_revisedIRslope']:
                    model_path = os.path.join(dir_sp, 'SSP_M11_MILES', 'ssp_M11_'+str_model)
                else:
                    model_path = os.path.join(dir_sp, 'SSP_M11_%s' % self.emp_stellar_lib, 'ssp_M11_'+str_model)
        
            # metallity grid
            Z_strs  = np.array(['z001', 'z002', 'z004', 'z0001.bhb', 'z0001.rhb', 'z10m4.bhb', 'z10m4.rhb']) #'z10m4'
            Z       = np.array([0.5, 1.0, 2.0, 10**-1.301, 10**-1.302, 10**-2.301, 10**-2.302])  #10**-2.300, 
            if self.imf == 'kroupa': 
                Z_strs  = np.concatenate([Z_strs, np.array(['z-0.6', 'z-0.9', 'z-1.2', 'z-1.6', 'z-1.9'])]) 
                Z       = np.concatenate([Z, np.array([10**-0.6, 10**-0.9, 10**-1.2, 10**-1.6, 10**-1.9])])
            else: 
                raise NotImplementedError
            
        elif self.ssp_model == 'm09':
            if downgraded: model_path = os.path.join(dir_sp, 'UVmodels_Marastonetal08b_downgraded')
            else: model_path = os.path.join(dir_sp, 'UVmodels_Marastonetal08b')

            Z_strs = np.array(['z001', 'z002', 'z004', 'z0001']) 
            Z = np.array([10**-0.3, 1.0, 10**0.3, 10**-1.3]) 
        else: 
            raise ValueError
    
        # keep only files with metallicities within the metallity prior range  
        inZlim = (Z > 10**self.priors[1][0]) & (Z < 10**self.priors[1][1])  # log Z/H priors
        metal = Z[inZlim] 
        metal_files = [model_path+zstr for zstr in Z_strs[inZlim]]
        if not silent: print('metal files included %s from %s' % (', '.join(metal_files), model_path))

        # constructs the model array
        flux_model, age_model, metal_model = [],[],[]
        for met, f_metal in zip(metal, metal_files):
            model_age, model_wave, model_flux = np.loadtxt(f_metal, unpack=True, usecols=[0,2,3]) 

            uniq_age = np.unique(model_age) # Gyr 
            for age in uniq_age:
                if (age < self.priors[2][0]) or (age > self.priors[2][1]): continue # outside of age limit 

                wave = model_wave[model_age == age] 
                flux = model_flux[model_age == age]
                    
                if self.data_wave_medium == 'vacuum':
                    # converts air wavelength to vacuum 
                    wavelength = airtovac(wave)
                else:
                    wavelength = wave 
                
                #if self.downgrade_models:
                # downgrades the model to match the data resolution 
                #mf = downgrade(wavelength, flux, deltal, self.vdisp_round, wave_instrument, r_instrument)
                #else: 
                mf = flux.copy() 

                # Reddens the models
                if self.ebv_mw != 0:
                    attenuations = unred(wavelength,ebv=0.0 - self.ebv_mw)
                    flux_model.append(mf*attenuations)
                else:
                    flux_model.append(mf)
                age_model.append(age) # Gyr
                metal_model.append(met) 

        return wavelength, np.array(flux_model), np.array(age_model), np.array(metal_model)

    def _match_data_models(self, w_d, flux_d, err_d, w_m, flux_m, mask=None, silent=True): 
        ''' match the wavelength resolution of the data and model
        '''
        w_min, w_max = w_d[~mask].min(), w_d[~mask].max()
        if not silent: print("%f < wavelength < %f" % (w_min, w_max))
        
        w_d     = w_d[~mask]
        flux_d  = flux_d[~mask] 
        err_d   = err_d[~mask]

        n_models = flux_m.shape[0]
        mask_m = ((w_m < w_min) | (w_m > w_max))
        assert np.sum(~mask_m) > 0, ('outside of model wavelength range %f < wave < %f' % (w_m.min(), w_m.max()))
        w_m     = w_m[~mask_m]
        flux_m  = flux_m[:,~mask_m] 

        if np.sum(~mask_m) >= np.sum(~mask): 
            if not silent: print("model has higher wavelength resolution")
            matched_wave    = w_d[1:-1]
            matched_data    = flux_d[1:-1]
            matched_error   = err_d[1:-1]
            matched_model   = np.zeros((n_models, len(matched_wave)))
            for i in range(n_models): 
                matched_model[i,:] = np.interp(matched_wave, w_m, flux_m[i,:]) # piecewise linear interpolation 
        else: 
            if not silent: print("model has lower wavelength resolution") 
            matched_wave = w_m[1:-1]
            matched_model = flux_m[:,1:-1]
            matched_data = np.interp(matched_wave, w_d, flux_d) # piecewise linear interpolation 
            matched_error = np.interp(matched_wave, w_d, err_d)
        return matched_wave, matched_data, matched_error, matched_model

    def _model_init(self, model_name): 
        ''' initalize 
        '''
        self.model_name = model_name # store model name 
        if model_name == 'vanilla': 
            self.ssp_model = 'm11' # SSP model using Maraston & Stromback (2011) based on ...
            self.emp_stellar_lib = 'MILES' # empirical stellar libraries 
            self.imf = 'kroupa'

            self.ebv_mw = 0.

            # some other settings that we won't play with for now...
            # dust stuff (pretty much untouched from original firefly) 
            self.dust_corr  = False 
            self.dust_law   = 'calzetti'
            self.max_ebv    = 1.5
            self.num_dust_vals = 200
            self.dust_smoothing_length = 200
            
            # specific fitting options
            self.pdf_sampling = 300 # sampling size when calculating the maximum pdf (100=recommended)
            # default is air, unless manga is used
            self.data_wave_medium = 'air'
        else: 
            raise NotImplementedError
        return None 

    def _determine_attenuation(self, wave, data_flux, ivar_flux, model_flux, age, metal):
        '''Determines the dust attenuation to be applied to the models based on the data.
        * 1. high pass filters the data and the models : makes hpf_model and hpf_data
        * 2. normalises the hpf_models to the median hpf_data
        * 3. fits the hpf models to data : chi2 maps

        :param wave: wavelength
        :param data_flux: data flux
        :param error_flux: error flux
        :param model_flux: model flux
        :param age: age
        :param metal: metallicity
        '''
        # 1. high pass filters the data and the models
        smoothing_length = self.dust_smoothing_length

        hpf_data    = hpf(data_flux)
        hpf_models  = np.zeros(model_flux.shape)
        for m in range(len(model_flux)):
            hpf_models[m] = hpf(model_flux[m])

        zero_dat = np.where((np.isnan(hpf_data)) & (np.isinf(hpf_data)))
        hpf_data[zero_dat] = 0.0
        hpf_models[:, zero_dat] = 0.0

        hpf_ivar = np.repeat(np.median(ivar_flux)/np.median(data_flux) * np.median(hpf_data), len(ivar_flux))
        hpf_ivar[zero_dat] = np.min(hpf_ivar) * 0.000001

        # 2. normalises the hpf_models to the median hpf_data
        hpf_data_norm = np.median(hpf_data)
        hpf_models_norm = np.median(hpf_models, axis=1)
        
        mass_factors = hpf_data_norm / hpf_models_norm 
        hpf_models *= mass_factors 

        # 3. fits the hpf models to data : chi2 maps
        hpf_weights, hpf_chis, hpf_branch = self._fitter(wave, hpf_data, hpf_ivar, hpf_models) 

        # 4. use best fit to determine the attenuation curve : fine_attenuation
        best_fit_index      = [np.argmin(hpf_chis)]
        best_fit_hpf        = np.dot(hpf_weights[best_fit_index], hpf_models)[0]
        best_fit            = np.dot(hpf_weights[best_fit_index], model_flux)[0]
        fine_attenuation    = (data_flux / best_fit) - (hpf_data / best_fit_hpf) + 1

        bad_atten           = np.isnan(fine_attenuation) | np.isinf(fine_attenuation)
        fine_attenuation[bad_atten] = 1.0
        hpf_ivar[bad_atten] = np.min(hpf_ivar)* 0.0000000001
        fine_attenuation    = fine_attenuation / np.median(fine_attenuation)

        # 5. propagates the hpf to the age and metallicity estimates
        av_age_hpf      = np.dot(hpf_weights, age)
        av_metal_hpf    = np.dot(hpf_weights, metal)

        # 6. smoothes the attenuation curve obtained
        smooth_attenuation = curve_smoother(wave, fine_attenuation, self.dust_smoothing_length) 
        
        # Fit a dust attenuation law to the best fit attenuation.
        num_laws = self.num_dust_vals
        ebv_arr  = np.arange(num_laws)/(self.max_ebv * num_laws * 1.0)
        chi_dust = np.zeros(num_laws)

        for ei,e in enumerate(ebv_arr): 
            if self.dust_law == 'calzetti':
                # Assume E(B-V) distributed 0 to max_ebv. 
                # Uses the attenuation curves of Calzetti (2000) for starburst galaxies.
                laws = np.array(dust_calzetti_py(e,wave)) 
            elif self.dust_law == 'allen':
                # Assume E(B-V) distributed 0 to max_ebv.
                # Uses the attenuation curves of Allen (1976) of the Milky Way.
                laws = np.array(dust_allen_py(e,wave))
            elif self.dust_law == 'prevot':
                # Assume E(B-V) distributed 0 to max_ebv.
                # Uses the attenuation curves of Prevot (1984) and Bouchert et al. (1985) 
                # for the Small Magellanic Cloud (SMC).
                laws = np.array(dust_prevot_py(e,wave))

            laws = laws/np.median(laws)
            chi_dust_arr    = (smooth_attenuation - laws)**2
            chi_clipped_arr = sigmaclip(chi_dust_arr, low=3.0, high=3.0)
            chi_clip_sq     = np.square(chi_clipped_arr[0])
            chi_dust[ei]    = np.sum(chi_clip_sq)

        dust_fit = ebv_arr[np.argmin(chi_dust)]
        return dust_fit, smooth_attenuation
    
    def _get_massloss_factors(self, imf, mass_per_ssp, age_per_ssp, metal_per_ssp): 
        ''' Gets the mass loss factors.'''
        f_ml = os.path.join(UT.dat_dir(),  'firefly_sp_models', 'data', 'massloss_%s.txt' % imf) 

        ML_metallicity, ML_age, ML_totM, ML_alive, ML_wd, ML_ns, ML_bh, ML_turnoff = np.loadtxt(f_ml, unpack=True, skiprows=2) 
        
        # first build the grids of the quantities. make sure they are in linear units.                  
        estimates = []
        for estimate, ML_ in zip(estimates, [ML_totM, ML_alive, ML_wd, ML_ns, ML_bh, ML_turnoff]): 
            estimates.append(estimation(10**ML_metallicity, ML_age, ML_))

        # now loop through ssps to find the nearest values for each.
        final_MLs = [[], [], [], [], [], []]
        final_gas_fraction = [] 
        for i in range(len(age_per_ssp)):
            for ii, estimate in enumerate(estimates): 
                new_ML_ = estimate.estimate(metal_per_ssp[i], age_per_ssp[i])
                final_MLs[ii].append(mass_per_ssp[i] * new_ML_)

                if ii == 0: new_ML_totM = new_ML_ # save to calculate gas fraction 

            final_gas_fraction.append(mass_per_ssp[i] - new_ML_totM) 

        return [np.array(final_ML) for final_ML in final_MLs]

    def _fitter(self, wavelength, data, ivar, models, fit_cap=1000, iter_max=10):
        '''
        '''
        n_models = self.n_model

        bic_n = np.log(len(wavelength))
        chi_models = (models - data) * np.sqrt(ivar)  

        # initialise fit objects over initial set of models
        fit_list, int_chi = [], [] 
        for im in range(len(models)):
            fit_first = {} 
            fit_first['index']          = im
            fit_first['weights']        = np.zeros(len(models)) 
            fit_first['weights'][im]    = 1 
            fit_first['branch_num']     = 0 
                
            # calculate chi-squared
            chi_arr                     = chi_models[im]
            chi_clipped_arr             = sigmaclip(chi_arr, low=3.0, high=3.0) # sigma clip chi 
            chi_clip_sq                 = np.square(chi_clipped_arr[0])
            fit_first['clipped_arr']    = (chi_arr > chi_clipped_arr[1]) & (chi_arr < chi_clipped_arr[2])
            fit_first['chi_squared']    = np.sum(chi_clip_sq)

            fit_list.append(fit_first) 
            int_chi.append(fit_first['chi_squared'])

        index_count = len(models) 

        # Find clipped array to remove artefacts:
        clipped_arr = fit_list[np.argmin(int_chi)]['clipped_arr']

        # fit_list is our initial guesses from which we will iterate
        final_fit_list = self._iterate(fit_list, bic_n, 0, clipped_arr, chi_models, fit_cap=fit_cap)

        chis = np.array([fdict['chi_squared'] for fdict in final_fit_list])
        best_fits = np.argsort(chis)    

        bf = len(best_fits)
        if bf > 10: bf = 10

        extra_fit_list  = self._mix(np.asarray(final_fit_list)[best_fits[:bf]].tolist(), final_fit_list, np.min(chis), index_count, chi_models, clipped_arr)
        extra_chis      = [fdict['chi_squared'] for fdict in extra_fit_list]
        total_fit_list  = final_fit_list + extra_fit_list

        weights     = np.array([fdict['weights'] for fdict in total_fit_list]) 
        chis        = np.array([fdict['chi_squared'] for fdict in total_fit_list]) 
        branches    = np.array([fdict['branch_num'] for fdict in total_fit_list]) 
        return weights, chis, branches

    def _iterate(self, fitlist, bic_n, iterate_count, clipped_arr, chi_models, fit_cap=None): 
        #print('iterate_count', iterate_count) 
        iterate_count += 1 

        count_new = 0

        len_list = len(fitlist)
        previous_chis = np.min([fdict['chi_squared'] for fdict in fitlist])

        for f in range(len_list):
            new_list = self._spawn_children(fitlist[f], iterate_count, clipped_arr, chi_models)

            len_new = len(new_list) 
            for n in range(len_new):
                # Check if any of the new spawned children represent better solutions
                new_chi = new_list[n]['chi_squared'] 
               
                if new_chi < (previous_chis - bic_n): # If they do, add them to the fit list!
                    count_new += 1
                    if count_new > fit_cap: break
                    fitlist.append(new_list[n])

            if count_new > fit_cap: break

        if count_new == 0:
            # If they don't, we have finished the iteration process and may return.
            return fitlist
        else:
            if iterate_count == 10: # hard max number of iterations 
                return fitlist
            fit_list_new = self._iterate(fitlist, bic_n, iterate_count, clipped_arr, chi_models, fit_cap=fit_cap)
            return fit_list_new

    def _spawn_children(self, fitdict, branch_num, clipped_arr, chi_models):
        # produce an array of children with iteratively increased weights
        fit_list = []
        new_weights = fitdict['weights'] * branch_num
        sum_weights = np.sum(new_weights)+1

        for im in range(self.n_model): 
            new_weights[im] += 1

            fitdict = {} 
            fitdict['weights'] = new_weights/sum_weights
            fitdict['branch_num'] = branch_num 
                            
            # Auto-calculate chi-squared
            index_weights   = np.nonzero(fitdict['weights']) # saves time!
            chi_arr         = np.dot(fitdict['weights'][index_weights], chi_models[index_weights])
            chi_clip_sq     = np.square(chi_arr[clipped_arr])
            fitdict['chi_squared'] = np.sum(chi_clip_sq)

            fit_list.append(fitdict) 
            new_weights[im] -= 1
        return fit_list
    
    def _mix(self, fit_list, full_fit_list, min_chi, index_count, chi_models, clipped_arr):
        """ Mix the best solutions together to improve error estimations.
        Never go more than 100 best solutions!  
        """
        # importance check:
        important_chi = min_chi + 10.0
        extra_fit_list = []#copy.copy(fit_list)

        for f1 in fit_list:
            for f2 in full_fit_list:
                for q in [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0]: 
                    fitdict = {} 
                    fitdict['weights'] = (f1['weights'] + q * f2['weights']) / (1. + q)
                    fitdict['branch_num'] = f1['branch_num'] + f2['branch_num'] 
                    fitdict['index'] = index_count
                    
                    # Auto-calculate chi-squared
                    index_weights = np.nonzero(fitdict['weights']) # saves time!
                    chi_arr = np.dot(fitdict['weights'][index_weights], chi_models[index_weights])
                    #if fitdict['branch_num'] == 0: 
                    #    chi_clipped_arr = sigmaclip(chi_arr, low=3.0, high=3.0)
                    #    chi_clip_sq = np.square(chi_clipped_arr[0])
                    #    self.clipped_arr = (chi_arr > chi_clipped_arr[1]) & (chi_arr < chi_clipped_arr[2])
                    #else: 
                    chi_clip_sq = np.square(chi_arr[clipped_arr])
                    fitdict['chi_squared'] = np.sum(chi_clip_sq)
                    index_count += 1 

                    extra_fit_list.append(fitdict)
        return extra_fit_list


class FlatDirichletPrior(object): 
    ''' flat dirichlet prior
    '''
    def __init__(self, ndim):
        self.ndim = ndim 
        self.alpha = [1. for i in range(ndim)] # alpha_i = 1 for flat prior 
        self._random = np.random.mtrand.RandomState()
        
    def __call__(self, theta=None):
        if theta is None:
            return self._random.dirchlet(self.alpha)
        else:
            return 1 if np.sum(theta) == 1 else 0
    
    def append(self, *arg, **kwargs): 
        raise ValueError("appending not supproted") 


class UniformPrior(object): 
    ''' uniform tophat prior
    
    :param min: 
        scalar or array of min values
    :param max: 
        scalar or array of max values
    '''
    
    def __init__(self, _min, _max):
        self.min = np.atleast_1d(_min)
        self.max = np.atleast_1d(_max)
        self.ndim = len(self.min) 
        self._random = np.random.mtrand.RandomState()
        assert self.min.shape == self.max.shape
        assert np.all(self.min < self.max)
        
    def __call__(self, theta=None):
        if theta is None:
            return np.array([self._random.uniform(mi, ma) for (mi, ma) in zip(self.min, self.max)])
        else:
            return 1 if np.all(theta < self.max) and np.all(theta >= self.min) else 0

    def append(self, _min, _max): 
        self.min = np.concatenate([self.min, _min]) 
        self.max = np.concatenate([self.max, _max]) 
        self.ndim = len(self.min)
        assert self.min.shape == self.max.shape
        assert np.all(self.min < self.max)
        return None 
