import fsps
import numpy as np 
# --- astropy --- 
from astropy import units as U
from astropy.cosmology import Planck13 as cosmo
# --- speclite ---
from speclite import filters as specFilter
# --- gqp_mc --- 
from . import util as UT


class Fitter(object): 
    def __init__(self): 
        pass 


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
        the free parameters are mass, Z, tage, dust2, tau. (default: vanilla)

    :param prior: (optional) 
        list of tuples specifying the prior of the free parameters. (default: None) 

    :cosmo : (optional) 
        astropy.cosmology object that specifies the cosmology.(default: astropy.cosmology.Planck13) 

    '''
    def __init__(self, model_name='vanilla', prior=None, cosmo=cosmo): 
        self.model_name = model_name # store model name 
        self.cosmo = cosmo # cosmology  
        self.ssp = self._ssp_initiate() # initial ssp
        self._set_prior(prior) # set prior 

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
        zred    = np.atleast_1d(zred)
        theta   = self._theta(tt_arr) 
        ntheta  = len(theta['mass']) 
        mfrac   = np.zeros(ntheta)

        if self.model_name == 'vanilla': 
            for i, t, z, d, tau in zip(range(ntheta), theta['tage'], theta['Z'], theta['dust2'], theta['tau']): 
                self.ssp.params['logzsol']  = np.log10(z/0.0190) # log(z/zsun) 
                self.ssp.params['dust2']    = d # dust2 parameter in fsps 
                self.ssp.params['tau']      = tau # sfh parameter 

                w, ssp_lum = self.ssp.get_spectrum(tage=t, peraa=True) 

                mfrac[i] = self.ssp.stellar_mass
                if i == 0: 
                    ws          = np.zeros((ntheta, len(w)))
                    ssp_lums    = np.zeros((ntheta, len(w)))
                ws[i,:] = w 
                ssp_lums[i,:] = ssp_lum

        elif self.model_name == 'dustless_vanilla': 
            for i, t, z, tau in zip(range(ntheta), theta['tage'], theta['Z'], theta['tau']): 
                self.ssp.params['logzsol']  = np.log10(z/0.0190) # log(z/zsun) 
                self.ssp.params['tau']      = tau # sfh parameter 
                w, ssp_lum = self.ssp.get_spectrum(tage=t, peraa=True) 
                
                mfrac[i] = self.ssp.stellar_mass
                if i == 0: 
                    ws          = np.zeros((ntheta, len(w)))
                    ssp_lums    = np.zeros((ntheta, len(w)))
                ws[i,:] = w 
                ssp_lums[i,:] = ssp_lum

        # mass normalization
        lum_ssp = theta['mass'][:,None] * ssp_lums

        # redshift the spectra
        w_z = ws * (1. + zred)[:,None] 
        d_lum = self.cosmo.luminosity_distance(zred).to(U.cm).value # luminosity distance in cm
        flux_z = lum_ssp * UT.Lsun() / (4. * np.pi * d_lum[:,None]**2) / (1. + zred)[:,None] * 1e17 # 10^-17 ergs/s/cm^2/Ang

        if wavelength is None: 
            outwave = w_z
            outspec = flux_z
        else: 
            outwave = np.atleast_2d(wavelength)
            outspec = np.zeros((flux_z.shape[0], outwave.shape[1]))
            for i, _w, _f in zip(range(outwave.shape[0]), w_z, flux_z): 
                outspec[i,:] = np.interp(outwave[i,:], _w, _f, left=0, right=0)

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
    
        nmaggies = filters.get_ab_maggies(spec * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom, wavelength=w.flatten()*U.Angstrom) # nanomaggies 
        return np.array(list(nmaggies[0]))

    def MCMC_spec(self, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None, 
            nwalkers=100, burnin=100, niter=1000, threads=1, writeout=None, silent=True): 
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

        :param nwalkers: (optional) 
            number of walkers. (default: 100) 
        
        :param burnin: (optional) 
            int specifying the burnin. (default: 100) 
        
        :param nwalkers: (optional) 
            int specifying the number of iterations. (default: 1000) 
        
        :param threads: (optional) 
            int specifying the number of threads. Not sure if this works or not... (default: 1) 

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
        import scipy.optimize as op
        import emcee
        ndim = len(self.priors) 
    
        # check mask 
        _mask = self._check_mask(mask, wave_obs, flux_ivar_obs) 

        # posterior function args and kwargs
        lnpost_args = (wave_obs, 
                flux_obs * 1e17,        # 10^-17 ergs/s/cm^2/Ang
                flux_ivar_obs * 1e-34,   # 1/(10^-17 ergs/s/cm^2/Ang)^2
                zred) 
        lnpost_kwargs = {
                'mask': _mask,           # emission line mask 
                'prior_shape': 'flat'   # shape of prior (hardcoded) 
                }

        # run emcee and get MCMC chains 
        chain = self._emcee(self._lnPost, lnpost_args, lnpost_kwargs, 
                nwalkers=nwalkers, burnin=burnin, niter=niter, threads=threads, silent=silent)
        # get quanitles of the posterior
        lowlow, low, med, high, highhigh = np.percentile(chain, [2.5, 16, 50, 84, 97.5], axis=0)
    
        output = {} 
        output['redshift'] = zred
        output['theta_med'] = med 
        output['theta_1sig_plus'] = high
        output['theta_2sig_plus'] = highhigh
        output['theta_1sig_minus'] = low
        output['theta_2sig_minus'] = lowlow
    
        w_model, flux_model = self.model(med, zred=zred, wavelength=wave_obs)
        output['wavelength_model'] = w_model
        output['flux_model'] = flux_model 
        output['wavelength_data'] = wave_obs
        output['flux_data'] = flux_obs
        output['flux_ivar_data'] = flux_ivar_obs

        if writeout is not None: 
            fh5  = h5py.File(writeout, 'w') 
            for k in output.keys(): 
                fh5.create_dataset(k, data=output[k]) 
            fh5.create_dataset('mcmc_chain', data=chain) 
            fh5.close() 
        return output  
    
    def MCMC_photo(self, photo_obs, photo_ivar_obs, zred, bands='desi', 
            nwalkers=100, burnin=100, niter=1000, threads=1, writeout=None, silent=True): 
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
        
        :param threads: (optional) 
            int specifying the number of threads. Not sure if this works or not... (default: 1) 

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
                'prior_shape': 'flat'   # shape of prior (hardcoded) 
                }
    
        # run emcee and get MCMC chains 
        chain = self._emcee(self._lnPost_photo, lnpost_args, lnpost_kwargs, 
                nwalkers=nwalkers, burnin=burnin, niter=niter, threads=threads, silent=silent)
        # get quanitles of the posterior
        lowlow, low, med, high, highhigh = np.percentile(chain, [2.5, 16, 50, 84, 97.5], axis=0)
    
        output = {} 
        output['redshift'] = zred
        output['theta_med'] = med 
        output['theta_1sig_plus'] = high
        output['theta_2sig_plus'] = highhigh
        output['theta_1sig_minus'] = low
        output['theta_2sig_minus'] = lowlow
    
        flux_model = self.model_photo(med, zred=zred, filters=filters)
        output['flux_model'] = flux_model 
        output['flux_data'] = photo_obs
        output['flux_ivar_data'] = photo_ivar_obs

        if writeout is not None: 
            fh5  = h5py.File(writeout, 'w') 
            for k in output.keys(): 
                fh5.create_dataset(k, data=output[k]) 
            fh5.create_dataset('mcmc_chain', data=chain) 
            fh5.close() 
        return output  
    
    def _emcee(self, lnpost_fn, lnpost_args, lnpost_kwargs, nwalkers=100, burnin=100, niter=1000, threads=1, silent=True): 
        ''' Runs MCMC (using emcee) for a given log posterior function.
        '''
        import scipy.optimize as op
        import emcee
        ndim = len(self.priors) 

        # get initial theta by minimization 
        if not silent: print('getting initial theta') 
        dprior = np.array(self.priors)[:,1] - np.array(self.priors)[:,0]  

        _lnpost = lambda *args: -2. * lnpost_fn(*args, **lnpost_kwargs) 

        min_result = op.minimize(
                _lnpost, 
                np.average(np.array(self.priors), axis=1), # guess the middle of the prior 
                args=lnpost_args, 
                method='BFGS', 
                options={'eps': 0.01 * dprior, 'maxiter': 100})
        tt0 = min_result['x'] 
        if not silent: print('initial theta = [%s]' % ', '.join([str(_t) for _t in tt0])) 
    
        # initial sampler 
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_fn, 
                args=lnpost_args, kwargs=lnpost_kwargs, threads=threads)
        # initial walker positions 
        p0 = [tt0 + 1.e-4 * dprior * np.random.randn(ndim) for i in range(nwalkers)]

        # burn in 
        if not silent: print('running burn-in') 
        pos, prob, state = self.sampler.run_mcmc(p0, burnin)
        self.sampler.reset()

        # run mcmc 
        if not silent: print('running main chain') 
        self.sampler.run_mcmc(pos, niter)
        
        return  self.sampler.flatchain

    def _lnPost(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None, prior_shape='flat'): 
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

        :param prior_shape: (optional) 
            shape of the prior. (default: 'flat') 
        '''
        lp = self._lnPrior(tt_arr, shape=prior_shape) # log prior
        if not np.isfinite(lp): 
            return -np.inf
        return lp - 0.5 * self._Chi2(tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred, mask=mask)

    def _Chi2(self, tt_arr, wave_obs, flux_obs, flux_ivar_obs, zred, mask=None): 
        ''' calculated the chi-squared between the data and model spectra. 
        '''
        # model(theta) 
        _, flux = self.model(tt_arr, zred=zred, wavelength=wave_obs) 
        # data - model(theta) with masking 
        dflux = (flux[:,~mask] - flux_obs[~mask]) 
        # calculate chi-squared
        _chi2 = np.sum(dflux**2 * flux_ivar_obs[~mask]) 
        return _chi2
    
    def _lnPost_photo(self, tt_arr, flux_obs, flux_ivar_obs, zred, filters=None, bands=None, prior_shape='flat'): 
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

        :param prior_shape: (optional) 
            shape of the prior. (default: 'flat') 
        '''
        lp = self._lnPrior(tt_arr, shape=prior_shape) # log prior
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
        return _chi2

    def _lnPrior(self, tt_arr, shape='flat'): 
        ''' log prior(theta). 
        **currenlty only supports flat priors**
        '''
        if shape not in ['flat']: raise NotImplementedError

        prior_arr = np.array(self.priors) 
        prior_min, prior_max = prior_arr[:,0], prior_arr[:,1]

        dtt_min = tt_arr - prior_min 
        dtt_max = prior_max - tt_arr

        if (np.min(dtt_min) < 0.) or (np.min(dtt_max) < 0.): 
            return -np.inf 
        else:
            return 0.

    def _set_prior(self, priors): 
        ''' sets the priors to be used for the MCMC parameter inference 

        parameters 
        ----------
        priors : list
            List of tuples that specify the priors of the free parameters 
        '''
        if priors is None: # default priors
            p_mass  = (8, 13)       # mass
            p_z     = (-3, 1)       # Z 
            p_tage  = (0., 13.)     # tage
            p_d2    = (0., 10.)     # dust 2 
            p_tau   = (0.1, 10.)    # tau 

            if self.model_name == 'vanilla':            # thetas: mass, Z, tage, dust2, tau
                priors = [p_mass, p_z, p_tage, p_d2, p_tau]
            elif self.model_name == 'dustless_vanilla': # thetas: mass, Z, tage, tau
                priors = [p_mass, p_z, p_tage, p_tau]
        else: 
            if self.model_name == 'vanilla': 
                assert len(priors) == 5, 'specify priors for mass, Z, tage, dust2, and tau'
            elif self.model_name == 'dustless_vanilla': 
                assert len(priors) == 4, 'specify priors for mass, Z, tage, and tau'
        
        self.priors = priors
        return None 

    def _ssp_initiate(self): 
        ''' initialize sps (FSPS StellarPopulaiton object) 
        '''
        if self.model_name == 'vanilla': 
            ssp = fsps.StellarPopulation(
                    zcontinuous=1,          # interpolate metallicities
                    sfh=4,                  # sfh type 
                    dust_type=2,            # Calzetti et al. (2000) attenuation curve. 
                    imf_type=1)             # chabrier 

        elif self.model_name == 'dustless_vanilla': 
            ssp = fsps.StellarPopulation(
                    zcontinuous=1,          # interpolate metallicities
                    sfh=4,                  # sfh type 
                    imf_type=1)             # chabrier 
        else: 
            raise NotImplementedError
        return ssp 
    
    def _theta(self, tt_arr): 
        ''' Given some theta array return dictionary of parameter values. 
        This is synchronized with self.model_name
        '''
        theta = {} 
        tt_arr = np.atleast_2d(tt_arr) 
        if self.model_name == 'vanilla': 
            # tt_arr columns: mass, Z, tage, dust2, tau
            theta['mass']   = 10**tt_arr[:,0]
            theta['Z']      = 10**tt_arr[:,1]
            theta['tage']   = tt_arr[:,2]
            theta['dust2']  = tt_arr[:,3]
            theta['tau']    = tt_arr[:,4]
        elif self.model_name == 'dustless_vanilla': 
            # tt_arr columns: mass, Z, tage, tau
            theta['mass']   = 10**tt_arr[:,0]
            theta['Z']      = 10**tt_arr[:,1]
            theta['tage']   = tt_arr[:,2]
            theta['tau']    = tt_arr[:,3]
        else: 
            raise NotImplementedError
        return theta

    def _check_mask(self, mask, wave_obs, flux_ivar_obs): 
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

        zero_err = np.isfinite(flux_ivar_obs)
        _mask = _mask | zero_err 
        return _mask 

    def _get_bands(self, bands): 
        ''' given bands
        '''
        if isinstance(bands, str): 
            if bands == 'desi': 
                bands_list = ['decam2014-g', 'decam2014-r', 'decam2014-z', 'wise2010-W1', 'wise2010-W2', 'wise2010-W3', 'wise2010-W4']
            else: 
                raise NotImplementedError("specified bands not implemented") 
        elif isinstance(bands, list): 
            bands_list = bands
        else: 
            raise NotImplementedError("specified bands not implemented") 
        return bands_list 
