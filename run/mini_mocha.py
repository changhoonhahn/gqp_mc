'''

script for the *mini* Mock Challenge (mini mocha) 

>>> python mini_mocha.py arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9 arg10 arg11 arg12 arg13 arg14 arg15

* arg1: 'photo' or 'specphoto'
* arg2: simulation (e.g. 'lgal') 
* arg3: first galaxy number 
* arg4: last galaxy number 
* arg5: noise model 
    * 'legacy' for `arg1 == photo`
    * 'bgs0_legacy' for `arg1 == specphoto`
* arg6: number of threads
* arg7: number of emcee walkers 
* arg8: number of iterations for burnin 
* arg9: number of iterations for main chain. If `arg11 == 'adaptive'`, it will
  adaptively determine the number of iterations until convergence 
* arg10: maximum number of iterations for `arg11 == adaptive`. This is ignored
  if the number of iteration is specified 
* arg11: IF True, overwrites the mcmc file. 
* arg12: If True, post processes the MCMC chain.

Here are some examples: 

e.g. ifsps run on forward modeled photometry of lgal galaxies 0 to 90 with vanilla
  model. 20 walkers, adaptive mcmc with maximum iteration of 50000. This call will 
  overwrite the mcmc file entirely. 

>>> python mini_mocha.py photo lgal 0 90 legacy ifsps vanilla 1 20 200 adaptive 50000 overwrite False 

e.g. ispeculator run on forward modeled spectrophotometry of lgal galaxies 0 to
  90 with emulator model. 40 walkers, adaptive mcmc with maximum iteration of
  50000. This call will append to existing MCMC chain. 

>>> python mini_mocha.py specphoto lgal 0 90 bgs0_legacy ispeculator emulator 1 40 200 adaptive 50000 append False

See `run/cori/lgal_ispeculator_photo.slurm` or 
`run/cori/lgal_ispeculator_specphoto.slurm` for examples on how to run this on
NERSC cori. 


'''
import sys 
import os 
import h5py 
import numpy as np 
import corner as DFM 
from functools import partial
from multiprocessing.pool import Pool 
from scipy.stats import gaussian_kde as gkde
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
# --- provabgs --- 
from provabgs import infer as Infer
from provabgs import flux_calib as FluxCalib
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
try: 
    if os.environ['NERSC_HOST'] == 'cori':  pass
except KeyError: 
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


lbl_dict = {
        'logmstar': r'$\log M_*$', 
        'z_metal': r'$\log Z$',
        'dust1': 'dust1', 
        'dust2': 'dust2', 
        'dust_index': 'dust index', 
        'tau': r'$\tau$', 
        'f_fiber': r'$f_{\rm fiber}$',
        'beta1_sfh': r'$\beta_1^{\rm SFH}$', 
        'beta2_sfh': r'$\beta_2^{\rm SFH}$', 
        'beta3_sfh': r'$\beta_3^{\rm SFH}$', 
        'beta4_sfh': r'$\beta_4^{\rm SFH}$', 
        'gamma1_zh': r'$\gamma_1^{\rm ZH}$', 
        'gamma2_zh': r'$\gamma_2^{\rm ZH}$',
        'logsfr.100myr': r'$\log {\rm SFR}_{\rm 100 Myr}$',
        'logsfr.1gyr': r'$\log {\rm SFR}_{\rm 1 Gyr}$',
        'logz.mw': r'$\log Z_{\rm MW}$',
        }


def construct_sample(sim): 
    ''' construct the mini Mock Challenge photometry, spectroscopy 
    '''
    # construct photometry and spectroscopy 
    dir_run = os.path.dirname(os.path.realpath(__file__)) 
    if sim == 'lgal': 
        fscript = os.path.join(dir_run, 'fm_lgal.py')
    elif sim == 'tng': 
        fscript = os.path.join(dir_run, 'fm_tng.py')
    os.system('python %s' % fscript)
    return None 


def validate_sample(sim): 
    ''' generate some plots to validate the mini Mock Challenge photometry
    and spectroscopy 
    '''
    # read photometry 
    photo, _ = Data.Photometry(sim=sim, noise='legacy', sample='mini_mocha') 
    photo_g = 22.5 - 2.5 * np.log10(photo['flux'][:,0])
    photo_r = 22.5 - 2.5 * np.log10(photo['flux'][:,1])
    photo_z = 22.5 - 2.5 * np.log10(photo['flux'][:,2])
        
    flux_g = photo['flux'][:,0] * 1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs())
    flux_r = photo['flux'][:,1] * 1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs())
    flux_z = photo['flux'][:,2] * 1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()) # convert to 10^-17 ergs/s/cm^2/Ang
    ivar_g = photo['ivar'][:,0] * (1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_r = photo['ivar'][:,1] * (1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_z = photo['ivar'][:,2] * (1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()))**-2. # convert to 10^-17 ergs/s/cm^2/Ang
    
    # read BGS targets from imaging 
    bgs_targets = h5py.File(os.path.join(UT.dat_dir(), 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    n_targets = len(bgs_targets['ra'][...]) 
    
    bgs_g = 22.5 - 2.5 * np.log10(bgs_targets['flux_g'][...])[::10]
    bgs_r = 22.5 - 2.5 * np.log10(bgs_targets['flux_r'][...])[::10]
    bgs_z = 22.5 - 2.5 * np.log10(bgs_targets['flux_z'][...])[::10]
    
    # photometry validation 
    fig = plt.figure(figsize=(6,6)) 
    sub = fig.add_subplot(111)
    DFM.hist2d(bgs_g - bgs_r, bgs_r - bgs_z, color='k', levels=[0.68, 0.95], 
            range=[[-1., 3.], [-1., 3.]], bins=40, smooth=0.5, 
            plot_datapoints=True, fill_contours=False, plot_density=False, linewidth=0.5, 
            ax=sub) 
    sub.scatter([-100.], [-100], c='k', s=1, label='BGS targets')
    sub.scatter(photo_g - photo_r, photo_r - photo_z, c='C0', s=1, label='LGal photometry') 
    sub.legend(loc='upper left', handletextpad=0, markerscale=10, fontsize=20) 
    sub.set_xlabel('$g-r$', fontsize=25) 
    sub.set_xlim(-1., 3.) 
    sub.set_ylabel('$r-z$', fontsize=25) 
    sub.set_ylim(-1., 3.) 
    ffig = os.path.join(UT.dat_dir(), 
            'mini_mocha', 'mini_mocha.%s.photo.png' % sim)
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 

    # read spectra
    spec_s, _ = Data.Spectra(sim=sim, noise='none', sample='mini_mocha') 
    spec_bgs0, _ = Data.Spectra(sim=sim, noise='bgs0', sample='mini_mocha') 
    spec_bgs1, _ = Data.Spectra(sim=sim, noise='bgs1', sample='mini_mocha') 
    spec_bgs2, _ = Data.Spectra(sim=sim, noise='bgs2', sample='mini_mocha') 
        
    # read sky brightness
    _fsky = os.path.join(UT.dat_dir(), 'mini_mocha', 'bgs.exposure.surveysim.150s.v0p4.sample.hdf5') 
    fsky = h5py.File(_fsky, 'r') 
    wave_sky    = fsky['wave'][...] # sky wavelength 
    sbright_sky = fsky['sky'][...]
    
    # spectra validationg 
    fig = plt.figure(figsize=(10,5))     
    sub = fig.add_subplot(111)
    
    for i, spec_bgs in enumerate([spec_bgs2, spec_bgs0]): 
        wsort = np.argsort(spec_bgs['wave']) 
        if i == 0: 
            _plt, = sub.plot(spec_bgs['wave'][wsort], spec_bgs['flux'][0,wsort], c='C%i' % i, lw=0.25) 
        else:
            sub.plot(spec_bgs['wave'][wsort], spec_bgs['flux'][0,wsort], c='C%i' % i, lw=0.25) 
    
    _plt_photo = sub.errorbar([4750, 6350, 9250], [flux_g[0], flux_r[0], flux_z[0]], 
            [ivar_g[0]**-0.5, ivar_r[0]**-0.5, ivar_z[0]**-0.5], fmt='.r') 

    _plt_sim, = sub.plot(spec_s['wave'][0,:], spec_s['flux'][0,:], c='k', ls='-', lw=1) 
    _plt_sim0, = sub.plot(spec_s['wave'][0,:], spec_s['flux_unscaled'][0,:], c='k', ls=':', lw=0.25) 

    leg = sub.legend(
            [_plt_sim0, _plt_photo, _plt_sim, _plt], 
            ['%s spectrum' % sim.upper(), '%s photometry' % sim.upper(), 
                '%s fiber spectrum' % sim.upper(), '%s BGS spectra' % sim.upper()],
            loc='upper right', fontsize=17) 
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    sub.set_xlabel('Wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3e3, 1e4) 
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$', fontsize=20) 
    if sim == 'lgal': sub.set_ylim(-2., 8.) 
    elif sim == 'tng': sub.set_ylim(0., None) 
    
    ffig = os.path.join(UT.dat_dir(), 
            'mini_mocha', 'mini_mocha.%s.spectra.png' % sim)
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    
    return None 


def postprocess_mcmc(desi_mcmc, fmcmc, dt_sfr=1., n_sample=50000): 
    '''  calculate derived properties log M*, log SFR, log Z_MW and prior
    correction weights to impose uniform priors on [log M*, log SFR, log Z_MW]
    '''
    mcmc = desi_mcmc.read_chain(fmcmc)
    flat_chain = desi_mcmc._flatten_chain(mcmc['mcmc_chain'][-1000:,:,:])
    z = mcmc['redshift'] 

    logmstar = flat_chain[:,0] 
    
    # calculate average SFRs 
    avg_sfr_100myr  = desi_mcmc.model.avgSFR(flat_chain, z, dt=0.1)
    avg_sfr_1gyr    = desi_mcmc.model.avgSFR(flat_chain, z, dt=1.)

    avg_logsfr_100myr   = np.log10(avg_sfr_100myr)        
    avg_logsfr_1gyr     = np.log10(avg_sfr_1gyr)        
    
    # calculate mass-weighted metallicity
    z_mw    = desi_mcmc.model.Z_MW(flat_chain, z)
    logz_mw = np.log10(z_mw) 
    
    output = {} 
    output['logmstar']      = logmstar
    output['logsfr.100myr'] = avg_logsfr_100myr
    output['logsfr.1gyr']   = avg_logsfr_1gyr
    output['logz.mw']       = logz_mw 
    
    # ------------------------------------------------------------
    # get prior correction 
    # 1. sample prior 
    theta_prior = np.array([desi_mcmc.prior.sample() for i in range(n_sample)]) 
    
    # 2. compute the derived properties we want to impose flat priors on  
    logm_prior          = theta_prior[:,0] 
    logsfr_1gyr_prior   = np.log10(desi_mcmc.model.avgSFR(theta_prior, z, dt=1.))
    logsfr_100myr_prior = np.log10(desi_mcmc.model.avgSFR(theta_prior, z, dt=0.1))
    logzmw_prior    = np.log10(desi_mcmc.model.Z_MW(theta_prior, z))
    
    # 3. fit a joint distirbution of the derived properties 
    kde_fit_1gyr    = gkde(np.array([logm_prior, logsfr_1gyr_prior, logzmw_prior])) 
    kde_fit_100myr  = gkde(np.array([logm_prior, logsfr_100myr_prior, logzmw_prior])) 

    # 4. calculate the max entropy weights
    p_prop_1gyr = kde_fit_1gyr.pdf(np.array([logmstar, avg_logsfr_1gyr, logz_mw]))
    p_prop_100myr = kde_fit_100myr.pdf(np.array([logmstar, avg_logsfr_100myr, logz_mw]))

    w_prior_1gyr = 1./p_prop_1gyr
    w_prior_100myr = 1./p_prop_100myr

    w_prior_1gyr[p_prop_1gyr < 1e-4] = 0. 
    w_prior_100myr[p_prop_100myr < 1e-4] = 0. 

    output['w_prior.100myr'] = w_prior_100myr
    output['w_prior.1gyr'] = w_prior_1gyr
    # ------------------------------------------------------------

    for k in ['redshift', 
            'wavelength_obs', 'flux_spec_obs', 'flux_ivar_spec_obs', 
            'flux_photo_obs', 'flux_ivar_photo_obs',
            'flux_spec_model', 'flux_photo_model']:
        if k in mcmc.keys(): 
            output[k] = mcmc[k] 

    fpost = fmcmc.replace('.mcmc.hdf5', '.postproc.hdf5')
    fh5  = h5py.File(fpost, 'w') 
    for k in output.keys(): 
        if output[k] is None: 
            print(k) 
            continue 
        fh5.create_dataset(k, data=output[k]) 
    fh5.close() 
    return output  


def fit_photometry(igal, sim='lgal', noise='legacy', nwalkers=30, burnin=1000,
        niter=1000, maxiter=200000, opt_maxiter=100, overwrite=False,
        postprocess=False):    
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without
    noise or with BGS-like noise. Produces an MCMC chain and, if not on nersc,
    a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 
    :param noise: 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 
    '''
    # read Lgal photometry of the mini_mocha mocks 
    photo, meta = Data.Photometry(sim=sim, noise=noise, lib='bc03', sample='mini_mocha') 
    assert meta['redshift'][igal] < 0.5, "speculator does not support z > 0.5 yet" 

    photo_obs   = photo['flux'][igal,:3]
    ivar_obs    = photo['ivar'][igal,:3]

    print('--- input ---') 
    print('  z = %f' % meta['redshift'][igal])
    print('  log M* total = %f' % meta['logM_total'][igal])
    print('  log SFR 100Myr = %f' % np.log10(meta['sfr_100myr'][igal]))
    print('  log SFR 1Gyr = %f' % np.log10(meta['sfr_1gyr'][igal]))
    print('  log Z_MW = %f' % np.log10(meta['Z_MW'][igal]))
    # ------------------------------------------------------------------------------------
    theta_names = ['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh', 'beta4_sfh',
            'gamma1_zh', 'gamma2_zh', 'dust1', 'dust2', 'dust_index'] 
    truths = [None for _ in theta_names] 
    truths[0] = meta['logM_total'][igal]

    # set up prior object
    priors = Infer.load_priors([
            Infer.UniformPrior(8, 12, label='sed'),     # uniform priors on logM*
            Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
            Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
            Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
            Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
            Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
            Infer.UniformPrior(-2.2, 0.4, label='sed')      # uniform priors on dust_index 
            ])

    desi_mcmc = Infer.desiMCMC(
            prior=priors, 
            flux_calib=FluxCalib.no_flux_factor # no flux calibration necessary
            )
    f_mcmc = os.path.join(UT.dat_dir(), 'mini_mocha', 'provabgs', 
            '%s.photo.noise_%s.%i.mcmc.hdf5' % (sim, noise, igal))

    if not overwrite and os.path.isfile(f_mcmc): 
        print('--- read mcmc ---') 
        mcmc = desi_mcmc.read_chain(f_mcmc, debug=True)
    else: 
        print('--- run mcmc ---') 
        mcmc = desi_mcmc.run(
                photo_obs=photo_obs, 
                photo_ivar_obs=ivar_obs, 
                zred=meta['redshift'][igal], 
                bands='desi', 
                sampler='zeus',
                nwalkers=nwalkers, 
                burnin=burnin,
                opt_maxiter=opt_maxiter,
                niter=niter, 
                writeout=f_mcmc, 
                overwrite=overwrite, 
                debug=True)

    if postprocess:
        print('--- postprocessing ---') 
        post = postprocess_mcmc(desi_mcmc, f_mcmc)

    print('---------------') 
    labels = [lbl_dict[_t] for _t in theta_names]

    # corner plot of the posteriors 
    flat_chain = desi_mcmc._flatten_chain(mcmc['mcmc_chain'])
    fig = DFM.corner(
        flat_chain, 
        range=np.array(mcmc['prior_range']).T, 
        quantiles=[0.16, 0.5, 0.84], 
        levels=[0.68, 0.95], 
        nbin=40,
        smooth=True, 
        truths=truths, 
        labels=labels, 
        label_kwargs={'fontsize': 20}) 
    fig.savefig(f_mcmc.replace('.hdf5', '.png'), bbox_inches='tight') 
    plt.close() 

    # corner plot of the posteriors for derive prop
    fig = DFM.corner(
            np.vstack([post['logmstar'], post['logsfr.1gyr'], post['logz.mw']]).T,
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95], 
            nbin=40,
            smooth=True,
            hist_kwargs={'density': True}) 
    _ = DFM.corner(
            np.vstack([post['logmstar'], post['logsfr.1gyr'], post['logz.mw']]).T,
            weights=post['w_prior.1gyr'],
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95], 
            nbin=40,
            smooth=True, 
            color='C1', 
            truths=[meta['logM_total'][igal], np.log10(meta['sfr_1gyr'][igal]),
                np.log10(meta['Z_MW'][igal])], 
            labels=[lbl_dict['logmstar'], lbl_dict['logsfr.1gyr'], lbl_dict['logz.mw']], 
            label_kwargs={'fontsize': 20}, 
            fig=fig, 
            hist_kwargs={'density': True}) 
    fig.savefig(f_mcmc.replace('.hdf5', '.prop.png'), bbox_inches='tight') 
    plt.close() 

    # best-fit observable 
    fig = plt.figure(figsize=(5,3))
    sub = fig.add_subplot(111)
    sub.errorbar(np.arange(len(photo_obs)), photo_obs,
            yerr=ivar_obs**-0.5, fmt='.k', label='data')
    sub.scatter(np.arange(len(photo_obs)), mcmc['flux_photo_model'], c='C1',
            label='best-fit') 
    sub.legend(loc='upper left', markerscale=3, handletextpad=0.2, fontsize=15) 
    sub.set_xticks([0, 1, 2]) 
    sub.set_xticklabels(['$g$', '$r$', '$z$']) 
    sub.set_xlim(-0.5, len(photo_obs)-0.5)
    fig.savefig(f_mcmc.replace('.hdf5', '.bestfit.png'), bbox_inches='tight') 
    plt.close()
    return None 


def fit_spectrum(igal, sim='lgal', noise='bgs', nwalkers=30, burnin=1000,
        niter=1000, maxiter=200000, opt_maxiter=100, overwrite=False,
        postprocess=False):    
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without
    noise or with BGS-like noise. Produces an MCMC chain and, if not on nersc,
    a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 
    :param noise: 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 
    '''
    # read Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim=sim, noise=noise, lib='bc03', sample='mini_mocha') 

    assert meta['redshift'][igal] < 0.5, "speculator does not support z > 0.5 yet" 

    wave_obs        = specs['wave']
    print(wave_obs.shape)
    flux_obs        = specs['flux'][igal]
    if noise != 'none': 
        ivar_obs    = specs['ivar'][igal]
        resolution  = [specs['res_b'][igal], specs['res_r'][igal], specs['res_z'][igal]]
    else: 
        ivar_obs    = np.ones(len(specs['wave']))
        resolution  = None 

    print('--- input ---') 
    print('  z = %f' % meta['redshift'][igal])
    print('  log M* fiber = %f' % meta['logM_fiber'][igal])
    print('  log SFR 100Myr = %f' % np.log10(meta['sfr_100myr'][igal]))
    print('  log SFR 1Gyr = %f' % np.log10(meta['sfr_1gyr'][igal]))
    print('  log Z_MW = %f' % np.log10(meta['Z_MW'][igal]))
    # ------------------------------------------------------------------------------------
    theta_names = ['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh', 'beta4_sfh',
            'gamma1_zh', 'gamma2_zh', 'dust1', 'dust2', 'dust_index'] 
    truths = [None for _ in theta_names] 
    truths[0] = meta['logM_fiber'][igal]

    # set up prior object
    priors = Infer.load_priors([
            Infer.UniformPrior(8, 12, label='sed'),     # uniform priors on logM*
            Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
            Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
            Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
            Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
            Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
            Infer.UniformPrior(-2.2, 0.4, label='sed')      # uniform priors on dust_index 
            ])

    desi_mcmc = Infer.desiMCMC(
            prior=priors, 
            flux_calib=FluxCalib.no_flux_factor # no flux calibration necessary
            )
    f_mcmc = os.path.join(UT.dat_dir(), 'mini_mocha', 'provabgs', 
            '%s.spec.noise_%s.%i.mcmc.hdf5' % (sim, noise, igal))

    if not overwrite and os.path.isfile(f_mcmc): 
        print('--- read mcmc ---') 
        mcmc = desi_mcmc.read_chain(f_mcmc, debug=True)
    else: 
        print('--- run mcmc ---') 
        mcmc = desi_mcmc.run(
                wave_obs=wave_obs, 
                flux_obs=flux_obs, 
                flux_ivar_obs=ivar_obs, 
                resolution=resolution, 
                zred=meta['redshift'][igal], 
                vdisp=0., # km/s velocity dispersion 
                mask='emline', 
                sampler='zeus',
                nwalkers=nwalkers, 
                burnin=burnin,
                opt_maxiter=opt_maxiter,
                niter=niter, 
                writeout=f_mcmc, 
                overwrite=overwrite, 
                debug=True)

    if postprocess:
        print('--- postprocessing ---') 
        post = postprocess_mcmc(desi_mcmc, f_mcmc)

    print('---------------') 
    labels = [lbl_dict[_t] for _t in theta_names]

    # corner plot of the posteriors 
    flat_chain = desi_mcmc._flatten_chain(mcmc['mcmc_chain'])
    fig = DFM.corner(
            flat_chain[:-3], 
        range=np.array(mcmc['prior_range']).T, 
        quantiles=[0.16, 0.5, 0.84], 
        levels=[0.68, 0.95], 
        nbin=40,
        smooth=True, 
        truths=truths, 
        labels=labels, 
        label_kwargs={'fontsize': 20}) 
    fig.savefig(f_mcmc.replace('.hdf5', '.png'), bbox_inches='tight') 
    plt.close() 
    
    # corner plot of the posteriors for derive prop
    fig = DFM.corner(
            np.vstack([post['logmstar'], post['logsfr.1gyr'], post['logz.mw']]).T,
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95], 
            nbin=40,
            smooth=True,
            hist_kwargs={'density': True}) 
    _ = DFM.corner(
            np.vstack([post['logmstar'], post['logsfr.1gyr'], post['logz.mw']]).T,
            weights=post['w_prior.1gyr'],
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95], 
            nbin=40,
            smooth=True, 
            color='C1', 
            truths=[meta['logM_fiber'][igal], np.log10(meta['sfr_1gyr'][igal]),
                np.log10(meta['Z_MW'][igal])], 
            labels=[lbl_dict['logmstar'], lbl_dict['logsfr.1gyr'], lbl_dict['logz.mw']], 
            label_kwargs={'fontsize': 20}, 
            fig=fig, 
            hist_kwargs={'density': True}) 
    fig.savefig(f_mcmc.replace('.hdf5', '.prop.png'), bbox_inches='tight') 
    plt.close() 

    # best-fit observable 
    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(111) 
    sub.plot(mcmc['wavelength_obs'], mcmc['flux_spec_obs'], 
        c='C0', lw=1, label='obs.') 
    sub.plot(mcmc['wavelength_obs'], mcmc['flux_spec_model'], 
        c='k', ls='--', lw=1, label='best-fit') 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3600., 9800.)
    sub.set_ylim(-1., 5.) 
    fig.savefig(f_mcmc.replace('.hdf5', '.bestfit.png'), bbox_inches='tight') 
    plt.close()
    return None 


def fit_spectrophotometry(igal, sim='lgal', noise='bgs0_legacy', nwalkers=30,
        burnin=1000, niter=1000, maxiter=200000, opt_maxiter=100,
        overwrite=False, postprocess=False):    
    ''' Fit Lgal spectra. `noise` specifies whether to fit spectra without
    noise or with BGS-like noise. Produces an MCMC chain and, if not on nersc,
    a corner plot of the posterior. 

    :param igal: 
        index of Lgal galaxy within the spectral_challenge 
    :param noise: 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 
    '''
    noise_spec = noise.split('_')[0]
    noise_photo = noise.split('_')[1]
    # read noiseless Lgal spectra of the spectral_challenge mocks 
    specs, meta = Data.Spectra(sim=sim, noise=noise_spec, lib='bc03', sample='mini_mocha') 
    # read Lgal photometry of the mini_mocha mocks 
    photo, _ = Data.Photometry(sim=sim, noise=noise_photo, lib='bc03', sample='mini_mocha') 
    
    assert meta['redshift'][igal] < 0.5, "speculator does not support z > 0.5 yet" 

    wave_obs        = specs['wave']
    flux_obs        = specs['flux'][igal]
    ivar_obs        = specs['ivar'][igal]
    resolution      = [specs['res_b'][igal], specs['res_r'][igal], specs['res_z'][igal]]
    photo_obs       = photo['flux'][igal,:3]
    photo_ivar_obs  = photo['ivar'][igal,:3]
    
    # get fiber flux factor prior range based on "measured" fiber flux 
    f_fiber_true = (photo['fiberflux_r_true'][igal]/photo['flux_r_true'][igal]) 
    prior_width = np.max([0.05, 5.*photo['fiberflux_r_ivar'][igal]**-0.5/photo['flux'][igal,1]])
    f_fiber_min = (photo['fiberflux_r_meas'][igal])/photo['flux'][igal,1] - prior_width
    f_fiber_max = (photo['fiberflux_r_meas'][igal])/photo['flux'][igal,1] + prior_width

    print('--- input ---') 
    print('  z = %f' % meta['redshift'][igal])
    print('  log M* total = %f' % meta['logM_total'][igal])
    print('  log M* fiber = %f' % meta['logM_fiber'][igal])
    print('  log SFR 100Myr = %f' % np.log10(meta['sfr_100myr'][igal]))
    print('  log SFR 1Gyr = %f' % np.log10(meta['sfr_1gyr'][igal]))
    print('  log Z_MW = %f' % np.log10(meta['Z_MW'][igal]))
    print('  f_fiber = %f' % f_fiber_true) 
    print('  f_fiber range = [%f, %f]' % (f_fiber_min, f_fiber_max))
    # ------------------------------------------------------------------------------------
    theta_names = ['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh', 'beta4_sfh',
            'gamma1_zh', 'gamma2_zh', 'dust1', 'dust2', 'dust_index', 'f_fiber'] 
    truths = [None for _ in theta_names] 
    truths[0] = meta['logM_total'][igal]
    truths[-1] = f_fiber_true 

    # set up prior object
    priors = Infer.load_priors([
            Infer.UniformPrior(8, 12, label='sed'),     # uniform priors on logM*
            Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
            Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
            Infer.UniformPrior(6.9e-5, 7.3e-3, label='sed'),# uniform priors on ZH coeff
            Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
            Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
            Infer.UniformPrior(-2.2, 0.4, label='sed'),     # uniform priors on dust_index 
            Infer.UniformPrior(f_fiber_min, f_fiber_max, label='flux_calib') 
            ])

    desi_mcmc = Infer.desiMCMC(
            prior=priors, 
            flux_calib=FluxCalib.constant_flux_factor # no flux calibration necessary
            )
    f_mcmc = os.path.join(UT.dat_dir(), 'mini_mocha', 'provabgs', 
            '%s.specphoto.noise_%s.%i.mcmc.hdf5' % (sim, noise, igal))

    if not overwrite and os.path.isfile(f_mcmc): 
        print('--- read mcmc ---') 
        mcmc = desi_mcmc.read_chain(f_mcmc, debug=True)
    else: 
        print('--- run mcmc ---') 
        mcmc = desi_mcmc.run(
                wave_obs=wave_obs, 
                flux_obs=flux_obs, 
                flux_ivar_obs=ivar_obs, 
                resolution=resolution, 
                photo_obs=photo_obs, 
                photo_ivar_obs=photo_ivar_obs, 
                zred=meta['redshift'][igal], 
                vdisp=0., 
                mask='emline', 
                bands='desi', 
                sampler='zeus',
                nwalkers=nwalkers, 
                burnin=burnin,
                opt_maxiter=opt_maxiter,
                niter=niter, 
                writeout=f_mcmc, 
                overwrite=overwrite, 
                debug=True)

    if postprocess:
        print('--- postprocessing ---') 
        post = postprocess_mcmc(desi_mcmc, f_mcmc)

    print('---------------') 

    # corner plot of the posteriors 
    flat_chain = desi_mcmc._flatten_chain(mcmc['mcmc_chain'])
    fig = DFM.corner(
        flat_chain, 
        range=np.array(mcmc['prior_range']).T, 
        quantiles=[0.16, 0.5, 0.84], 
        levels=[0.68, 0.95], 
        nbin=40,
        smooth=True, 
        truths=truths, 
        labels=[lbl_dict[_t] for _t in theta_names],
        label_kwargs={'fontsize': 20}) 
    fig.savefig(f_mcmc.replace('.hdf5', '.png'), bbox_inches='tight') 
    plt.close() 
    
    # corner plot of the posteriors for derive prop
    fig = DFM.corner(
            np.vstack([post['logmstar'], post['logsfr.1gyr'], post['logz.mw']]).T,
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95], 
            nbin=40,
            smooth=True,
            hist_kwargs={'density': True}) 
    _ = DFM.corner(
            np.vstack([post['logmstar'], post['logsfr.1gyr'], post['logz.mw']]).T,
            weights=post['w_prior.1gyr'],
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95], 
            nbin=40,
            smooth=True, 
            color='C1', 
            truths=[meta['logM_total'][igal], np.log10(meta['sfr_1gyr'][igal]),
                np.log10(meta['Z_MW'][igal])], 
            labels=[lbl_dict['logmstar'], lbl_dict['logsfr.1gyr'], lbl_dict['logz.mw']], 
            label_kwargs={'fontsize': 20}, 
            fig=fig, 
            hist_kwargs={'density': True}) 
    fig.savefig(f_mcmc.replace('.hdf5', '.prop.png'), bbox_inches='tight') 
    plt.close() 

    # best-fit observable 
    fig = plt.figure(figsize=(18,5))
    gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,3], wspace=0.2) 
    sub = plt.subplot(gs[0]) 
    sub.errorbar(np.arange(len(photo_obs)), photo_obs,
            yerr=photo_ivar_obs**-0.5, fmt='.k', label='data')
    sub.scatter(np.arange(len(photo_obs)), mcmc['flux_photo_model'], c='C1',
            label='model') 
    sub.legend(loc='upper left', markerscale=3, handletextpad=0.2, fontsize=15) 
    sub.set_xticks([0, 1, 2]) 
    sub.set_xticklabels(['$g$', '$r$', '$z$'], fontsize=25) 
    sub.set_xlim(-0.5, len(photo_obs)-0.5)

    sub = plt.subplot(gs[1]) 
    sub.plot(mcmc['wavelength_obs'], mcmc['flux_spec_obs'], 
        c='C0', lw=1, label='obs.') 
    sub.plot(mcmc['wavelength_obs'], mcmc['flux_spec_model'], 
        c='k', ls='--', lw=1, label='best-fit') 
    sub.legend(loc='upper right', fontsize=15) 
    sub.set_xlabel('wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3600., 9800.)
    sub.set_ylim(-1., 5.) 
    fig.savefig(f_mcmc.replace('.hdf5', '.bestfit.png'), bbox_inches='tight') 
    plt.close()
    return None 


def MP_sed_fit(spec_or_photo, igals, sim='lgal', noise='none', nthreads=1,
        nwalkers=100, burnin=100, niter=1000, maxiter=200000, overwrite=False,
        postprocess=False): 
    ''' multiprocessing wrapepr for fit_spectra and fit_photometry. This does *not* parallelize 
    the MCMC sampling of individual fits but rather runs multiple fits simultaneously. 
    
    :param spec_or_photo: 
        fit spectra or photometry 

    :param igals: 
        array/list of spectral_challenge galaxy indices

    :param noise: 
        If 'none', fit noiseless spectra. 
        If 'bgs1'...'bgs8', fit BGS-like spectra. (default: 'none') 

    :param dust: 
        If True, fit the spectra w/ dust using a model with dust 
        If False, fit the spectra w/o dust using a model without dust. 
        (default: False) 

    :param nthreads: 
        Number of threads. If nthreads == 1, just runs fit_spectra
    '''
    args = igals # galaxy indices 

    kwargs = {
            'sim': sim, 
            'noise': noise, 
            'nwalkers': nwalkers,
            'burnin': burnin,
            'niter': niter, 
            'maxiter': maxiter, 
            'opt_maxiter': 10000, 
            'overwrite': overwrite, 
            'postprocess': postprocess
            }
    if spec_or_photo == 'spec': 
        fit_func = fit_spectrum
    elif spec_or_photo == 'photo': 
        fit_func = fit_photometry
    elif spec_or_photo == 'specphoto': 
        fit_func = fit_spectrophotometry

    if nthreads > 1: 
        pool = Pool(processes=nthreads) 
        pool.map(partial(fit_func, **kwargs), args)
        pool.close()
        pool.terminate()
        pool.join()
    else: 
        # single thread, loop over 
        for igal in args: fit_func(igal, **kwargs)
    return None 


if __name__=="__main__": 
    # >>> python mini_mocha.py
    spec_or_photo   = sys.argv[1]
    sim             = sys.argv[2] 

    if spec_or_photo == 'construct':  
        construct_sample(sim)
        validate_sample(sim)
        sys.exit() 

    igal0           = int(sys.argv[3]) 
    igal1           = int(sys.argv[4]) 
    noise           = sys.argv[5]
    nthreads        = int(sys.argv[6]) 
    nwalkers        = int(sys.argv[7]) 
    burnin          = int(sys.argv[8]) 
    niter           = sys.argv[9] 
    maxiter         = int(sys.argv[10]) 
    overwrite       = sys.argv[11] == 'True'
    postprocess     = sys.argv[12] == 'True'

    try: 
        niter = int(niter)
    except ValueError: 
        pass

    print('----------------------------------------') 
    print('fitting %s of mini_mocha galaxies %i to %i' % 
            (spec_or_photo, igal0, igal1))
    print('using %i threads' % nthreads) 
    igals = range(igal0, igal1+1)

    MP_sed_fit(spec_or_photo, 
            igals,
            sim=sim,
            noise=noise, 
            nthreads=nthreads,
            nwalkers=nwalkers, 
            burnin=burnin,
            niter=niter,
            maxiter=maxiter,
            overwrite=overwrite, 
            postprocess=postprocess)
