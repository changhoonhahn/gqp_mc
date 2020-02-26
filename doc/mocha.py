'''

generate plots for the mock challenge paper 


'''
import os 
import h5py 
import numpy as np 
import corner as DFM 
from scipy.stats import norm as Norm
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
from gqp_mc import fitters as Fitters
# --- astro ---
from astropy.io import fits
from astroML.datasets import fetch_sdss_specgals
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
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


dir_fig = os.path.join(UT.dat_dir(), 'mini_mocha') 
dir_doc = '/Users/ChangHoon/projects/gqp_mc/doc/paper/figs/'
dir_fbgs = '/Users/ChangHoon/data/feasiBGS/'


def BGS(): 
    ''' plot highlighting BGS footprint and redshift number density
    '''
    # read BGS MXXL galaxies 
    mxxl    = h5py.File(os.path.join(dir_fbgs, 'BGS_r20.0.hdf5'), 'r') 
    bgs     = mxxl['Data']
    ra_bgs  = bgs['ra'][...][::10]
    dec_bgs = bgs['dec'][...][::10]
    z_bgs   = np.array(bgs['z_obs'])
    
    # read SDSS galaxies from astroML (https://www.astroml.org/modules/generated/astroML.datasets.fetch_sdss_specgals.html)
    sdss        = fetch_sdss_specgals()
    ra_sdss     = sdss['ra'] 
    dec_sdss    = sdss['dec'] 
    
    # read GAMA objects
    f_gama = os.path.join(dir_fbgs, 'gama', 'dr3', 'SpecObj.fits') 
    gama = fits.open(f_gama)[1].data 

    fig = plt.figure(figsize=(15,5))
    
    gs1 = mpl.gridspec.GridSpec(1,1, figure=fig) 
    gs1.update(left=0.02, right=0.7)
    sub = plt.subplot(gs1[0], projection='mollweide')
    sub.grid(True, linewidth=0.1) 
    # DESI footprint 
    sub.scatter((ra_bgs - 180.) * np.pi/180., dec_bgs * np.pi/180., s=1, lw=0, c='k')
    # SDSS footprint 
    sub.scatter((ra_sdss - 180.) * np.pi/180., dec_sdss * np.pi/180., s=1, lw=0, c='C0')#, alpha=0.01)
    # GAMA footprint for comparison 
    gama_ra_min = (np.array([30.2, 129., 174., 211.5, 339.]) - 180.) * np.pi/180.
    gama_ra_max = (np.array([38.8, 141., 186., 223.5, 351.]) - 180.) * np.pi/180. 
    gama_dec_min = np.array([-10.25, -2., -3., -2., -35.]) * np.pi/180.
    gama_dec_max = np.array([-3.72, 3., 2., 3., -30.]) * np.pi/180.
    for i_f, field in enumerate(['g02', 'g09', 'g12', 'g15', 'g23']): 
        rect = patches.Rectangle((gama_ra_min[i_f], gama_dec_min[i_f]), 
                gama_ra_max[i_f] - gama_ra_min[i_f], 
                gama_dec_max[i_f] - gama_dec_min[i_f], 
                facecolor='C1')
        sub.add_patch(rect)
    sub.set_xlabel('RA', fontsize=20, labelpad=10) 
    sub.set_xticklabels(['', '', '$90^o$', '', '', '$180^o$', '', '', '$270^o$'])#, fontsize=10)
    sub.set_ylabel('Dec', fontsize=20)

    gs2 = mpl.gridspec.GridSpec(1,1, figure=fig) 
    gs2.update(left=0.70, right=0.98)#, wspace=0.05)
    sub = plt.subplot(gs2[0])
    sub.hist(z_bgs, range=[0.0, 1.], color='k', bins=100) 
    sub.hist(sdss['z'], range=[0.0, 1.], color='C0', bins=100) 
    sub.hist(np.array(gama['Z']), range=[0.0, 1.], color='C1', bins=100) 
    sub.set_xlabel('Redshift', fontsize=20) 
    sub.set_xlim([0., 0.6])
    sub.set_ylabel('dN/dz', fontsize=20) 

    def _fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        a = a.split('.')[0]
        b = int(b)
        if b == 0: 
            return r'${}$'.format(a)
        else: 
            return r'${}\times10^{{{}}}$'.format(a, b)

    sub.get_yaxis().set_major_formatter(ticker.FuncFormatter(_fmt))
    plts = []  
    for clr in ['k', 'C0', 'C1']: 
        _plt = sub.fill_between([0], [0], [0], color=clr, linewidth=0)
        plts.append(_plt) 
    sub.legend(plts, ['DESI', 'SDSS', 'GAMA'], loc='upper right', handletextpad=0.3, prop={'size': 20}) 
    sub.set_xticklabels([]) 

    ffig = os.path.join(dir_fig, 'bgs.png')
    fig.savefig(ffig, bbox_inches='tight')
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def FM_photo():
    ''' plot illustrating the forward model for photometry 
    '''
    from speclite import filters as specFilter

    # read forward modeled Lgal photometry
    photo, meta = Data.Photometry(sim='lgal', noise='legacy', lib='bc03', sample='mini_mocha')
    flux_g = photo['flux'][:,0] * 1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs())
    flux_r = photo['flux'][:,1] * 1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs())
    flux_z = photo['flux'][:,2] * 1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()) # convert to 10^-17 ergs/s/cm^2/Ang
    ivar_g = photo['ivar'][:,0] * (1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_r = photo['ivar'][:,1] * (1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_z = photo['ivar'][:,2] * (1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()))**-2. # convert to 10^-17 ergs/s/cm^2/Ang

    # read noiseless Lgal spectroscopy 
    specs, _ = Data.Spectra(sim='lgal', noise='none', lib='bc03', sample='mini_mocha') 
    # read in photometric bandpass filters 
    filter_response = specFilter.load_filters(
            'decam2014-g', 'decam2014-r', 'decam2014-z',
            'wise2010-W1', 'wise2010-W2', 'wise2010-W3', 'wise2010-W4')
    wave_eff = [filter_response[i].effective_wavelength.value for i in range(len(filter_response))]

    fig = plt.figure(figsize=(14,4))
    gs1 = mpl.gridspec.GridSpec(1,1, figure=fig) 
    gs1.update(left=0.02, right=0.7)
    sub = plt.subplot(gs1[0])
    _plt_sed, = sub.plot(specs['wave'], specs['flux_unscaled'][0], c='k', lw=0.5, ls=':', 
            label='LGal Spectral Energy Distribution')
    _plt_photo = sub.errorbar(wave_eff[:3], [flux_g[0], flux_r[0], flux_z[0]], 
            yerr=[ivar_g[0]**-0.5, ivar_r[0]**-0.5, ivar_z[0]**-0.5], fmt='.C3', markersize=10,
            label='forward modeled DESI photometry') 
    _plt_filter, = sub.plot([0., 0.], [0., 0.], c='k', ls='--', label='Broadband Filter Response') 
    for i in range(3): # len(filter_response)): 
        sub.plot(filter_response[i].wavelength, filter_response[i].response, ls='--') 
        sub.text(filter_response[i].effective_wavelength.value, 0.9, ['g', 'r', 'z'][i], fontsize=20, color='C%i' % i)
    sub.set_xlabel('wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3500, 1e4)
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=20) 
    sub.set_ylim(0., 6.) 
    sub.legend([_plt_sed, _plt_photo, _plt_filter], 
            ['LGal Spectral Energy Distribution', 'forward modeled DESI photometry', 'Broadband Filter Response'], 
            loc='upper right', handletextpad=0.2, fontsize=15) 
    
    # Legacy imaging target photometry DR8
    bgs_true = h5py.File(os.path.join(UT.dat_dir(), 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    bgs_gmag = 22.5 - 2.5 * np.log10(bgs_true['flux_g'][...])
    bgs_rmag = 22.5 - 2.5 * np.log10(bgs_true['flux_r'][...]) 
    bgs_zmag = 22.5 - 2.5 * np.log10(bgs_true['flux_z'][...])
    rlim = (bgs_rmag < 20.) 
    
    photo_g = 22.5 - 2.5 * np.log10(photo['flux'][:,0])
    photo_r = 22.5 - 2.5 * np.log10(photo['flux'][:,1])
    photo_z = 22.5 - 2.5 * np.log10(photo['flux'][:,2])
        
    gs2 = mpl.gridspec.GridSpec(1,1, figure=fig) 
    gs2.update(left=0.76, right=1.)
    sub = plt.subplot(gs2[0])
    DFM.hist2d(bgs_gmag[rlim] - bgs_rmag[rlim], bgs_rmag[rlim] - bgs_zmag[rlim], color='k', levels=[0.68, 0.95], 
            range=[[-1., 3.], [-1., 3.]], bins=40, smooth=0.5, 
            plot_datapoints=False, fill_contours=False, plot_density=False, linewidth=0.5, ax=sub)
    sub.fill_between([0],[0],[0], fc='none', ec='k', label='Legacy Surveys Imaging') 
    sub.scatter(photo_g - photo_r, photo_r - photo_z, c='C3', s=1)#, label='forward modeled DESI photometry') 
    sub.set_xlabel('$g-r$', fontsize=20) 
    sub.set_xlim(0., 2.) 
    sub.set_xticks([0., 1., 2.]) 
    sub.set_ylabel('$r-z$', fontsize=20) 
    sub.set_ylim(0., 1.5) 
    sub.set_yticks([0., 1.]) 
    sub.legend(loc='upper left', fontsize=15) 

    ffig = os.path.join(dir_fig, 'fm_photo.png')
    fig.savefig(ffig, bbox_inches='tight')
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def FM_spec():
    ''' plot illustrating the forward model for spectroscopy 
    '''
    # read noiseless Lgal spectroscopy 
    spec_s, meta = Data.Spectra(sim='lgal', noise='none', lib='bc03', sample='mini_mocha') 
    spec_bgs, _ = Data.Spectra(sim='lgal', noise='bgs0', lib='bc03', sample='mini_mocha') 
    
    fig = plt.figure(figsize=(10,4))
    sub = fig.add_subplot(111) 

    wsort = np.argsort(spec_bgs['wave']) 
    _plt, = sub.plot(spec_bgs['wave'][wsort], spec_bgs['flux'][0,wsort], c='C0', lw=0.5) 
    
    _plt_lgal, = sub.plot(spec_s['wave'], spec_s['flux'][0,:], c='k', ls='-', lw=1) 
    _plt_lgal0, = sub.plot(spec_s['wave'], spec_s['flux_unscaled'][0,:], c='k', ls=':', lw=0.5) 
    
    leg = sub.legend([_plt_lgal0, _plt_lgal, _plt], 
            ['LGal Spectral Energy Distribution', 'fiber aperture SED', 'forward modeled BGS spectrum'],
            loc='upper right', handletextpad=0.3, fontsize=17) 
    sub.set_xlabel('wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3500, 1e4)
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=20) 
    sub.set_ylim(0., 6.) 

    ffig = os.path.join(dir_fig, 'fm_spec.png')
    fig.savefig(ffig, bbox_inches='tight')
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 

def mini_mocha_comparison(sfr='100myr'):
    ''' ultimate comparison between the different SED fitting methods 
    '''
    # read noiseless Lgal spectra of the spectral_challenge mocks
    specs, meta = Data.Spectra(sim='lgal', noise='bgs0', lib='bc03', sample='mini_mocha') 
    # read Lgal photometry of the mini_mocha mocks
    photo, _ = Data.Photometry(sim='lgal', noise='legacy', lib='bc03', sample='mini_mocha')

    Mstar_input = meta['logM_total'][:97] # total mass 
    M_fib_input = meta['logM_fiber'][:97] 
    logSFR_input= np.log10(meta['sfr_%s' % sfr][:97])
    Z_MW_input  = meta['Z_MW'][:97]  # mass-weighted metallicity
    tage_input  = meta['t_age_MW'][:97]  # mass-weighted age
    
    fig = plt.figure(figsize=(16,12))
    for i_m, method in enumerate(['ifsps', 'pfirefly', 'cigale']): 
        if method == 'ifsps': 
            theta_inf = [] 
            for igal in range(97): 
                # read best-fit file and get inferred parameters
                _fbf = Fbestfit_specphoto(igal, noise='bgs0_legacy', method=method) 
                fbf = h5py.File(_fbf, 'r')  

                theta_inf_i = np.array([
                    fbf['theta_2sig_minus'][...], 
                    fbf['theta_1sig_minus'][...], 
                    fbf['theta_med'][...], 
                    fbf['theta_1sig_plus'][...], 
                    fbf['theta_2sig_plus'][...]])
                if method == 'ifsps': 
                    ifsps = Fitters.iFSPS()
                if sfr == '1gyr': 
                    sfr_inf = ifsps._SFR_MCMC(fbf['mcmc_chain'][...], dt=1.)
                elif sfr == '100myr': 
                    sfr_inf = ifsps._SFR_MCMC(fbf['mcmc_chain'][...], dt=0.1)
                theta_inf_i = np.concatenate([theta_inf_i, np.atleast_2d(sfr_inf).T], axis=1) 

                theta_inf.append(theta_inf_i) 
            theta_inf = np.array(theta_inf) 

            # inferred properties
            Mstar_inf   = theta_inf[:,:,0]
            logSFR_inf  = np.log10(theta_inf[:,:,-1]) 
            Z_MW_inf    = 10**theta_inf[:,:,1]
            tage_inf    = theta_inf[:,:,2]
        else: 
            Mstar_inf   = np.tile(-999., (97, 5))  
            logSFR_inf  = np.tile(-999., (97, 5))
            Z_MW_inf    = np.tile(-999., (97, 5))
            tage_inf    = np.tile(-999., (97, 5))
        
        # compare total stellar mass 
        sub = fig.add_subplot(3,4,4*i_m+1) 
        sub.errorbar(Mstar_input, Mstar_inf[:,2], 
                yerr=[Mstar_inf[:,2]-Mstar_inf[:,1], Mstar_inf[:,3]-Mstar_inf[:,2]], fmt='.C0')
        sub.plot([9., 12.], [9., 12.], c='k', ls='--') 
        sub.text(0.05, 0.95, method, ha='left', va='top', transform=sub.transAxes, fontsize=25)

        sub.set_xlim(9., 12.) 
        if i_m < 2: sub.set_xticklabels([]) 
        sub.set_ylim(9., 12.) 
        sub.set_yticks([9., 10., 11., 12.]) 
        if i_m == 0:  sub.set_title(r'$\log~M_*$', fontsize=25)

        # compare SFR 
        sub = fig.add_subplot(3,4,4*i_m+2) 
        sub.errorbar(logSFR_input, logSFR_inf[:,2], 
                yerr=[logSFR_inf[:,2]-logSFR_inf[:,1], logSFR_inf[:,3]-logSFR_inf[:,2]], fmt='.C0')
        sub.plot([-3., 2.], [-3., 2.], c='k', ls='--') 
        sub.set_xlim(-3., 2.) 
        if i_m < 2: sub.set_xticklabels([]) 
        sub.set_ylim(-3., 2.) 
        sub.set_yticks([-2., 0., 2.]) 
        if sfr == '1gyr': lbl_sfr = '1Gyr'
        elif sfr == '100myr': lbl_sfr = '100Myr'
        if i_m == 0:  sub.set_title(r'$\log~{\rm SFR}_{%s}$' % lbl_sfr, fontsize=25)
        
        # compare metallicity
        sub = fig.add_subplot(3,4,4*i_m+3) 
        sub.errorbar(Z_MW_input, Z_MW_inf[:,2], 
                yerr=[Z_MW_inf[:,2]-Z_MW_inf[:,1], Z_MW_inf[:,3]-Z_MW_inf[:,2]], fmt='.C0')
        sub.plot([1e-3, 1], [1e-3, 1.], c='k', ls='--') 
        sub.set_xscale('log') 
        sub.set_xlim(1e-3, 5e-2) 
        if i_m < 2: sub.set_xticklabels([]) 
        sub.set_yscale('log') 
        sub.set_ylim(1e-3, 5e-2) 
        if i_m == 0: sub.set_title(r'MW $Z$', fontsize=25)

        # compare age 
        sub = fig.add_subplot(3,4,4*i_m+4) 
        sub.errorbar(tage_input, tage_inf[:,2], 
                yerr=[tage_inf[:,2]-tage_inf[:,1], tage_inf[:,3]-tage_inf[:,2]], fmt='.C0')
        sub.plot([0, 13], [0, 13.], c='k', ls='--') 
        sub.set_xlim(0, 13) 
        if i_m < 2: sub.set_xticklabels([]) 
        sub.set_ylim(0, 13) 
        sub.set_yticks([0., 5., 10.]) 
        if i_m == 0: sub.set_title(r'MW $t_{\rm age}$', fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$\theta_{\rm true}$', fontsize=25) 
    bkgd.set_ylabel(r'$\widehat{\theta}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.225, hspace=0.1)
    _ffig = os.path.join(dir_fig, 'mini_mocha.sfr_%s.comparison.png' % sfr) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def mock_challenge_photo(noise='none', dust=False, method='ifsps'): 
    ''' Compare properties inferred from forward modeled photometry to input properties
    '''
    # read Lgal input input properties
    _, meta = Data.Photometry(sim='lgal', noise=noise, lib='bc03', sample='spectral_challenge') 
    Mstar_input = meta['logM_total'] # total mass 
    Z_MW_input  = meta['Z_MW'] # mass-weighted metallicity
    tage_input  = meta['t_age_MW'] # mass-weighted age
    
    theta_inf = [] 
    for igal in range(97): 
        # read best-fit file and get inferred parameters
        _fbf = Fbestfit_photo(igal, noise=noise, dust=dust, method=method) 
        fbf = h5py.File(_fbf, 'r')  

        theta_inf_i = np.array([
            fbf['theta_2sig_minus'][...], 
            fbf['theta_1sig_minus'][...], 
            fbf['theta_med'][...], 
            fbf['theta_1sig_plus'][...], 
            fbf['theta_2sig_plus'][...]])
        theta_inf.append(theta_inf_i) 
    theta_inf = np.array(theta_inf) 
    
    # inferred properties
    Mstar_inf   = theta_inf[:,:,0]
    Z_MW_inf    = 10**theta_inf[:,:,1]
    tage_inf    = theta_inf[:,:,2]
    
    fig = plt.figure(figsize=(15,4))
    # compare total stellar mass 
    sub = fig.add_subplot(131) 
    sub.errorbar(Mstar_input, Mstar_inf[:,2], 
            yerr=[Mstar_inf[:,2]-Mstar_inf[:,1], Mstar_inf[:,3]-Mstar_inf[:,2]], fmt='.C0')
    sub.plot([9., 12.], [9., 12.], c='k', ls='--') 
    sub.set_xlabel(r'input $\log~M_{\rm tot}$', fontsize=25)
    sub.set_xlim(9., 12.) 
    sub.set_ylabel(r'inferred $\log~M_{\rm tot}$', fontsize=25)
    sub.set_ylim(9., 12.) 
    
    # compare metallicity
    sub = fig.add_subplot(132)
    sub.errorbar(Z_MW_input, Z_MW_inf[:,2], 
            yerr=[Z_MW_inf[:,2]-Z_MW_inf[:,1], Z_MW_inf[:,3]-Z_MW_inf[:,2]], fmt='.C0')
    sub.plot([1e-3, 1], [1e-3, 1.], c='k', ls='--') 
    sub.set_xlabel(r'input MW $Z$', fontsize=20)
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 5e-2) 
    sub.set_ylabel(r'inferred MW $Z$', fontsize=20)
    sub.set_yscale('log') 
    sub.set_ylim(1e-3, 5e-2) 

    # compare age 
    sub = fig.add_subplot(133)
    sub.errorbar(tage_input, tage_inf[:,2], 
            yerr=[tage_inf[:,2]-tage_inf[:,1], tage_inf[:,3]-tage_inf[:,2]], fmt='.C0')
    sub.plot([0, 13], [0, 13.], c='k', ls='--') 
    sub.set_xlabel(r'input MW $t_{\rm age}$', fontsize=20)
    sub.set_xlim(0, 13) 
    sub.set_ylabel(r'inferred MW $t_{\rm age}$', fontsize=20)
    sub.set_ylim(0, 13) 

    fig.subplots_adjust(wspace=0.4)
    _ffig = os.path.join(dir_fig, 'mock_challenge.photofit.%s.noise_%s.dust_%s.png' % (method, noise, ['no', 'yes'][dust]))
    fig.savefig(_ffig, bbox_inches='tight') 
    return None 


def mini_mocha_spec(noise='bgs0', method='ifsps', sfr='1gyr'): 
    ''' Compare properties inferred from forward modeled photometry to input properties
    '''
    # read noiseless Lgal spectra of the spectral_challenge mocks
    specs, meta = Data.Spectra(sim='lgal', noise=noise, lib='bc03', sample='mini_mocha') 

    Mstar_input = meta['logM_fiber'][:97] # total mass 
    logSFR_input= np.log10(meta['sfr_%s' % sfr][:97])
    Z_MW_input  = meta['Z_MW'][:97]  # mass-weighted metallicity
    tage_input  = meta['t_age_MW'][:97]  # mass-weighted age
    
    theta_inf = [] 
    for igal in range(97): 
        # read best-fit file and get inferred parameters
        _fbf = Fbestfit_spec(igal, noise=noise, method=method) 
        fbf = h5py.File(_fbf, 'r')  

        theta_inf_i = np.array([
            fbf['theta_2sig_minus'][...], 
            fbf['theta_1sig_minus'][...], 
            fbf['theta_med'][...], 
            fbf['theta_1sig_plus'][...], 
            fbf['theta_2sig_plus'][...]])
        
        if method == 'ifsps': 
            ifsps = Fitters.iFSPS()
        if sfr == '1gyr': 
            sfr_inf = ifsps._SFR_MCMC(fbf['mcmc_chain'][...], dt=1.)
        elif sfr == '100myr': 
            sfr_inf = ifsps._SFR_MCMC(fbf['mcmc_chain'][...], dt=0.1)
        theta_inf_i = np.concatenate([theta_inf_i, np.atleast_2d(sfr_inf).T], axis=1) 

        theta_inf.append(theta_inf_i) 
    theta_inf = np.array(theta_inf) 
    
    # inferred properties
    Mstar_inf   = theta_inf[:,:,0]
    logSFR_inf  = np.log10(theta_inf[:,:,-1]) 
    Z_MW_inf    = 10**theta_inf[:,:,1]
    tage_inf    = theta_inf[:,:,2]
    
    fig = plt.figure(figsize=(20,4))
    # compare total stellar mass 
    sub = fig.add_subplot(141) 
    sub.errorbar(Mstar_input, Mstar_inf[:,2], 
            yerr=[Mstar_inf[:,2]-Mstar_inf[:,1], Mstar_inf[:,3]-Mstar_inf[:,2]], fmt='.C0')
    sub.plot([9., 12.], [9., 12.], c='k', ls='--') 
    sub.set_xlabel(r'input $\log~M_{\rm fib.}$', fontsize=25)
    sub.set_xlim(9., 12.) 
    sub.set_ylabel(r'inferred $\log~M_{\rm fib.}$', fontsize=25)
    sub.set_ylim(9., 12.) 

    # compare SFR
    sub = fig.add_subplot(142) 
    sub.errorbar(logSFR_input, logSFR_inf[:,2], 
            yerr=[logSFR_inf[:,2]-logSFR_inf[:,1], logSFR_inf[:,3]-logSFR_inf[:,2]], fmt='.C0')
    sub.plot([-3., 2.], [-3., 2.], c='k', ls='--') 
    if sfr == '1gyr': lbl_sfr = '1Gyr'
    elif sfr == '100myr': lbl_sfr = '100Myr'
    sub.set_xlabel(r'input $\log~{\rm SFR}_{%s}$' % lbl_sfr, fontsize=25)
    sub.set_xlim(-3., 2.) 
    sub.set_ylabel(r'inferred $\log~{\rm SFR}_{%s}$' % lbl_sfr, fontsize=25)
    sub.set_ylim(-3., 2.) 
    
    # compare metallicity
    sub = fig.add_subplot(143)
    sub.errorbar(Z_MW_input, Z_MW_inf[:,2], 
            yerr=[Z_MW_inf[:,2]-Z_MW_inf[:,1], Z_MW_inf[:,3]-Z_MW_inf[:,2]], fmt='.C0')
    sub.plot([1e-3, 1], [1e-3, 1.], c='k', ls='--') 
    sub.set_xlabel(r'input MW $Z$', fontsize=20)
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 5e-2) 
    sub.set_ylabel(r'inferred MW $Z$', fontsize=20)
    sub.set_yscale('log') 
    sub.set_ylim(1e-3, 5e-2) 

    # compare age 
    sub = fig.add_subplot(144)
    sub.errorbar(tage_input, tage_inf[:,2], 
            yerr=[tage_inf[:,2]-tage_inf[:,1], tage_inf[:,3]-tage_inf[:,2]], fmt='.C0')
    sub.plot([0, 13], [0, 13.], c='k', ls='--') 
    sub.set_xlabel(r'input MW $t_{\rm age}$', fontsize=20)
    sub.set_xlim(0, 13) 
    sub.set_ylabel(r'inferred MW $t_{\rm age}$', fontsize=20)
    sub.set_ylim(0, 13) 

    fig.subplots_adjust(wspace=0.4)
    _ffig = os.path.join(dir_fig, 'mini_mocha.sfr_%s.%s.specfit.vanilla.noise_%s.png' % (sfr, method, noise)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def mini_mocha_photo(noise='legacy', method='ifsps', sfr='1gyr'): 
    ''' Compare properties inferred from forward modeled photometry to input properties
    '''
    # read noiseless Lgal spectra of the spectral_challenge mocks
    photo, meta = Data.Photometry(sim='lgal', noise=noise, lib='bc03', sample='mini_mocha')

    Mstar_input = meta['logM_total'][:97] # total mass 
    logSFR_input= np.log10(meta['sfr_%s' % sfr][:97])
    Z_MW_input  = meta['Z_MW'][:97]  # mass-weighted metallicity
    tage_input  = meta['t_age_MW'][:97]  # mass-weighted age
    
    theta_inf = [] 
    for igal in range(97): 
        # read best-fit file and get inferred parameters
        _fbf = Fbestfit_photo(igal, noise=noise, method=method) 
        fbf = h5py.File(_fbf, 'r')  

        theta_inf_i = np.array([
            fbf['theta_2sig_minus'][...], 
            fbf['theta_1sig_minus'][...], 
            fbf['theta_med'][...], 
            fbf['theta_1sig_plus'][...], 
            fbf['theta_2sig_plus'][...]])
        
        if method == 'ifsps': 
            ifsps = Fitters.iFSPS()
        if sfr == '1gyr': 
            sfr_inf = ifsps._SFR_MCMC(fbf['mcmc_chain'][...], dt=1.)
        elif sfr == '100myr': 
            sfr_inf = ifsps._SFR_MCMC(fbf['mcmc_chain'][...], dt=0.1)
        theta_inf_i = np.concatenate([theta_inf_i, np.atleast_2d(sfr_inf).T], axis=1) 
        theta_inf.append(theta_inf_i) 

    theta_inf = np.array(theta_inf) 
    
    # inferred properties
    Mstar_inf   = theta_inf[:,:,0]
    logSFR_inf  = np.log10(theta_inf[:,:,-1]) 
    Z_MW_inf    = 10**theta_inf[:,:,1]
    tage_inf    = theta_inf[:,:,2]
    
    fig = plt.figure(figsize=(20,4))
    # compare total stellar mass 
    sub = fig.add_subplot(141) 
    sub.errorbar(Mstar_input, Mstar_inf[:,2], 
            yerr=[Mstar_inf[:,2]-Mstar_inf[:,1], Mstar_inf[:,3]-Mstar_inf[:,2]], fmt='.C0')
    sub.plot([9., 12.], [9., 12.], c='k', ls='--') 
    sub.set_xlabel(r'input $\log~M_{\rm tot.}$', fontsize=25)
    sub.set_xlim(9., 12.) 
    sub.set_ylabel(r'inferred $\log~M_{\rm tot.}$', fontsize=25)
    sub.set_ylim(9., 12.) 
    
    # compare SFR 
    sub = fig.add_subplot(142) 
    sub.errorbar(logSFR_input, logSFR_inf[:,2], 
            yerr=[logSFR_inf[:,2]-logSFR_inf[:,1], logSFR_inf[:,3]-logSFR_inf[:,2]], fmt='.C0')
    sub.plot([-3., 2.], [-3., 2.], c='k', ls='--') 
    if sfr == '1gyr': lbl_sfr = '1Gyr'
    elif sfr == '100myr': lbl_sfr = '100Myr'
    sub.set_xlabel(r'input $\log~{\rm SFR}_{%s}$' % lbl_sfr, fontsize=25)
    sub.set_xlim(-3., 2.) 
    sub.set_ylabel(r'inferred $\log~{\rm SFR}_{%s}$' % lbl_sfr, fontsize=25)
    sub.set_ylim(-3., 2.) 
    
    # compare metallicity
    sub = fig.add_subplot(143)
    sub.errorbar(Z_MW_input, Z_MW_inf[:,2], 
            yerr=[Z_MW_inf[:,2]-Z_MW_inf[:,1], Z_MW_inf[:,3]-Z_MW_inf[:,2]], fmt='.C0')
    sub.plot([1e-3, 1], [1e-3, 1.], c='k', ls='--') 
    sub.set_xlabel(r'input MW $Z$', fontsize=20)
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 5e-2) 
    sub.set_ylabel(r'inferred MW $Z$', fontsize=20)
    sub.set_yscale('log') 
    sub.set_ylim(1e-3, 5e-2) 

    # compare age 
    sub = fig.add_subplot(144)
    sub.errorbar(tage_input, tage_inf[:,2], 
            yerr=[tage_inf[:,2]-tage_inf[:,1], tage_inf[:,3]-tage_inf[:,2]], fmt='.C0')
    sub.plot([0, 13], [0, 13.], c='k', ls='--') 
    sub.set_xlabel(r'input MW $t_{\rm age}$', fontsize=20)
    sub.set_xlim(0, 13) 
    sub.set_ylabel(r'inferred MW $t_{\rm age}$', fontsize=20)
    sub.set_ylim(0, 13) 

    fig.subplots_adjust(wspace=0.4)
    _ffig = os.path.join(dir_fig, 'mini_mocha.sfr_%s.%s.photofit.vanilla.noise_%s.png' % (sfr, method, noise)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def mini_mocha_specphoto(noise='bgs0_legacy', method='ifsps', sfr='1gyr'): 
    ''' Compare properties inferred from forward modeled photometry to input properties
    '''
    if noise != 'none':
        noise_spec = noise.split('_')[0]
        noise_photo = noise.split('_')[1]
    else:
        noise_spec = 'none'
        noise_photo = 'none'
    # read noiseless Lgal spectra of the spectral_challenge mocks
    specs, meta = Data.Spectra(sim='lgal', noise=noise_spec, lib='bc03', sample='mini_mocha') 
    # read Lgal photometry of the mini_mocha mocks
    photo, _ = Data.Photometry(sim='lgal', noise=noise_photo, lib='bc03', sample='mini_mocha')

    Mstar_input = meta['logM_total'][:97] # total mass 
    logSFR_input= np.log10(meta['sfr_%s' % sfr][:97])
    Z_MW_input  = meta['Z_MW'][:97]  # mass-weighted metallicity
    tage_input  = meta['t_age_MW'][:97]  # mass-weighted age
    
    theta_inf = [] 
    for igal in range(97): 
        # read best-fit file and get inferred parameters
        _fbf = Fbestfit_specphoto(igal, noise=noise, method=method) 
        fbf = h5py.File(_fbf, 'r')  

        theta_inf_i = np.array([
            fbf['theta_2sig_minus'][...], 
            fbf['theta_1sig_minus'][...], 
            fbf['theta_med'][...], 
            fbf['theta_1sig_plus'][...], 
            fbf['theta_2sig_plus'][...]])
        if method == 'ifsps': 
            ifsps = Fitters.iFSPS()
        if sfr == '1gyr': 
            sfr_inf = ifsps._SFR_MCMC(fbf['mcmc_chain'][...], dt=1.)
        elif sfr == '100myr': 
            sfr_inf = ifsps._SFR_MCMC(fbf['mcmc_chain'][...], dt=0.1)
        theta_inf_i = np.concatenate([theta_inf_i, np.atleast_2d(sfr_inf).T], axis=1) 

        theta_inf.append(theta_inf_i) 
    theta_inf = np.array(theta_inf) 
    
    # inferred properties
    Mstar_inf   = theta_inf[:,:,0]
    logSFR_inf  = np.log10(theta_inf[:,:,-1]) 
    Z_MW_inf    = 10**theta_inf[:,:,1]
    tage_inf    = theta_inf[:,:,2]
    
    fig = plt.figure(figsize=(20,4))
    # compare total stellar mass 
    sub = fig.add_subplot(141) 
    sub.errorbar(Mstar_input, Mstar_inf[:,2], 
            yerr=[Mstar_inf[:,2]-Mstar_inf[:,1], Mstar_inf[:,3]-Mstar_inf[:,2]], fmt='.C0')
    sub.plot([9., 12.], [9., 12.], c='k', ls='--') 
    sub.set_xlabel(r'input $\log~M_{\rm tot}$', fontsize=25)
    sub.set_xlim(9., 12.) 
    sub.set_ylabel(r'inferred $\log~M_{\rm tot}$', fontsize=25)
    sub.set_ylim(9., 12.) 
    
    # compare SFR 
    sub = fig.add_subplot(142) 
    sub.errorbar(logSFR_input, logSFR_inf[:,2], 
            yerr=[logSFR_inf[:,2]-logSFR_inf[:,1], logSFR_inf[:,3]-logSFR_inf[:,2]], fmt='.C0')
    sub.plot([-3., 2.], [-3., 2.], c='k', ls='--') 
    if sfr == '1gyr': lbl_sfr = '1Gyr'
    elif sfr == '100myr': lbl_sfr = '100Myr'
    sub.set_xlabel(r'input $\log~{\rm SFR}_{%s}$' % lbl_sfr, fontsize=25)
    sub.set_xlim(-3., 2.) 
    sub.set_ylabel(r'inferred $\log~{\rm SFR}_{%s}$' % lbl_sfr, fontsize=25)
    sub.set_ylim(-3., 2.) 
    
    # compare metallicity
    sub = fig.add_subplot(143)
    sub.errorbar(Z_MW_input, Z_MW_inf[:,2], 
            yerr=[Z_MW_inf[:,2]-Z_MW_inf[:,1], Z_MW_inf[:,3]-Z_MW_inf[:,2]], fmt='.C0')
    sub.plot([1e-3, 1], [1e-3, 1.], c='k', ls='--') 
    sub.set_xlabel(r'input MW $Z$', fontsize=20)
    sub.set_xscale('log') 
    sub.set_xlim(1e-3, 5e-2) 
    sub.set_ylabel(r'inferred MW $Z$', fontsize=20)
    sub.set_yscale('log') 
    sub.set_ylim(1e-3, 5e-2) 

    # compare age 
    sub = fig.add_subplot(144)
    sub.errorbar(tage_input, tage_inf[:,2], 
            yerr=[tage_inf[:,2]-tage_inf[:,1], tage_inf[:,3]-tage_inf[:,2]], fmt='.C0')
    sub.plot([0, 13], [0, 13.], c='k', ls='--') 
    sub.set_xlabel(r'input MW $t_{\rm age}$', fontsize=20)
    sub.set_xlim(0, 13) 
    sub.set_ylabel(r'inferred MW $t_{\rm age}$', fontsize=20)
    sub.set_ylim(0, 13) 

    fig.subplots_adjust(wspace=0.4)
    _ffig = os.path.join(dir_fig, 'mini_mocha.sfr_%s.%s.specphotofit.vanilla.noise_%s.png' % (sfr, method, noise)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def photo_vs_specphoto(noise_photo='legacy', noise_specphoto='bgs0_legacy', method='ifsps', sfr='1gyr'):  
    ''' Compare properties inferred from photometry versus spectrophotometry to see how much
    information is gained from adding spectra
    '''
    import scipy.optimize as op
    assert noise_specphoto.split('_')[1] == noise_photo
    # read noiseless Lgal spectra of the spectral_challenge mocks
    specs, meta = Data.Spectra(sim='lgal', noise=noise_specphoto.split('_')[0], lib='bc03', sample='mini_mocha') 
    # read Lgal photometry of the mini_mocha mocks
    photo, _ = Data.Photometry(sim='lgal', noise=noise_specphoto.split('_')[1], lib='bc03', sample='mini_mocha')

    # --------------------------------------------------------------------------------
    # true paramater values
    logMstar_input  = np.array(meta['logM_total'][:97]) # total mass 
    logSFR_input    = np.log10(np.array(meta['sfr_%s' % sfr][:97])) 
    Z_MW_input      = meta['Z_MW'][:97]  # mass-weighted metallicity
    tage_input      = meta['t_age_MW'][:97]  # mass-weighted age

    # --------------------------------------------------------------------------------
    # assemble all markov chains 
    dlogMs_photo, dlogMs_specphoto = [], []
    dlogSFR_photo, dlogSFR_specphoto = [], []
    for igal in range(97): 
        # read best-fit file and get inferred parameters from photometry
        _fbf = Fbestfit_photo(igal, noise=noise_photo, method=method) 
        fbf = h5py.File(_fbf, 'r')  
        # d log M* chain
        chain = fbf['mcmc_chain'][...][::10]
        dlogMs_chain = chain[:,0] - logMstar_input[igal] 
        # calculate average SFR
        if method == 'ifsps': 
            ifsps = Fitters.iFSPS()
            if sfr == '1gyr': 
                sfr_chain, _ = ifsps.get_SFR(chain, dt=1.)
            elif sfr == '100myr': 
                sfr_chain, _ = ifsps.get_SFR(chain, dt=0.1)
        dlogSFR_chain = np.log10(sfr_chain) - logSFR_input[igal]

        dlogMs_photo.append(dlogMs_chain)
        dlogSFR_photo.append(dlogSFR_chain)

        # read best-fit file and get inferred parameters from spectrophoto
        _fbf = Fbestfit_specphoto(igal, noise=noise_specphoto, method=method) 
        fbf = h5py.File(_fbf, 'r')  
        # d log M* chain
        chain = fbf['mcmc_chain'][...][::10]
        dlogMs_chain = chain[:,0] - logMstar_input[igal] 
        # calculate average SFR
        if method == 'ifsps': 
            ifsps = Fitters.iFSPS()
            if sfr == '1gyr': 
                sfr_chain, _ = ifsps.get_SFR(chain, dt=1.)
            elif sfr == '100myr': 
                sfr_chain, _ = ifsps.get_SFR(chain, dt=0.1)
        dlogSFR_chain = np.log10(sfr_chain) - logSFR_input[igal]

        dlogMs_specphoto.append(dlogMs_chain)
        dlogSFR_specphoto.append(dlogSFR_chain)
    
    dlogMs_photo = np.array(dlogMs_photo)  
    dlogSFR_photo = np.array(dlogSFR_photo) 
    dlogMs_specphoto = np.array(dlogMs_specphoto) 
    dlogSFR_specphoto = np.array(dlogSFR_specphoto) 

    # --------------------------------------------------------------------------------
    # maximum likelihood for the population hyperparameters
    Mbins = np.linspace(9., 12., 16) 
    Mbin_mid = [] 
    mu_dMstar_photo, sig_dMstar_photo = [], []
    mu_dMstar_specphoto, sig_dMstar_specphoto = [], [] 
    for i_m in range(len(Mbins)-1): 
        inmbin = ((logMstar_input > Mbins[i_m]) & (logMstar_input <= Mbins[i_m+1])) 
        if np.sum(inmbin) == 0: continue 
        Mbin_mid.append(0.5 * (Mbins[i_m] + Mbins[i_m+1])) 
        print('%.f < log M* < %.f' % (Mbins[i_m], Mbins[i_m+1]))  
        print('%i galaxies' % np.sum(inmbin))

        L_pop_photo = lambda _theta: -1.*logL_pop(_theta[0], _theta[1], delta_chains=dlogMs_photo[inmbin])  
        L_pop_specphoto = lambda _theta: -1.*logL_pop(_theta[0], _theta[1], delta_chains=dlogMs_specphoto[inmbin])  

        min_photo = op.minimize(
                L_pop_photo, 
                np.array([0., 0.1]), # guess the middle of the prior 
                method='L-BFGS-B', 
                bounds=((None, None), (1e-4, None)),
                options={'eps': np.array([0.01, 0.005]), 'maxiter': 100})
        print(min_photo['x'])

        min_specphoto = op.minimize(
                L_pop_specphoto, 
                np.array([0., 0.1]), # guess the middle of the prior 
                method='L-BFGS-B', 
                bounds=((None, None), (1e-4, None)),
                options={'eps': np.array([0.01, 0.005]), 'maxiter': 100})
        print(min_specphoto['x'])

        mu_dMstar_photo.append(min_photo['x'][0]) 
        sig_dMstar_photo.append(min_photo['x'][1]) 

        mu_dMstar_specphoto.append(min_specphoto['x'][0]) 
        sig_dMstar_specphoto.append(min_specphoto['x'][1]) 

    mu_dMstar_photo = np.array(mu_dMstar_photo) 
    sig_dMstar_photo = np.array(sig_dMstar_photo) 
    mu_dMstar_specphoto = np.array(mu_dMstar_specphoto) 
    sig_dMstar_specphoto = np.array(sig_dMstar_specphoto) 
    
    # calculate delta log SFR 
    logSFRbins = np.linspace(-3., 3., 25) 
    logSFRbin_mid = [] 
    mu_dlogSFR_photo, sig_dlogSFR_photo = [], []
    mu_dlogSFR_specphoto, sig_dlogSFR_specphoto = [], [] 
    for i_m in range(len(Mbins)-1): 
        inbin = ((logSFR_input > logSFRbins[i_m]) & (logSFR_input <= logSFRbins[i_m+1])) 
        if sfr == '1gyr': 
            inbin = inbin & (np.array(tage_input) > 1.0) 
        elif sfr == '100myr': 
            inbin = inbin & (np.array(tage_input) > 0.1)
        if np.sum(inbin) == 0: continue 
        print('%i galaxies' % np.sum(inbin))
        logSFRbin_mid.append(0.5 * (logSFRbins[i_m] + logSFRbins[i_m+1])) 

        L_pop_photo = lambda _theta: -1.*logL_pop(_theta[0], _theta[1], delta_chains=dlogSFR_photo[inbin])  
        L_pop_specphoto = lambda _theta: -1.*logL_pop(_theta[0], _theta[1], delta_chains=dlogSFR_specphoto[inbin])  

        min_photo = op.minimize(
                L_pop_photo, 
                np.array([0., 0.1]), # guess the middle of the prior 
                method='L-BFGS-B', 
                bounds=((None, None), (1e-4, None)),
                options={'eps': np.array([0.01, 0.005]), 'maxiter': 100})
        print(min_photo['x'])

        min_specphoto = op.minimize(
                L_pop_specphoto, 
                np.array([0., 0.1]), # guess the middle of the prior 
                method='L-BFGS-B', 
                bounds=((None, None), (1e-4, None)),
                options={'eps': np.array([0.01, 0.005]), 'maxiter': 100})
        print(min_specphoto['x'])

        mu_dlogSFR_photo.append(min_photo['x'][0]) 
        sig_dlogSFR_photo.append(min_photo['x'][1]) 

        mu_dlogSFR_specphoto.append(min_specphoto['x'][0]) 
        sig_dlogSFR_specphoto.append(min_specphoto['x'][1]) 

    mu_dlogSFR_photo = np.array(mu_dlogSFR_photo) 
    sig_dlogSFR_photo = np.array(sig_dlogSFR_photo) 
    mu_dlogSFR_specphoto = np.array(mu_dlogSFR_specphoto) 
    sig_dlogSFR_specphoto = np.array(sig_dlogSFR_specphoto) 

    fig = plt.figure(figsize=(12,5))
    # compare total stellar mass 
    sub = fig.add_subplot(121) 
    sub.plot([9., 12.], [0., 0.], c='k', ls='--')
    sub.fill_between(Mbin_mid, mu_dMstar_photo - sig_dMstar_photo, mu_dMstar_photo + sig_dMstar_photo, 
            fc='C0', ec='none', alpha=0.5, label='Photometry only') 
    sub.scatter(Mbin_mid, mu_dMstar_photo, c='C0', s=2) 
    sub.plot(Mbin_mid, mu_dMstar_photo, c='C0') 
    sub.fill_between(Mbin_mid, mu_dMstar_specphoto - sig_dMstar_specphoto, mu_dMstar_specphoto + sig_dMstar_specphoto, 
            fc='C1', ec='none', alpha=0.5, label='Photometry+Spectroscopy') 
    sub.scatter(Mbin_mid, mu_dMstar_specphoto, c='C1', s=1) 
    sub.plot(Mbin_mid, mu_dMstar_specphoto, c='C1') 
    #sub.scatter(Mstar_input, (Mstar_inf_photo[:,2]-Mstar_input), c='C0')
    #sub.scatter(Mstar_input, (Mstar_inf_specphoto[:,2]-Mstar_input), c='C1')
    sub.set_xlabel(r'$\log(~M_*~[M_\odot]~)$', fontsize=25)
    sub.set_xlim(9., 12.) 
    sub.set_ylabel(r'$\Delta_{\log M_*}$', fontsize=25)
    sub.set_ylim(-1., 1.) 
    sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    
    # compare SFR 
    sub = fig.add_subplot(122) 
    sub.plot([-3., 3.], [0., 0.], c='k', ls='--')
    sub.fill_between(logSFRbin_mid, mu_dlogSFR_photo - sig_dlogSFR_photo, mu_dlogSFR_photo + sig_dlogSFR_photo, 
            fc='C0', ec='none', alpha=0.5, label='Photometry only') 
    sub.scatter(logSFRbin_mid, mu_dlogSFR_photo, c='C0', s=2) 
    sub.plot(logSFRbin_mid, mu_dlogSFR_photo, c='C0') 
    sub.fill_between(logSFRbin_mid, mu_dlogSFR_specphoto - sig_dlogSFR_specphoto, mu_dlogSFR_specphoto + sig_dlogSFR_specphoto, 
            fc='C1', ec='none', alpha=0.5, label='Photometry+Spectroscopy') 
    sub.scatter(logSFRbin_mid, mu_dlogSFR_specphoto, c='C1', s=1) 
    sub.plot(logSFRbin_mid, mu_dlogSFR_specphoto, c='C1') 
    
    if sfr == '1gyr': lbl_sfr = '1Gyr'
    elif sfr == '100myr': lbl_sfr = '100Myr'
    sub.set_xlabel(r'$\log(~{\rm SFR}_{%s}~[M_\odot/yr]~)$' % lbl_sfr, fontsize=25)
    sub.set_xlim(-3., 1.) 
    sub.set_ylabel(r'$\Delta_{\log{\rm SFR}_{%s}}$' % lbl_sfr, fontsize=25)
    sub.set_ylim(-3., 3.) 
    #sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    
    fig.subplots_adjust(wspace=0.3)
    _ffig = os.path.join(dir_doc, 'photo_vs_specphoto.%s.sfr_%s.vanilla.noise_%s_%s.png' % (method, sfr, noise_photo, noise_specphoto)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def logL_pop(mu_pop, sigma_pop, delta_chains=None, prior=None): 
    ''' log likelihood of population variables mu, sigma
    
    :param mu_pop: 

    :param sigma_pop: 

    :param delta_chains: (default: None) 
        Ngal x Niter 

    :param prior: (default: None) 
        prior function  
    '''
    if prior is None: prior = lambda x: 1. # uninformative prior default 

    N = delta_chains.shape[0] 

    logp_D_pop = 0. 
    for i in range(N): 
        K = len(delta_chains[i]) 
        gauss = Norm(loc=mu_pop, scale=sigma_pop) 

        p_Di_pop = np.sum(gauss.pdf(delta_chains[i])/prior(delta_chains[i]))/float(K)

        logp_D_pop += np.log(p_Di_pop) 

    #print('%.4f, %.4f, %.4f' % (mu_pop, sigma_pop, logp_D_pop)) 
    if np.isnan(logp_D_pop): raise ValueError
    return logp_D_pop     


def Fbestfit_spec(igal, noise='none', method='ifsps'): 
    ''' file name of best-fit of spectra of spectral_challenge galaxy #igal 

    :param igal: 
        index of spectral_challenge galaxy 

    :param noise:
        noise of the spectra. If noise == 'none', no noise. If noise =='bgs0' - 'bgs7', 
        then BGS like noise. (default: 'none') 

    :param dust: 
        spectra has dust or not. 
    
    :param method: 
        fitting method. (default: ifsps)

    '''
    model = 'vanilla'
    if method == 'ifsps': 
        f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ifsps', 'lgal.spec.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    elif method == 'pfirefly': 
        f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'pff', 'lgal.spec.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    return f_bf


def Fbestfit_photo(igal, noise='none', method='ifsps'): 
    ''' file name of best-fit of photometry of spectral_challenge galaxy #igal

    :param igal: 
        index of spectral_challenge galaxy 

    :param noise:
        noise of the spectra. If noise == 'none', no noise. If noise =='legacy', 
        then legacy like noise. (default: 'none') 

    :param dust: 
        spectra has dust or not. 
    
    :param method: 
        fitting method. (default: ifsps)
    '''
    model = 'vanilla'

    if method == 'ifsps': 
        f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ifsps', 'lgal.photo.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    elif method == 'pfirefly': 
        f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'pff', 'lgal.photo.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    return f_bf


def Fbestfit_specphoto(igal, noise='bgs0_legacy', method='ifsps'): 
    ''' file name of best-fit of photometry of spectral_challenge galaxy #igal

    :param igal: 
        index of spectral_challenge galaxy 

    :param noise:
        noise of the spectra. If noise == 'none', no noise. If noise =='legacy', 
        then legacy like noise. (default: 'none') 

    :param dust: 
        spectra has dust or not. 
    
    :param method: 
        fitting method. (default: ifsps)
    '''
    model = 'vanilla'   
    f_bf = os.path.join(UT.dat_dir(), 'mini_mocha', 'ifsps', 'lgal.specphoto.noise_%s.%s.%i.hdf5' % (noise, model, igal))
    return f_bf


if __name__=="__main__": 
    #BGS()
    #FM_photo()
    #FM_spec()
    #photo_vs_specphoto(noise_photo='legacy', noise_specphoto='bgs0_legacy', method='ifsps', sfr='1gyr')
    photo_vs_specphoto(noise_photo='legacy', noise_specphoto='bgs0_legacy', method='ifsps', sfr='100myr')
    #mini_mocha_comparison(sfr='100myr')

    #mock_challenge_photo(noise='none', dust=False, method='ifsps')
    #mock_challenge_photo(noise='none', dust=True, method='ifsps')
    #mock_challenge_photo(noise='legacy', dust=False, method='ifsps')
    #mock_challenge_photo(noise='legacy', dust=True, method='ifsps')

    #mini_mocha_spec(noise='bgs0', method='ifsps', sfr='1gyr')
    #mini_mocha_spec(noise='bgs0', method='ifsps', sfr='100myr')
    #mini_mocha_photo(noise='legacy', method='ifsps', sfr='1gyr')
    #mini_mocha_photo(noise='legacy', method='ifsps', sfr='100myr')
    #mini_mocha_specphoto(noise='bgs0_legacy', method='ifsps', sfr='1gyr')
    #mini_mocha_specphoto(noise='bgs0_legacy', method='ifsps', sfr='100myr')
    
    #mini_mocha_spec(noise='bgs0', method='pfirefly')
    
