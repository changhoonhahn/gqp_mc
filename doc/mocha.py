'''

generate plots for the mock challenge paper 


'''
import os 
import h5py 
import numpy as np 
import corner as DFM 
import scipy.optimize as op
from scipy.stats import norm as Norm
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
# --- provabgs --- 
from provabgs import infer as Infer
# --- astro ---
from astropy.io import fits
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
if os.environ.get('NERSC_HOST') is None: 
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


dir_mm = os.path.join(UT.dat_dir(), 'mini_mocha') 
dir_fig = os.path.join(UT.dat_dir(), 'mini_mocha') 

dir_doc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', 'figs') 
dir_fbgs = os.path.join(os.path.dirname(os.path.dirname(UT.dat_dir())), 'feasiBGS') 

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



def BGS(): 
    ''' plot highlighting BGS footprint and redshift number density
    '''
    # read BGS MXXL mock galaxies 
    fmxxl = os.path.join(UT.dat_dir(), 'mxxl.bgs_r20.6.hdf5')
    mxxl = h5py.File(fmxxl, 'r')

    bgs     = mxxl['Data']
    ra_bgs  = bgs['ra'][...]
    dec_bgs = bgs['dec'][...]
    z_bgs   = np.array(bgs['z_obs'])

    M_r = bgs['abs_mag'][...] # absolute magnitude
    m_r = bgs['app_mag'][...] # r-band magnitude
    g_r = bgs['g_r'][...] # g-r color 

    # read SDSS 
    from astrologs.astrologs import Astrologs 
    sdss = Astrologs('vagc', sample='vagc', cross_nsa=False) 
    ra_sdss     = sdss.data['ra'] 
    dec_sdss    = sdss.data['dec'] 
    z_sdss      = sdss.data['redshift'] 
    M_r_sdss    = sdss.data['M_r']
    g_r_sdss    = sdss.data['m_g'] - sdss.data['m_r'] 
     
    # read GAMA objects
    f_gama = os.path.join(dir_fbgs, 'gama', 'dr3', 'SpecObj.fits') 
    gama = fits.open(f_gama)[1].data 
    
    #-------------------------------------------------------------------
    # stellar mass estimate using g-r color 
    #-------------------------------------------------------------------
    M_r_sun = 4.67 # Bell+(2003)

    def MtoL_Bell2003(g_r):
        # Bell+(2003) M/L ratio
        return 10**(-0.306 + (1.097 * g_r))

    def Mr_to_Mstar(Mr, g_r):
        '''given r-band abs mag calculate M*
        '''
        M_to_L = MtoL_Bell2003(g_r)

        L_r = 10**((M_r_sun - Mr)/2.5)

        return M_to_L * L_r

    Mstar_bgs   = Mr_to_Mstar(M_r, g_r)
    Mstar_sdss  = Mr_to_Mstar(M_r_sdss, g_r_sdss) 
    #-------------------------------------------------------------------
    # BGS samples 
    #-------------------------------------------------------------------
    main_sample     = (m_r < 19.5)
    faint_sample    = (m_r < 20.)
    
    #-------------------------------------------------------------------
    # footprint comparison 
    #-------------------------------------------------------------------
    fig = plt.figure(figsize=(15,5))
    gs1 = mpl.gridspec.GridSpec(1,4, figure=fig) 
    sub = plt.subplot(gs1[0,:-1], projection='mollweide')
    sub.grid(True, linewidth=0.1) 
    # DESI footprint 
    sub.scatter(
            (ra_bgs[faint_sample] - 180.) * np.pi/180., 
            dec_bgs[faint_sample] * np.pi/180., s=1, lw=0, c='C0',
            rasterized=True)
    # SDSS footprint 
    sub.scatter((ra_sdss - 180.) * np.pi/180., dec_sdss * np.pi/180., 
            s=1, lw=0, c='C1', rasterized=True)
    # GAMA footprint for comparison 
    gama_ra_min = (np.array([30.2, 129., 174., 211.5, 339.]) - 180.) * np.pi/180.
    gama_ra_max = (np.array([38.8, 141., 186., 223.5, 351.]) - 180.) * np.pi/180. 
    gama_dec_min = np.array([-10.25, -2., -3., -2., -35.]) * np.pi/180.
    gama_dec_max = np.array([-3.72, 3., 2., 3., -30.]) * np.pi/180.
    for i_f, field in enumerate(['g02', 'g09', 'g12', 'g15', 'g23']): 
        rect = patches.Rectangle((gama_ra_min[i_f], gama_dec_min[i_f]), 
                gama_ra_max[i_f] - gama_ra_min[i_f], 
                gama_dec_max[i_f] - gama_dec_min[i_f], 
                facecolor='r')
        sub.add_patch(rect)
    sub.set_xlabel('RA', fontsize=20, labelpad=10) 
    #sub.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]) 
    sub.set_xticklabels(['', '', '$90^o$', '', '', '$180^o$', '', '', '$270^o$', '', ''])#, fontsize=10)
    sub.set_ylabel('Dec', fontsize=20)
    
    #-------------------------------------------------------------------
    # n(z) comparison 
    #-------------------------------------------------------------------
    sub = plt.subplot(gs1[0,-1])
    sub.hist(z_bgs, range=[0.0, 1.], color='C0', bins=100) 
    sub.hist(z_sdss, range=[0.0, 1.], color='C1', bins=100) 
    sub.hist(np.array(gama['Z']), range=[0.0, 1.], color='r', bins=100) 
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
    for clr in ['C0', 'C1', 'r']: 
        _plt = sub.fill_between([0], [0], [0], color=clr, linewidth=0)
        plts.append(_plt) 
    sub.legend(plts, ['DESI', 'SDSS', 'GAMA'], loc='upper right', handletextpad=0.3, prop={'size': 20}) 
    ffig = os.path.join(dir_doc, 'bgs.png')
    fig.savefig(ffig, bbox_inches='tight')
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    #-------------------------------------------------------------------
    # mstar(z) comparison 
    #-------------------------------------------------------------------
    fig = plt.figure(figsize=(10,5))
    sub = plt.subplot(111) 
    _faint = sub.scatter(z_bgs[faint_sample],
            np.log10(Mstar_bgs[faint_sample]), c='C0', s=1, rasterized=True)
    _main = sub.scatter(z_bgs[main_sample], np.log10(Mstar_bgs[main_sample]),
            c='C1', s=1, rasterized=True)
    #_sdss = sub.scatter(z_sdss, np.log10(Mstar_sdss), c='C1', s=1, rasterized=True)

    sub.legend([_main, _faint], ['BGS Bright', 'BGS Faint'], loc='lower right',
            fontsize=25, markerscale=10, handletextpad=0.)
    sub.set_xlabel('Redshift', fontsize=20)
    sub.set_xlim(0., 0.5)
    sub.set_ylabel('$\log M_*$ [$M_\odot$]', fontsize=20)
    sub.set_ylim(6, 12.5)
    sub.text(0.97, 0.4, 'MXXL BGS mock', ha='right', va='top', 
            transform=sub.transAxes, fontsize=25)
    ffig = os.path.join(dir_doc, 'bgs_mstar_z.png')
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


def speculator():
    '''plot the SFH and ZH bases of speculator 
    '''
    ispeculator = Fitters.iSpeculator()

    tage = np.linspace(0., 13.7, 50) 
    
    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(121)
    for i, basis in enumerate(ispeculator._sfh_basis): 
        sub.plot(tage, basis(tage), label=r'$s_{%i}^{\rm SFH}$' % (i+1)) 
    sub.set_xlim(0., 13.7) 
    sub.set_ylabel(r'star formation rate [$M_\odot/{\rm Gyr}$]', fontsize=20) 
    sub.set_ylim(0., 0.18) 
    sub.set_yticks([0.05, 0.1, 0.15]) 
    sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 

    sub = fig.add_subplot(122)
    for i, basis in enumerate(ispeculator._zh_basis): 
        sub.plot(tage, basis(tage), label=r'$s_{%i}^{\rm ZH}$' % (i+1)) 
    sub.set_xlim(0., 13.7) 
    sub.set_ylabel('metallicity $Z$', fontsize=20) 
    sub.set_ylim(0., None) 
    sub.legend(loc='upper left', fontsize=20, handletextpad=0.2) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$t_{\rm age}$ [Gyr]', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.2)
    _ffig = os.path.join(dir_doc, 'speculator.png') 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def mcmc_posterior(sim='lgal', obs='spec', noise='bgs', dt_sfr='100myr'): 
    ''' plot that includes a corner plot and also demonstrates how the best-fit
    reproduces the data 
    '''
    igal = 20 
    ########################################################################
    # read meta data 
    ########################################################################
    _, meta = Data.Photometry(sim=sim, noise='legacy', lib='bc03', sample='mini_mocha') 
    print('z=%f' % meta['redshift'][igal])
    ########################################################################
    # read markov chain 
    ########################################################################
    f_mcmc = os.path.join(dir_mm, 'provabgs',
            '%s.%s.noise_%s.%i.mcmc.hdf5' % (sim, obs, noise, igal))
    mcmc = h5py.File(f_mcmc, 'r')
    f_post = os.path.join(dir_mm, 'provabgs',
            '%s.%s.noise_%s.%i.postproc.hdf5' % (sim, obs, noise, igal))
    post = h5py.File(f_post, 'r')
    ########################################################################
    # compile labels and truths
    ########################################################################
    theta_names = ['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh', 'beta4_sfh',
            'gamma1_zh', 'gamma2_zh', 'dust1', 'dust2', 'dust_index']
    lbls = [lbl_dict[_theta] for _theta in theta_names]
    truths = [None for _ in theta_names]
    truths[0] = meta['logM_%s' % ['total', 'fiber'][obs=='spec']][igal]
    if obs == 'specphoto': 
        theta_names += ['f_fiber']
        truths[-1] = (photo['fiberflux_r_true'][igal]/photo['flux_r_true'][igal]) 
    ########################################################################
    # figure 
    ########################################################################
    # MCMC 
    chain = mcmc['mcmc_chain'][...]
    fig = DFM.corner(
            chain.reshape((np.prod(chain.shape[:2]), chain.shape[2])),
            range=np.array(mcmc['prior_range'][...]).T,
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95],
            nbin=40,
            smooth=True, 
            truths=truths, 
            labels=lbls, 
            label_kwargs={'fontsize': 15})
    _ffig = os.path.join(dir_doc, 'mcmc_posterior.%s.%s.%s.png' % (sim, obs, noise))
    fig.savefig(_ffig, bbox_inches='tight') 
    #fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    ########################################################################
    prop_names = ['logmstar', 'logsfr.%s' % dt_sfr, 'logz.mw']
    props = np.array([
        post['logmstar'][...],
        post['logsfr.%s' % dt_sfr][...] - post['logmstar'][...], 
        post['logz.mw'][...]]).T
    lbls = [lbl_dict['logmstar'], r'$\log~{\rm SSFR}_{%s}$' % dt_sfr, lbl_dict['logz.mw']]
    truths = [meta['logM_%s' % ['total', 'fiber'][obs=='spec']][igal], 
            np.log10(meta['sfr_%s' % dt_sfr][igal]) - meta['logM_%s' % ['total', 'fiber'][obs=='spec']][igal], 
            np.log10(meta['Z_MW'][igal])]
    plot_range = [(np.median(props[:,i]) - 0.5, np.median(props[:,i]) + 0.5) for i in range(len(truths))]

    fig = DFM.corner(
            props,
            #weights=post['w_prior.%s' % dt_sfr], 
            range=plot_range, 
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95],
            nbin=40,
            smooth=True, 
            truths=truths, 
            labels=lbls, 
            label_kwargs={'fontsize': 20})
    _ffig = os.path.join(dir_doc, 'mcmc_posterior.%s.%s.%s.props.sfr%s.png' %
            (sim, obs, noise, dt_sfr))
    fig.savefig(_ffig, bbox_inches='tight') 
    #fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    
    if obs == 'specphoto': 
        fig = plt.figure(figsize=(20,5))
        gs2 = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1,3])#, top=0.17, bottom=0.05)
        ax1 = fig.add_subplot(gs2[0,0]) 
        ax2 = fig.add_subplot(gs2[0,1]) 
    elif obs == 'spec': 
        fig = plt.figure(figsize=(15,5))
        ax2 = fig.add_subplot(111)
    elif obs == 'photo': 
        fig = plt.figure(figsize=(5,5))
        ax1 = fig.add_subplot(111)

    if 'photo' in obs: 
        n_photo = len(mcmc['flux_photo_obs'][...])
        ax1.errorbar(np.arange(n_photo), mcmc['flux_photo_obs'][...], 
                yerr=mcmc['flux_ivar_photo_obs'][...]**-0.5, fmt='.k',
                label='observations')
        ax1.scatter(np.arange(n_photo), mcmc['flux_photo_model'][...], 
                c='C1', label=r'best-fit') 
        ax1.legend(loc='upper left', markerscale=3, handletextpad=0.2, fontsize=15) 
        ax1.set_xticks([0, 1, 2])#, 3, 4]) 
        ax1.set_xticklabels(['$g$', '$r$', '$z$'], fontsize=25) 
        ax1.set_xlim(-0.5, n_photo-0.5)
    if 'spec' in obs: 
        ax2.plot(mcmc['wavelength_obs'][...], mcmc['flux_spec_obs'][...], c='k',
                lw=1) 
        ax2.plot(mcmc['wavelength_obs'][...], mcmc['flux_spec_model'][...], c='C1',
                ls='--', lw=1) 
        ax2.set_xlabel('wavelength [$\AA$]', fontsize=20) 
        ax2.set_xlim(3600., 9800.)
        ax2.set_ylim(-1., 5.) 
        
    _ffig = os.path.join(dir_doc, 'mcmc_posterior.%s.%s.%s.bestfit.png' % (sim, obs, noise))
    fig.savefig(_ffig, bbox_inches='tight') 
    #fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def inferred_props(sim='lgal', obs='photo', noise='legacy', dt_sfr='100myr'):
    ''' compare inferred physical galaxy properties of each galaxy to its
    corresponding intrinsic values 
    '''
    ngal = 96
    
    # read noiseless Lgal spectra of the spectral_challenge mocks
    _, meta = Data.Spectra(sim=sim, noise='bgs', lib='bc03', sample='mini_mocha') 
    
    igals, theta_inf = [], [] 
    for igal in range(ngal):  
        _fbf = os.path.join(UT.dat_dir(), 'mini_mocha', 'provabgs', 
                '%s.%s.noise_%s.%i.postproc.hdf5' % (sim, obs, noise, igal))

        if not os.path.isfile(_fbf): 
            print('     %s does not exist' % _fbf)
            continue 
        fbf = h5py.File(_fbf, 'r')  

        chain_logm      = fbf['logmstar'][...]
        chain_logsfr    = fbf['logsfr.%s' % dt_sfr][...]
        chain_logzmw    = fbf['logz.mw'][...]
        try: 
            chain_w         = fbf['w_prior.%s' % dt_sfr][...]
            print(_fbf)
        except KeyError: 
            pass

        fbf.close() 

        igals.append(igal)
        theta_inf.append(np.array([ 
            np.percentile(chain_logm, [2.5, 16, 50, 84, 97.5]), 
            np.percentile(chain_logsfr, [2.5, 16, 50, 84, 97.5]), 
            np.percentile(chain_logzmw, [2.5, 16, 50, 84, 97.5])
            ]))
    
    # input Mstar, logSSFR 100 Myr/1 Gyr, log Z_MW 
    logm_true   = np.array([meta['logM_%s' % ['total', 'fiber'][obs == 'spec']][i] for i in igals])
    logm_infer  = np.array(theta_inf)[:,0,:]

    logSFR_true     = np.log10([meta['sfr_%s' % dt_sfr][i] for i in igals])
    logSFR_infer    = np.array(theta_inf)[:,1,:]

    logZ_MW_true    = np.log10([meta['Z_MW'][i] for i in igals])
    logZ_MW_infer   = np.array(theta_inf)[:,2,:]
    
    fig = plt.figure(figsize=(16,4))
    # compare total stellar mass 
    sub = fig.add_subplot(1,3,1) 
    sub.errorbar(
            logm_true,
            logm_infer[:,2], 
            yerr=[logm_infer[:,2] - logm_infer[:,1], logm_infer[:,3]-logm_infer[:,2]],
            fmt='.C0')
    sub.plot([9., 12.], [9., 12.], c='k', ls='--') 
    sub.text(0.05, 0.95, sim.upper(), ha='left', va='top', transform=sub.transAxes, fontsize=25)

    sub.set_xlim(9., 12.) 
    sub.set_ylim(9., 12.) 
    sub.set_yticks([9., 10., 11., 12.]) 
    sub.set_title(r'$\log M_*$', fontsize=25)

    # compare SFR 
    sub = fig.add_subplot(1,3,2) 
    sub.errorbar(
            logSFR_true, 
            logSFR_infer[:,2], 
            yerr=[logSFR_infer[:,2]-logSFR_infer[:,1],
                logSFR_infer[:,3]-logSFR_infer[:,2]], 
            fmt='.C0')
    sub.plot([-3., 1.], [-3., 1.], c='k', ls='--') 
    sub.set_xlim(-3., 1.) 
    sub.set_ylim(-3., 1.) 
    if dt_sfr == '100myr': 
        sub.set_title(r'$\log {\rm SFR}$ 100 Myr', fontsize=25)
    elif dt_sfr == '1gyr': 
        sub.set_title(r'$\log {\rm SFR}$ 1 Gyr', fontsize=25)
    
    # compare Z 
    sub = fig.add_subplot(1,3,3) 
    sub.errorbar(
            logZ_MW_true, 
            logZ_MW_infer[:,2], 
            yerr=[logZ_MW_infer[:,2]-logZ_MW_infer[:,1],
                logZ_MW_infer[:,3]-logZ_MW_infer[:,2]], 
            fmt='.C0')
    sub.plot([-3., -1.], [-3., -1.], c='k', ls='--') 
    sub.set_xlim(-3., -1.) 
    sub.set_xticks([-3., -2., -1.]) 
    sub.set_ylim(-3., -1.) 
    sub.set_yticks([-3., -2., -1.]) 
    sub.set_title(r'$\log$ MW $Z$', fontsize=25)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$\theta_{\rm true}$', fontsize=25) 
    bkgd.set_ylabel(r'$\widehat{\theta}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.225, hspace=0.1)
    _ffig = os.path.join(dir_doc, 'inferred_prop.%s.%s.%s.%s.comparison.png' % (sim, obs, noise, dt_sfr)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    #fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def _speculator_fsps(sfr='100myr'):
    ''' compare inferred physical galaxy properties from speculator versus
    actually running fsps 

    notes
    -----
    * only Lgal implemented
    '''
    if sfr == '100myr': dt = 0.1
    ngal = 97
    
    Mstar_emul, logSFR_emul = [], [] 
    Mstar_fsps, logSFR_fsps = [], [] 

    igals = []
    for igal in range(ngal):  
        _femul = Fbestfit_specphoto(igal, sim='lgal', noise='bgs0_legacy',
                method='ispeculator', model='emulator')
        _ffsps = Fbestfit_specphoto(igal, sim='lgal', noise='bgs0_legacy',
                method='ispeculator', model='fsps')
        if not (os.path.isfile(_femul) and os.path.isfile(_ffsps)):
            print('     %s or' % _femul)
            print('     %s does not exist' % _ffsps)
            continue 
        igals.append(igal) 

        femul       = h5py.File(_femul, 'r')  
        mcmc_emul   = femul['mcmc_chain'][...].copy() 
        _theta_emul = np.percentile(mcmc_emul, [2.5, 16, 50, 84, 97.5], axis=0)
        femul.close() 
        
        ffsps       = h5py.File(_ffsps, 'r')  
        mcmc_fsps   = ffsps['mcmc_chain'][...].copy() 
        _theta_fsps = np.percentile(mcmc_fsps, [2.5, 16, 50, 84, 97.5], axis=0)
        ffsps.close() 
    
        Mstar_emul.append(_theta_emul[:,0])
        logSFR_emul.append(_theta_emul[:,-1])
        Mstar_fsps.append(_theta_fsps[:,0])
        logSFR_fsps.append(_theta_fsps[:,-1])

    igals = np.array(igals)
    Mstar_emul = np.array(Mstar_emul)
    logSFR_emul = np.array(logSFR_emul)
    Mstar_fsps = np.array(Mstar_fsps)
    logSFR_fsps = np.array(logSFR_fsps)
    
    fig = plt.figure(figsize=(8,4))
    # compare total stellar mass 
    sub = fig.add_subplot(121) 
    #sub.errorbar(Mstar_input, Mstar_inf[:,2], 
    #        yerr=[Mstar_inf[:,2]-Mstar_inf[:,1], Mstar_inf[:,3]-Mstar_inf[:,2]], fmt='.C0')
    sub.scatter(Mstar_fsps[:,2], Mstar_emul[:,2], c='C0') 
    sub.plot([9., 12.], [9., 12.], c='k', ls='--') 
    sub.text(0.05, 0.95, 'L-Galaxies', ha='left', va='top', transform=sub.transAxes, fontsize=25)

    sub.set_xlim(9., 12.) 
    sub.set_ylim(9., 12.) 
    sub.set_yticks([9., 10., 11., 12.]) 
    sub.set_title(r'$\log M_*$', fontsize=25)

    # compare SFR 
    sub = fig.add_subplot(122) 
    #sub.errorbar(logSFR_input, logSFR_inf[:,2], 
    #        yerr=[logSFR_inf[:,2]-logSFR_inf[:,1], logSFR_inf[:,3]-logSFR_inf[:,2]], fmt='.C0')
    sub.scatter(logSFR_fsps[:,2], logSFR_emul[:,2], c='C0')
    sub.plot([-3., 2.], [-3., 2.], c='k', ls='--') 
    sub.set_xlim(-3., 2.) 
    sub.set_ylim(-3., 2.) 
    sub.set_yticks([-2., 0., 2.]) 
    if sfr == '1gyr': lbl_sfr = '1Gyr'
    elif sfr == '100myr': lbl_sfr = r'100{\rm Myr}'
    sub.set_title(r'$\log{\rm SFR}_{%s}$' % lbl_sfr, fontsize=25)
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$\theta_{\rm true}$', fontsize=25) 
    bkgd.set_ylabel(r'$\widehat{\theta}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.225, hspace=0.1)
    _ffig = os.path.join(dir_doc, '_speculator_fsps.prop_comparison.png') 
    fig.savefig(_ffig, bbox_inches='tight') 

    
    igal = np.random.choice(igals) 
    _femul = Fbestfit_specphoto(igal, sim='lgal', noise='bgs0_legacy',
            method='ispeculator', model='emulator')
    _ffsps = Fbestfit_specphoto(igal, sim='lgal', noise='bgs0_legacy',
            method='ispeculator', model='fsps')
    
    femul       = h5py.File(_femul, 'r')  
    mcmc_emul   = femul['mcmc_chain'][...].copy() 
    _theta_emul = np.percentile(mcmc_emul, [2.5, 16, 50, 84, 97.5], axis=0)
    femul.close() 
    
    ffsps       = h5py.File(_ffsps, 'r')  
    mcmc_fsps   = ffsps['mcmc_chain'][...].copy() 
    _theta_fsps = np.percentile(mcmc_fsps, [2.5, 16, 50, 84, 97.5], axis=0)
    ffsps.close() 
    
    fig = DFM.corner(mcmc_emul, quantiles=[0.16, 0.5, 0.84], levels=[0.68, 0.95], 
            color='C0', nbin=20, smooth=True) 
    DFM.corner(mcmc_fsps, quantiles=[0.16, 0.5, 0.84], levels=[0.68, 0.95],
            color='C1', nbin=20, smooth=True, fig=fig) 
    _ffig = os.path.join(dir_doc, '_speculator_fsps.prosterior_comparison.png') 
    fig.savefig(_ffig, bbox_inches='tight') 
    return None 


def photo_vs_specphoto(sim='lgal', noise_photo='legacy', noise_specphoto='bgs0_legacy',
        method='ifsps', model='vanilla', sample='mini_mocha'):  
    ''' Compare properties inferred from photometry versus spectrophotometry to see how much
    information is gained from adding spectra
    '''
    # read noiseless Lgal spectra of the spectral_challenge mocks
    _, meta = Data.Spectra(sim=sim, noise=noise_photo, lib='bc03', sample='mini_mocha') 
    logMstar_sim  = np.array(meta['logM_total']) # total mass 
    logSFR_sim    = np.log10(np.array(meta['sfr_100myr'])) 
    if sim == 'tng': 
        print('correcting for possible h?') 
        logMstar_sim += np.log10(1./0.6774**2)
        logSFR_sim += np.log10(1./0.6774**2)
    # --------------------------------------------------------------------------------
    # compile MC chains 
    _igal0, _dlogMs_photo = _read_dprop_mcmc('logmstar', obs='photo', sim=sim, 
            noise=noise_photo, method=method, model=model, sample=sample)
    _igal1, _dlogMs_specphoto = _read_dprop_mcmc('logmstar', obs='specphoto', sim=sim, 
            noise=noise_specphoto, method=method, model=model, sample=sample)
    igal_m, m0, m1 = np.intersect1d(_igal0, _igal1, return_indices=True) 
    logm_input      = logMstar_sim[igal_m]
    dlogm_photo     = _dlogMs_photo[m0]
    dlogm_specphoto = _dlogMs_specphoto[m1]
    
    _igal0, _dlogSFRs_photo = _read_dprop_mcmc('logsfr.100myr', sim=sim,
            obs='photo', noise=noise_photo, method=method, model=model,
            sample=sample) 
    _igal1, _dlogSFRs_specphoto = _read_dprop_mcmc('logsfr.100myr', sim=sim, 
            obs='specphoto', noise=noise_specphoto, method=method, model=model,
            sample=sample) 
    igal_s, m0, m1 = np.intersect1d(_igal0, _igal1, return_indices=True) 
    logsfr_input        = logSFR_sim[igal_s]
    dlogsfr_photo       = _dlogSFRs_photo[m0]
    dlogsfr_specphoto   = _dlogSFRs_specphoto[m1]
    # --------------------------------------------------------------------------------
    # maximum likelihood for the population hyperparameters
    bins_logm = np.linspace(9., 12., 7) 
    logm_mid, eta_logm_photo   = _get_etaMAP(logm_input, dlogm_photo, bins_logm)
    logm_mid, eta_logm_specphoto  = _get_etaMAP(logm_input, dlogm_specphoto, bins_logm)
    
    bins_logsfr = np.linspace(-3., 3., 7) 
    logsfr_mid, eta_logsfr_photo   = _get_etaMAP(logsfr_input, dlogsfr_photo, bins_logsfr)
    logsfr_mid, eta_logsfr_specphoto  = _get_etaMAP(logsfr_input, dlogsfr_specphoto, bins_logsfr)
    # --------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12,5))
    # compare total stellar mass 
    sub = fig.add_subplot(121) 
    sub.plot([9., 12.], [0., 0.], c='k', ls='--')
    sub.fill_between(logm_mid, 
            eta_logm_photo[:,0] - eta_logm_photo[:,1],
            eta_logm_photo[:,0] + eta_logm_photo[:,1], 
            fc='C0', ec='none', alpha=0.5, label='Photometry only') 
    sub.scatter(logm_mid, eta_logm_photo[:,0], c='C0', s=2) 
    sub.plot(logm_mid, eta_logm_photo[:,0], c='C0')
    sub.fill_between(logm_mid, 
            eta_logm_specphoto[:,0] - eta_logm_specphoto[:,1],
            eta_logm_specphoto[:,0] + eta_logm_specphoto[:,1], 
            fc='C1', ec='none', alpha=0.5, label='Photometry+Spectroscopy') 
    sub.scatter(logm_mid, eta_logm_specphoto[:,0], c='C1', s=1) 
    sub.plot(logm_mid, eta_logm_specphoto[:,0], c='C1') 
    sub.set_xlabel(r'$\log(~M_*~[M_\odot]~)$', fontsize=25)
    sub.set_xlim(9., 12.) 
    sub.set_ylabel(r'$\Delta_{\log M_*}$', fontsize=25)
    sub.set_ylim(-1., 1.) 
    sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    
    # compare SFR 
    sub = fig.add_subplot(122) 
    sub.plot([-3., 3.], [0., 0.], c='k', ls='--')
    sub.fill_between(logsfr_mid,  
            eta_logsfr_photo[:,0] - eta_logsfr_photo[:,1],
            eta_logsfr_photo[:,0] + eta_logsfr_photo[:,1], 
            fc='C0', ec='none', alpha=0.5) 
    sub.scatter(logsfr_mid, eta_logsfr_photo[:,0], c='C0', s=2) 
    sub.plot(logsfr_mid, eta_logsfr_photo[:,0], c='C0') 
    sub.fill_between(logsfr_mid,
            eta_logsfr_specphoto[:,0] - eta_logsfr_specphoto[:,1],
            eta_logsfr_specphoto[:,0] + eta_logsfr_specphoto[:,1], 
            fc='C1', ec='none', alpha=0.5) 
    sub.scatter(logsfr_mid, eta_logsfr_specphoto[:,0], c='C1', s=1) 
    sub.plot(logsfr_mid, eta_logsfr_specphoto[:,0], c='C1') 
    
    sub.set_xlabel(r'$\log(~{\rm SFR}_{100Myr}~[M_\odot/yr]~)$', fontsize=25)
    sub.set_xlim(-3., 1.) 
    sub.set_ylabel(r'$\Delta_{\log{\rm SFR}_{100Myr}}$', fontsize=25)
    sub.set_ylim(-3., 3.) 
    
    fig.subplots_adjust(wspace=0.3)
    _ffig = os.path.join(dir_doc,
            'photo_vs_specphoto.%s.%s.%s.noise_%s_%s.png' % 
            (sim, method, model, noise_photo, noise_specphoto)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def eta_Delta(sim='lgal', noise='legacy', obs='photo', dt_sfr='100myr', sample='mini_mocha'):  
    ''' population hyperparameters eta_Delta as a function of different
    galaxy properties (r mag, color, etc) 
    '''
    # --------------------------------------------------------------------------------
    # read Lgal photometry of the mini_mocha mocks
    photo, meta = Data.Photometry(sim=sim, noise='legacy', lib='bc03', sample='mini_mocha')

    r_mag   = 22.5 - 2.5 * np.log10(photo['flux_r_true']) # true r-band magnitude
    g_r     = (22.5 - 2.5 * np.log10(photo['flux_g_true'])) - r_mag 
    r_z     = r_mag - (22.5 - 2.5 * np.log10(photo['flux_z_true']))
    # --------------------------------------------------------------------------------
    # assemble all markov chains 
    # --------------------------------------------------------------------------------
    igal_m, dlogm_chain = _read_dprop_mcmc('logmstar', sim=sim, obs=obs,
            noise=noise, sample=sample)
    igal_s, dlogsfr_chain = _read_dprop_mcmc('logsfr.%s' % dt_sfr, sim=sim,
            obs=obs, noise=noise, sample=sample)
    # --------------------------------------------------------------------------------
    # get MAP for the population hyperparameters
    # --------------------------------------------------------------------------------
    props       = [r_mag, g_r, r_z]
    prop_lbl    = ['$r$ magnitude', '$g - r$ color', '$r - z$ color']
    prop_lim    = [(19, 20), (0., 2.), (0., 1.)]
    
    hyper_logm, hyper_logsfr = [], [] 
    for i, prop in enumerate(props) : 
        bins_prop = np.linspace(prop_lim[i][0], prop_lim[i][1], 11) 

        mid_prop_m, eta_logm    = _get_etaMAP(prop[igal_m], dlogm_chain, bins_prop)
        mid_prop_s, eta_logsfr  = _get_etaMAP(prop[igal_s], dlogsfr_chain, bins_prop)

        hyper_logm.append([mid_prop_m, eta_logm[:,0], eta_logm[:,1]]) 
        hyper_logsfr.append([mid_prop_s, eta_logsfr[:,0], eta_logsfr[:,1]]) 
    # --------------------------------------------------------------------------------
    # plot the MAP hyperparameters
    # --------------------------------------------------------------------------------
    fig = plt.figure(figsize=(4*len(props),8))
    for i in range(len(props)): 

        prop_mid_m, mu_dmstar, sig_dmstar = hyper_logm[i]
        prop_mid_s, mu_dsfr, sig_dsfr = hyper_logsfr[i]

        # M* hyperparametrs
        sub = fig.add_subplot(2,len(props),i+1) 

        sub.plot([prop_lim[i][0], prop_lim[i][1]], [0., 0.], c='k', ls='--')
        sub.fill_between(prop_mid_m, mu_dmstar - sig_dmstar, mu_dmstar + sig_dmstar, 
                fc='C0', ec='none', alpha=0.5) 
        sub.scatter(prop_mid_m, mu_dmstar, c='C0', s=2) 
        sub.plot(prop_mid_m, mu_dmstar, c='C0') 

        sub.set_xlim(prop_lim[i][0], prop_lim[i][1]) 
        if i == 0: sub.set_ylabel(r'$\Delta_{\log M_*}$', fontsize=25)
        sub.set_ylim(-1., 1.) 
        #sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    
        # SFR hyperparametrs
        sub = fig.add_subplot(2,len(props),len(props)+i+1) 
        sub.plot([prop_lim[i][0], prop_lim[i][1]], [0., 0.], c='k', ls='--')
        sub.fill_between(prop_mid_s, mu_dsfr - sig_dsfr, mu_dsfr + sig_dsfr, 
                fc='C0', ec='none', alpha=0.5) 
        sub.scatter(prop_mid_s, mu_dsfr, c='C0', s=2) 
        sub.plot(prop_mid_s, mu_dsfr, c='C0') 
        
        sub.set_xlabel(prop_lbl[i], fontsize=25)
        sub.set_xlim(prop_lim[i][0], prop_lim[i][1]) 
        if i == 0: sub.set_ylabel(r'$\Delta_{\log{\rm SFR}_{100Myr}}$', fontsize=25)
        sub.set_ylim(-3., 3.) 
    
    fig.subplots_adjust(wspace=0.3)
    _ffig = os.path.join(dir_doc, 'eta_Delta.%s.%s.noise_%s.sfr%s.png' % (sim,
        obs, noise, dt_sfr)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    #fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
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
    
        if np.any(np.isnan(delta_chains[i])): 
            p_Di_pop = 0.
        else: 
            p_Di_pop = np.sum(gauss.pdf(delta_chains[i])/prior(delta_chains[i]))/float(K)

        # clip likelihood at 1e-8 
        logp_D_pop += np.log(np.clip(p_Di_pop, 1e-8, None)) 

    if np.isnan(logp_D_pop): 
        for i in range(N): 
            K = len(delta_chains[i]) 
            print(i)
            gauss = Norm(loc=mu_pop, scale=sigma_pop) 
            print('mu:', mu_pop, 'sigma:', sigma_pop) 
            print(delta_chains[i]) 
            p_Di_pop = np.sum(gauss.pdf(delta_chains[i])/prior(delta_chains[i]))/float(K)
            print(p_Di_pop) 
        raise ValueError
    return logp_D_pop     


def _get_etaMAP(prop, chain, bins): 
    ''' get MAP hyperparameters
    '''
    assert len(prop) == chain.shape[0]

    # optimization kwargs
    opt_kwargs = {'method': 'L-BFGS-B', 'bounds': ((None, None), (1e-4, None))}
    #, options={'eps': np.array([0.01, 0.005]), 'maxiter': 100})

    bin_mid, eta = [], [] 
    for i in range(len(bins)-1): 
        inbin = (prop > bins[i]) & (prop <= bins[i+1]) 
        if np.sum(inbin) == 0: continue 
       
        bin_mid.append(0.5 * (bins[i] + bins[i+1])) 

        L_pop = lambda _theta: -1.*logL_pop(_theta[0], _theta[1],
                delta_chains=chain[inbin])  

        _min = op.minimize(L_pop, np.array([0., 0.1]), **opt_kwargs) 

        eta.append([_min['x'][0], _min['x'][1]]) 

    return np.array(bin_mid), np.array(eta)


def _read_dprop_mcmc(prop, sim='lgal', obs='specphoto', noise='bgs0_legacy', sample='mini_mocha'): 
    ''' read prop_inf - prop_input for MC chains 
    '''
    # get true galaxy properties  
    _, meta = Data.Spectra(sim=sim, noise='bgs0', lib='bc03', sample=sample) 
    if prop == 'logmstar': 
        if obs == 'spec': 
            prop_sim = np.array(meta['logM_fiber']) # fiber mass 
        else: 
            prop_sim = np.array(meta['logM_total']) # total mass 
    elif prop == 'logsfr.100myr': 
        prop_sim = np.log10(np.array(meta['sfr_100myr'])) 
    elif prop == 'logsfr.1gyr': 
        prop_sim = np.log10(np.array(meta['sfr_1gyr'])) 

    # --------------------------------------------------------------------------------
    # assemble all markov chains 
    igals, dprop = [], [] 
    for igal in range(97): 
        # read best-fit file and get inferred parameters from photometry
        f_mcmc = file_name_mcmc_postproc(igal, sim=sim, noise=noise, obs=obs)
        if not os.path.isfile(f_mcmc): continue 

        mcmc    = h5py.File(f_mcmc, 'r')  
        dprop_i = mcmc[prop]- prop_sim[igal]
        
        assert np.all(~np.isnan(dprop_i))

        dprop.append(dprop_i)
        igals.append(igal) 
    
    return np.array(igals), np.array(dprop)


def file_name_mcmc_postproc(igal, sim='lgal', noise='none', obs='spec'): 
    ''' name of the file that contains the postprocessed markov chain of galaxy #igal 

    :param igal: 
        index of spectral_challenge galaxy 
    '''
    return os.path.join(UT.dat_dir(), 'mini_mocha', 'provabgs',
            '%s.%s.noise_%s.%i.postproc.hdf5' % (sim, obs, noise, igal))


def dust_comparison(sim='lgal', noise='bgs0_legacy', method='ifsps'): 
    ''' Compare properties inferred from photometry versus spectrophotometry to see how much
    information is gained from adding spectra
    '''
    # read noiseless Lgal spectra of the spectral_challenge mocks
    _, meta = Data.Spectra(sim=sim, noise=noise.split('_')[0], lib='bc03', sample='mini_mocha') 
    logMstar_sim  = np.array(meta['logM_total']) # total mass 
    logSFR_sim    = np.log10(np.array(meta['sfr_100myr'])) 
    # --------------------------------------------------------------------------------
    # compile MC chains 
    _igal0, _dlogMs_simple = _read_dprop_mcmc('logmstar', obs='specphoto', noise=noise,
            method='ifsps', model='vanilla', sample='mini_mocha')
    _igal1, _dlogMs_complex = _read_dprop_mcmc('logmstar', obs='specphoto', noise=noise,
            method='ifsps', model='vanilla_complexdust', sample='mini_mocha')
    igal_m, m0, m1 = np.intersect1d(_igal0, _igal1, return_indices=True) 
    logm_input      = logMstar_sim[igal_m]
    dlogm_simple    = _dlogMs_simple[m0]
    dlogm_complex   = _dlogMs_complex[m1]
    
    _igal0, _dlogSFRs_simple = _read_dprop_mcmc('logsfr.100myr', sim=sim,
            obs='specphoto', noise=noise, method=method, model='vanilla',
            sample='mini_mocha') 
    _igal1, _dlogSFRs_complex = _read_dprop_mcmc('logsfr.100myr', sim=sim, 
            obs='specphoto', noise=noise, method=method,
            model='vanilla_complexdust', sample='mini_mocha')
    igal_s, m0, m1 = np.intersect1d(_igal0, _igal1, return_indices=True) 
    logsfr_input    = logSFR_sim[igal_s]
    dlogsfr_simple  = _dlogSFRs_simple[m0]
    dlogsfr_complex = _dlogSFRs_complex[m1]
    # --------------------------------------------------------------------------------
    # maximum likelihood for the population hyperparameters
    bins_logm = np.linspace(9., 12., 7) 
    logm_mid, eta_logm_simple   = _get_etaMAP(logm_input, dlogm_simple, bins_logm)
    logm_mid, eta_logm_complex  = _get_etaMAP(logm_input, dlogm_complex, bins_logm)
    
    bins_logsfr = np.linspace(-3., 3., 7) 
    logsfr_mid, eta_logsfr_simple   = _get_etaMAP(logsfr_input, dlogsfr_simple, bins_logsfr)
    logsfr_mid, eta_logsfr_complex  = _get_etaMAP(logsfr_input, dlogsfr_complex, bins_logsfr)

    # --------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12,5))
    # compare log M*
    sub = fig.add_subplot(121) 
    sub.plot([9., 12.], [0., 0.], c='k', ls='--')
    sub.fill_between(logm_mid, eta_logm_simple[:,0] - eta_logm_simple[:,1], eta_logm_simple[:,0] + eta_logm_simple[:,1],
            fc='C0', ec='none', alpha=0.5, label='Simple Dust') 
    sub.scatter(logm_mid, eta_logm_simple[:,0], c='C0', s=2) 
    sub.plot(logm_mid, eta_logm_simple[:,0], c='C0')
    sub.fill_between(logm_mid, eta_logm_complex[:,0] - eta_logm_complex[:,1], eta_logm_complex[:,0] + eta_logm_complex[:,1], 
            fc='C1', ec='none', alpha=0.5, label='Complex Dust') 
    sub.scatter(logm_mid, eta_logm_complex[:,0], c='C1', s=1) 
    sub.plot(logm_mid, eta_logm_complex[:,0], c='C1') 
    sub.set_xlabel(r'$\log(~M_*~[M_\odot]~)$', fontsize=25)
    sub.set_xlim(9., 12.) 
    sub.set_ylabel(r'$\Delta_{\log M_*}$', fontsize=25)
    sub.set_ylim(-1., 1.) 
    sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    
    # compare log SFR 
    sub = fig.add_subplot(122) 
    sub.plot([-3., 3.], [0., 0.], c='k', ls='--')
    sub.fill_between(logsfr_mid, 
            eta_logsfr_simple[:,0] - eta_logsfr_simple[:,1], 
            eta_logsfr_simple[:,0] + eta_logsfr_simple[:,1],
            fc='C0', ec='none', alpha=0.5) 
    sub.scatter(logsfr_mid, eta_logsfr_simple[:,0], c='C0', s=2) 
    sub.plot(logsfr_mid, eta_logsfr_simple[:,0], c='C0') 
    sub.fill_between(logsfr_mid, 
            eta_logsfr_complex[:,0] - eta_logsfr_complex[:,1], 
            eta_logsfr_complex[:,0] + eta_logsfr_complex[:,1],
            fc='C1', ec='none', alpha=0.5) 
    sub.scatter(logsfr_mid, eta_logsfr_complex[:,0], c='C1', s=2) 
    sub.plot(logsfr_mid, eta_logsfr_complex[:,0], c='C1') 

    sub.set_xlabel(r'$\log(~{\rm SFR}_{100Myr}~[M_\odot/yr]~)$', fontsize=25)
    sub.set_xlim(-3., 2.) 
    sub.set_ylabel(r'$\Delta_{\log{\rm SFR}_{100Myr}}$', fontsize=25)
    sub.set_ylim(-3., 3.) 
    
    fig.subplots_adjust(wspace=0.3)
    _ffig = os.path.join(dir_doc,
            'dust_comparison.%s.%s.noise_%s.png' % (sim, method, noise)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def _NMF_bases(): 
    ''' There's some confusion around the NMF SFH bases. The basis read
    directly from the file does not correspond to the ordering in Justin's
    paper. There's also discrepancies in the normalization. 
    '''
    fsfh = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'gqp_mc', 'dat', 'NMF_2basis_SFH_components_nowgt_lin_Nc4.txt')
    ft = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'gqp_mc', 'dat', 'sfh_t_int.txt') 

    nmf_sfh = np.loadtxt(fsfh) 
    nmf_t   = np.loadtxt(ft) # look back time 
    
    ispec0 = Fitters.iSpeculator()
    tt0 = np.array([0, 1, 0, 0, 0, 0.1, 0.1, 0.]) 
    w_emu, flux_emu = ispec0.model(tt0, zred=0.0, dont_transform=True)

    ispec1 = Fitters.iSpeculator(model_name='fsps')
    #tt1 = tt0.copy() 
    tt1 = np.array([0, 1, 0, 0, 0, 0.1, 0.1, 0.]) 
    w_fsps, flux_fsps = ispec1.model(tt1, zred=0.0, dont_transform=True)
    
    tt_sfh = tt1[1:5]
    tage = ispec0._tage_z_interp(0.0)
    _t = np.linspace(0, tage, 50)
    tages   = max(_t) - _t + 1e-8 
    sfh = np.sum(np.array([
        tt_sfh[i] *
        ispec1._sfh_basis[i](_t)/np.trapz(ispec1._sfh_basis[i](_t), _t) 
        for i in range(4)]), 
        axis=0)
    sfh /= np.sum(sfh)
    
    fig = plt.figure(figsize=(6,4))
    sub = fig.add_subplot(111)
    for i, basis in enumerate(nmf_sfh): 
        sub.plot(nmf_t, basis, label=r'$s_{%i}^{\rm SFH}$' % (i+1)) 
    sub.plot(tages, sfh, c='k', ls='--') 
    sub.set_xlim(0., 13.7) 
    sub.set_ylabel(r'star formation rate [$M_\odot/{\rm Gyr}$]', fontsize=20) 
    sub.set_ylim(0., 0.18) 
    sub.set_yticks([0.05, 0.1, 0.15]) 
    sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'lookback time [Gyr]', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.2)
    _ffig = os.path.join(dir_doc, '_NMF_bases_check.png') 
    fig.savefig(_ffig, bbox_inches='tight') 


    fig = plt.figure(figsize=(10,4))
    sub = fig.add_subplot(111)
    sub.plot(w_emu, flux_emu)
    sub.plot(w_fsps, flux_fsps, ls=':')
    sub.set_xlim(3000., 10000) 
    #sub.set_ylim(0., 0.18) 
    sub.set_yscale('log') 
    _ffig = os.path.join(dir_doc, '_emu_vs_fsps.png') 
    fig.savefig(_ffig, bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #BGS()

    #FM_photo()
    #FM_spec()
    
    #speculator()
    #_NMF_bases() 
    
    mcmc_posterior(sim='fsps', obs='spec', noise='bgs', dt_sfr='100myr')
    mcmc_posterior(sim='fsps', obs='spec', noise='bgs', dt_sfr='1gyr')
    
    #inferred_props(sim='fsps', obs='spec', noise='none', dt_sfr='100myr')
    #inferred_props(sim='fsps', obs='spec', noise='none', dt_sfr='1gyr')
    
    #inferred_props(sim='fsps', obs='spec', noise='bgs', dt_sfr='100myr')
    #inferred_props(sim='fsps', obs='spec', noise='bgs', dt_sfr='1gyr')
    
    #inferred_props(obs='spec', noise='bgs0', dt_sfr='100myr')
    #inferred_props(obs='spec', noise='bgs0', dt_sfr='1gyr')

    #inferred_props(obs='photo', noise='legacy', dt_sfr='100myr')
    #inferred_props(obs='photo', noise='legacy', dt_sfr='1gyr')
    
    #inferred_props(obs='specphoto', noise='bgs0_legacy', dt_sfr='100myr')
    #inferred_props(obs='specphoto', noise='bgs0_legacy', dt_sfr='1gyr')

    #_speculator_fsps(sfr='100myr')

    #photo_vs_specphoto(sim='lgal', noise_photo='legacy', noise_specphoto='bgs0_legacy', 
    #        method='ifsps', model='vanilla')
    #photo_vs_specphoto(sim='tng', noise_photo='legacy', noise_specphoto='bgs0_legacy', 
    #        method='ifsps', model='vanilla')

    #photo_vs_specphoto(sim='lgal', noise_photo='legacy', noise_specphoto='bgs0_legacy', 
    #        method='ispeculator', model='emulator')
    #photo_vs_specphoto(sim='tng', noise_photo='legacy', noise_specphoto='bgs0_legacy', 
    #        method='ispeculator', model='emulator')

    #eta_Delta(sim='fsps', obs='spec', noise='none', dt_sfr='100myr')
    #eta_Delta(sim='fsps', obs='spec', noise='none', dt_sfr='1gyr')
    #eta_Delta(sim='lgal', obs='spec', noise='bgs0', dt_sfr='100myr')
    #eta_Delta(sim='lgal', obs='spec', noise='bgs0', dt_sfr='1gyr')

    #eta_Delta(sim='lgal', obs='photo', noise='legacy', dt_sfr='100myr')
    #eta_Delta(sim='lgal', obs='photo', noise='legacy', dt_sfr='1gyr')

    #eta_Delta(sim='lgal', obs='specphoto', noise='bgs0_legacy', dt_sfr='100myr')
    #eta_Delta(sim='lgal', obs='specphoto', noise='bgs0_legacy', dt_sfr='1gyr')

    #dust_comparison(sim='lgal', noise='bgs0_legacy')


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
    
