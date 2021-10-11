'''

generate plots for the mock challenge paper 


'''
import os 
import h5py 
import pickle 
import numpy as np 
import corner as DFM 
import scipy.optimize as op
from scipy.stats import norm as Norm
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
from gqp_mc import popinf as PopInf
# --- provabgs --- 
from provabgs import infer as Infer
from provabgs import models as Models
# --- astro ---
from astropy.io import fits
import astropy.table as aTable 
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
    fig = plt.figure(figsize=(15,4))
    gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.65)
    sub = plt.subplot(gs1[0,0], projection='mollweide')
    #sub = fig.add_subplot(gs1[0, :7])
    #gs1 = mpl.gridspec.GridSpec(1,3, figure=fig) 
    #sub = plt.subplot(gs1[0,:-1], projection='mollweide')
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
    gs2 = fig.add_gridspec(nrows=1, ncols=1, wspace=0.05, left=0.67, right=0.95)
    sub = plt.subplot(gs2[0,0])
    sub.hist(z_bgs, range=[0.0, 1.], color='C0', bins=100) 
    sub.hist(z_sdss, range=[0.0, 1.], color='C1', bins=100) 
    sub.hist(np.array(gama['Z']), weights=10.*np.ones(len(gama['Z'])),
            range=[0.0, 1.], color='r', bins=100, alpha=0.5) 
    sub.set_xlabel('Redshift', fontsize=20) 
    sub.set_xlim([0., 0.7])
    sub.set_ylabel('dN/dz', fontsize=20) 
    sub.set_ylim(0., 1.5e6)

    def _fmt(x, pos):
        a, b = '{:.1e}'.format(x).split('e')
        #a = a.split('.')[0]
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
    sub.legend(plts, ['DESI BGS', 'SDSS', r'GAMA $\times 10$'], loc='upper right', handletextpad=0.2, prop={'size': 20}) 
    ffig = os.path.join(dir_doc, 'bgs.pdf')
    fig.savefig(ffig, bbox_inches='tight')

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

    sub.legend([_main, _faint], ['BGS Bright', '$r < 20$'], loc='lower right',
            fontsize=25, markerscale=10, handletextpad=0.)
    sub.set_xlabel('Redshift', fontsize=20)
    sub.set_xlim(0., 0.5)
    sub.set_ylabel('$\log M_*$ [$M_\odot$]', fontsize=20)
    sub.set_ylim(6, 12.5)
    sub.text(0.95, 0.43, 'MXXL sim.', ha='right', va='top', 
            transform=sub.transAxes, fontsize=25)
    ffig = os.path.join(dir_doc, 'bgs_mstar_z.pdf')
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    return None 


def FM_photo():
    ''' plot illustrating the forward model for photometry 
    '''
    from speclite import filters as specFilter

    # read forward modeled Lgal photometry
    photo, meta = Data.Photometry(sim='lgal', noise='legacy', lib='fsps')
    flux_g = photo['flux'][:,0] * 1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs())
    flux_r = photo['flux'][:,1] * 1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs())
    flux_z = photo['flux'][:,2] * 1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()) # convert to 10^-17 ergs/s/cm^2/Ang
    ivar_g = photo['ivar'][:,0] * (1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_r = photo['ivar'][:,1] * (1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_z = photo['ivar'][:,2] * (1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()))**-2. # convert to 10^-17 ergs/s/cm^2/Ang

    # read noiseless Lgal spectroscopy 
    specs, _ = Data.Spectra(sim='lgal', noise='none', lib='fsps') 
    # read in photometric bandpass filters 
    filter_response = specFilter.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z', 'wise2010-W1', 'wise2010-W2')
    wave_eff = [filter_response[i].effective_wavelength.value for i in range(len(filter_response))]

    fig = plt.figure(figsize=(14,4))
    gs1 = mpl.gridspec.GridSpec(1,1, figure=fig) 
    gs1.update(left=0.02, right=0.7)
    sub = plt.subplot(gs1[0])
    
    _plt_sed, = sub.plot(specs['wave'], specs['flux_unscaled'][0], c='k', lw=0.5, ls=':', 
            label='LGal SED')
    _plt_photo = sub.errorbar(wave_eff[:3], [flux_g[0], flux_r[0], flux_z[0]], 
            yerr=[ivar_g[0]**-0.5, ivar_r[0]**-0.5, ivar_z[0]**-0.5], fmt='.C3', markersize=10,
            label='forward modeled DESI photometry') 
    _plt_filter, = sub.plot([0., 0.], [0., 0.], c='k', ls='--', label='broadband filter response') 
    for i in range(3): # len(filter_response)): 
        sub.plot(filter_response[i].wavelength, specs['flux_unscaled'][0].max() * filter_response[i].response, ls='--') 
        sub.text(filter_response[i].effective_wavelength.value, 0.6 * specs['flux_unscaled'][0].max(), ['g', 'r', 'z'][i], fontsize=20, color='C%i' % i)
    sub.set_xlabel('wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3500, 1.05e4)
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=20) 
    sub.set_ylim(0., 1.4*(specs['flux_unscaled'][0].max()))
    sub.legend([_plt_sed, _plt_photo, _plt_filter], 
            ['LGAL SED',  'forward modeled photometry', 'broadband filter response'], 
            loc='upper right', handletextpad=0.2, fontsize=15) 
    
    # Legacy imaging target photometry DR8
    bgs_true = aTable.Table.read(os.path.join(UT.dat_dir(), 'provabgs.sv3.empty.fits'))
    #bgs_true = h5py.File(os.path.join(UT.dat_dir(), 'bgs.1400deg2.rlim21.0.hdf5'), 'r')
    bgs_gmag = 22.5 - 2.5 * np.log10(bgs_true['FLUX_G'])
    bgs_rmag = 22.5 - 2.5 * np.log10(bgs_true['FLUX_R']) 
    bgs_zmag = 22.5 - 2.5 * np.log10(bgs_true['FLUX_Z'])

    rlim = (bgs_rmag < 20.) 
    
    photo_g = 22.5 - 2.5 * np.log10(photo['flux'][:,0])
    photo_r = 22.5 - 2.5 * np.log10(photo['flux'][:,1])
    photo_z = 22.5 - 2.5 * np.log10(photo['flux'][:,2])

    sigma_g = np.abs(-2.5 * (photo['ivar'][:,0]**-0.5)/photo['flux'][:,0]/np.log(10))
    sigma_r = np.abs(-2.5 * (photo['ivar'][:,1]**-0.5)/photo['flux'][:,1]/np.log(10))
    sigma_z = np.abs(-2.5 * (photo['ivar'][:,2]**-0.5)/photo['flux'][:,2]/np.log(10))
        
    gs2 = mpl.gridspec.GridSpec(1,1, figure=fig) 
    gs2.update(left=0.76, right=1.)
    sub = plt.subplot(gs2[0])
    DFM.hist2d(bgs_gmag[rlim] - bgs_rmag[rlim], bgs_rmag[rlim] - bgs_zmag[rlim], color='k', levels=[0.68, 0.95], 
            range=[[-1., 3.], [-1., 3.]], bins=40, smooth=0.5, 
            plot_datapoints=False, fill_contours=False, plot_density=False, linewidth=0.5, ax=sub)
    sub.fill_between([0],[0],[0], fc='none', ec='k', label='BGS Legacy Surveys') 
    sub.errorbar(photo_g - photo_r, photo_r - photo_z, 
            xerr=np.sqrt(sigma_g**2 + sigma_r**2),
            yerr=np.sqrt(sigma_r**2 + sigma_z**2),
            fmt='.C3', markersize=1)#, label='forward modeled DESI photometry') 
    sub.set_xlabel('$g-r$', fontsize=20) 
    sub.set_xlim(0., 2.) 
    sub.set_xticks([0., 1., 2.]) 
    sub.set_ylabel('$r-z$', fontsize=20) 
    sub.set_ylim(0., 1.5) 
    sub.set_yticks([0., 1.]) 
    sub.legend(loc='upper left', fontsize=15) 

    ffig = os.path.join(dir_doc, 'fm_photo.pdf')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def FM_spec():
    ''' plot illustrating the forward model for spectroscopy 
    '''
    igal = 1 
    # read noiseless Lgal spectroscopy 
    spec_s, meta    = Data.Spectra(sim='lgal', noise='none', lib='fsps') 
    spec_bgs, _     = Data.Spectra(sim='lgal', noise='bgs', lib='fsps') 
    
    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(111) 
    
    for i, band in enumerate(['b', 'r', 'z']): 
        if band == 'b': 
            _plt, = sub.plot(spec_bgs['wave_%s' % band], 
                    spec_bgs['flux_%s' % band][igal], c='C%i' % i, lw=0.25) 
        else: 
            sub.plot(spec_bgs['wave_%s' % band], 
                    spec_bgs['flux_%s' % band][igal],
                c='C%i' % i, lw=0.25) 
    
    _plt_lgal, = sub.plot(spec_s['wave'], spec_s['flux'][igal,:], c='k', ls='--', lw=1) 
    _plt_lgal0, = sub.plot(spec_s['wave'], spec_s['flux_unscaled'][igal,:], c='k', ls=':', lw=1) 
    
    leg = sub.legend(
            [_plt, _plt_lgal, _plt_lgal0], 
            ['forward modeled spectrum', 'fiber fraction scaled SED', 'LGAL SED'],
            loc='upper right', handletextpad=0.3, fontsize=17) 
    sub.set_xlabel('wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3500, 1.05e4)
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=20) 
    sub.set_ylim(0., 15)#3.*(spec_s['flux_unscaled'][0].max()))

    ffig = os.path.join(dir_doc, 'fm_spec.pdf')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def nmf_bases():
    '''plot the SFH and ZH bases of our NMF SPS model  
    '''
    m_nmf = Models.NMF(burst=False, emulator=True)
    
    # plot SFH bases 
    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(121)
    for i in range(4): 
        sub.plot(m_nmf._t_lb_hr, m_nmf._sfh_basis_hr[i], label=r'$s_{%i}^{\rm SFH}$' % (i+1)) 
    sub.set_xlim(0., 13.7) 
    sub.set_ylabel(r'star formation rate [$M_\odot/{\rm Gyr}$]', fontsize=20) 
    sub.set_ylim(0., 0.18) 
    sub.set_yticks([0.05, 0.1, 0.15]) 
    sub.legend(loc='upper right', ncol=2, fontsize=20, handletextpad=0.2) 

    # plot ZH bases 
    sub = fig.add_subplot(122)
    for i in range(2):
        sub.plot(m_nmf._t_lb_hr, m_nmf._zh_basis[i](m_nmf._t_lb_hr), label=r'$s_{%i}^{\rm ZH}$' % (i+1)) 
    sub.set_xlim(0., 13.7) 
    sub.set_ylabel('metallicity $Z$', fontsize=20) 
    sub.set_ylim(0., None) 
    sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$t_{\rm lookback}$ [Gyr]', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.2)

    _ffig = os.path.join(dir_doc, 'nmf_bases.pdf') 
    fig.savefig(_ffig, bbox_inches='tight') 
    return None 


def posterior_demo(): 
    ''' Figure demonstrating the inferred posterior with comparison to FMed
    observables. 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))

    wave_obs = np.load(os.path.join(dat_dir, 'mocha_s2.wave.npy'))
    flux_obs = np.load(os.path.join(dat_dir, 'mocha_s2.flux.npy'))
    ivar_obs = np.load(os.path.join(dat_dir, 'mocha_s2.ivar.npy'))

    photo_obs       = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))
    photo_ivar_obs  = np.load(os.path.join(dat_dir, 'mocha_p2.ivar.npy'))

    mags_obs = 22.5 - 2.5 * np.log10(photo_obs) 
    mags_sig_obs = np.abs(-2.5 * (photo_ivar_obs**-0.5)/photo_obs/np.log(10))
    
    igal = 0 
    chain = pickle.load(open(os.path.join(dat_dir, 'L2',
        'SP2.provabgs.%i.chain.p' % igal), 'rb'))

    # corner plot of the posterior and comparison of best-fit to data
    lbls = [r'$\log M_*$', r'$\beta^{\rm SFH}_1$', r'$\beta^{\rm SFH}_2$', r'$\beta^{\rm SFH}_3$', r'$\beta^{\rm SFH}_4$', 
            r'$f_{\rm burst}$', r'$t_{\rm burst}$', r'$\gamma_1^{\rm ZH}$', r'$\gamma_2^{\rm ZH}$', 
            r'$\tau_{\rm BC}$', r'$\tau_{\rm ISM}$', r'$n_{\rm dust}$', r'$f_{\rm fiber}$'] 
    param_lims = [(9.6, 12.), (0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.), (1e-2, 13.27), (4.5e-5, 1.5e-2), (4.5e-5, 1.5e-2), (0., 3.), (0., 3.), (-2., 1.), (0.12, 0.24)]
    ndim = len(lbls)

    fig = plt.figure(figsize=(15, 20))
    gs0 = fig.add_gridspec(nrows=ndim, ncols=ndim, top=0.95, bottom=0.275)
    for yi in range(ndim):
        for xi in range(ndim):
            sub = fig.add_subplot(gs0[yi, xi])
    
    flat_chain = UT.flatten_chain(chain['mcmc_chain'][1500:,:,:])
    _fig = DFM.corner(
            flat_chain[::10,:], 
            quantiles=None, #[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95],
            bins=20,
            smooth=True,
            labels=lbls, 
            label_kwargs={'fontsize': 20, 'labelpad': 0.1}, 
            range=param_lims, 
            fig=fig)
    #axes = np.array(_fig.axes).reshape(ndim, ndim)
    #for i in range(ndim): 
    #    axes[i,i].set_xlim(param_lims[i])
    #DFM.overplot_points(_fig, [chain['theta_bestfit']], color='C1',
    #        markersize=10)

    axes = np.array(fig.axes).reshape((ndim, ndim))
    for yi in range(1, ndim):
        ax = axes[yi, 0]
        ax.set_ylabel(lbls[yi], fontsize=20, labelpad=30)
        ax.yaxis.set_label_coords(-0.6, 0.5)
    for xi in range(ndim): 
        ax = axes[-1, xi]
        ax.set_xlabel(lbls[xi], fontsize=20, labelpad=30)
        ax.xaxis.set_label_coords(0.5, -0.55)

    gs1 = fig.add_gridspec(nrows=1, ncols=30, top=0.2, bottom=0.05)
    sub = fig.add_subplot(gs1[0, :7])
    #sub.errorbar([4720., 6415., 9260.], photo_obs[igal][:3],
    #        yerr=photo_ivar_obs[igal][:3]**-0.5, fmt='.k')
    #sub.scatter([4720., 6415., 9260.], chains[i]['flux_photo_model'], c='C1')
    sub.errorbar([4720., 6415., 9260.], mags_obs[igal][:3],
            yerr=mags_sig_obs[igal][:3], fmt='.k')
    sub.scatter([4720., 6415., 9260.], 22.5 - 2.5 *
            np.log10(chain['flux_photo_model']), marker='s', facecolor='none', s=70, c='C1')
    sub.set_xlim(4000, 1e4)
    sub.set_xticks([4720., 6415., 9260])
    sub.set_xticklabels(['g', 'r', 'z'], fontsize=25)
    sub.set_ylabel('magnitude', fontsize=25)

    sub = fig.add_subplot(gs1[0, 10:])
    sub.plot(wave_obs, flux_obs[igal], c='k', lw=0.5, label='mock observations')
    sub.plot(chain['wavelength_obs'], chain['flux_spec_model'], c='C1', lw=1, label='best-fit model')
    sub.legend(loc='upper right', fontsize=20, handletextpad=0.2)
    sub.set_xlabel('wavelength [$A$]', fontsize=25) 
    sub.set_xlim(3.6e3, 9.8e3)
    sub.set_ylim(0., 20)
    sub.set_ylabel('flux [$erg/s/cm^2/A$]', fontsize=20) 

    _ffig = os.path.join(dir_doc, 'mcmc_posterior_demo.pdf')
    fig.savefig(_ffig, bbox_inches='tight') 
    return None 


def inferred_props(): 
    ''' Figure comparing inferred galaxy properties to the true galaxy
    properties of the LGal mocks.
    '''
    lbls    = [r'$\log M_*$', r'$\log \overline{\rm SFR}_{1Gyr}$', r'$\log Z_{\rm MW}$', 
            r'$t_{\rm age, MW}$', r'$\tau_{\rm ISM}$']
    minmax  = [[8., 12.], [-3., 2], [-2.5, -1.5], [0., 13.2], [0., 2.]]
    widths  = [0.04, 0.04, 0.01, 0.1, 0.03]

    fig = plt.figure(figsize=(25, 15))    
    for i, sample in enumerate(['S2', 'P2', 'SP2']): 
        props_infer, props_truth = L2_chains(sample, derived_properties=True)
        
        for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 

            sub = fig.add_subplot(3,5,5*i+ii+1)
            #dprop = prop_infer - prop_truth[:,None]
            dprop = prop_infer
            if i == 2: 
                print(lbls[ii])
                print(np.std(dprop, axis=1))
                print(np.median(np.std(dprop, axis=1)))

            violins = sub.violinplot(dprop.T, positions=prop_truth,
                                 widths=widths[ii], showextrema=False)
            for violin in violins['bodies']:
                violin.set_facecolor('C%i' % i) 
                #violin.set_edgecolor('black')
                #violin.set_linewidths(0.1)
                violin.set_alpha(0.25)

            sub.plot(minmax[ii], minmax[ii], c='k', ls='--')
            sub.set_xlim(minmax[ii])
            sub.set_ylim(minmax[ii])
            if ii == 0: sub.set_yticks([8., 9., 10., 11., 12.])
            if ii == 1: sub.set_yticks([-3., -2., -1., 0., 1., 2.])
            if ii == 3: sub.set_yticks([0., 5., 10.]) 
            if ii == 4: sub.set_yticks([0., 0.5, 1.0, 1.5, 2.]) 
            if ii == 0: sub.text(0.95, 0.05, ['spectra', 'photometry', 'spectra+photometry'][i], 
                    ha='right', va='bottom', transform=sub.transAxes, fontsize=25)
            if i != 2: sub.set_xticklabels([])
            if i == 0: sub.set_title(lbls[ii], pad=10, fontsize=25)
            sub.set_rasterization_zorder(10)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$\theta_{\rm true}$', fontsize=30) 
    bkgd.set_ylabel(r'$\hat{\theta}$', labelpad=10, fontsize=30) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.2, hspace=0.1)

    ffig = os.path.join(dir_doc, 'inferred_props.pdf')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def eta_l2(method='opt'):
    ''' calculate bias as a function of galaxy properties
    '''
    lbls    = [r'$\log M_*$', r'$\log \overline{\rm SFR}_{\rm 1Gyr}$', r'$\log Z_{\rm MW}$', r'$t_{\rm age, MW}$', r'$\tau_{\rm ISM}$']
    minmax  = [[9., 12.], [-3., 2], [-2.3, -1.7], [0., 11.], [0., 2.]]

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(18, 8))

    #for ii, sample in enumerate(['S2', 'P2', 'SP2']): 
    for ii, sample, clr in zip(range(2), ['P2', 'SP2'], ['C1', 'C2']): 
        props_infer, props_truth = L2_chains(sample)
        
        # get eta for log M*, log SFR, and log Z_MW
        logM_bin    = np.arange(8, 13, 0.2)
        logSFR_bin  = np.arange(-4, 2.5, 0.5)
        logZMW_bin  = np.arange(-2.3, -1.7, 0.05)
        tageMW_bin  = np.arange(0.02, 13.2, 0.5)
        tauism_bin  = np.arange(0., 2., 0.1) 
        
        x_props, eta_mus, eta_sigs = [], [], []
        for prop_infer, prop_truth, prop_bin in zip(props_infer, props_truth, [logM_bin, logSFR_bin, logZMW_bin, tageMW_bin, tauism_bin]): 

            x_prop, eta_mu, eta_sig, nbins = [], [], [], [] 
            for ibin in range(len(prop_bin)-1): 
                inbin = (prop_truth > prop_bin[ibin]) & (prop_truth < prop_bin[ibin+1])
                if np.sum(inbin) > 1: 
                    nbins.append(np.sum(inbin))
                    x_prop.append(0.5 * (prop_bin[ibin] + prop_bin[ibin+1]))
                    
                    if method == 'opt': 
                        _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
                        eta_mu.append(_mu)
                        eta_sig.append(_sig)
                    elif method == 'mcmc':  
                        _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                                niter=1000, burnin=500, thin=5)
                        _mu, _sig = _theta[:,0], _theta[:,1]
                        eta_mu.append(np.median(_mu))
                        eta_sig.append(np.median(_sig))
            print()
            print(x_prop)
            print(nbins)
            print(eta_mu)
            print(eta_sig)
            x_props.append(np.array(x_prop))
            eta_mus.append(np.array(eta_mu))
            eta_sigs.append(np.array(eta_sig))
        
        for i, x_prop, eta_mu, eta_sig in zip(range(len(x_props)), x_props, eta_mus, eta_sigs): 
            sub = fig.add_subplot(2, 3, i+1) 

            sub.plot(minmax[i], [0., 0.], c='k', ls='--')
            sub.fill_between(x_prop, eta_mu - eta_sig, eta_mu + eta_sig,
                    fc=clr, ec='none', alpha=[0.3, 0.6][ii], label=['photometry', 'spectra+photometry'][ii]) 
            sub.scatter(x_prop, eta_mu, c=clr, s=2) 
            sub.plot(x_prop, eta_mu, c=clr)
            if ii == 0: sub.set_xlabel(lbls[i], fontsize=25)
            sub.set_xlim(minmax[i])
            if i not in [0, 3]: sub.set_yticklabels([]) 
            sub.set_ylim(-1.5, 1.5) 
            #if i == 3: sub.text(0.95, 0.05, ['spectra', 'photometry', 'spectra + photometry'][ii], 
            #        ha='right', va='bottom', transform=sub.transAxes, fontsize=25)

        #sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    sub = fig.add_subplot(2, 3, 6) 
    sub.fill_between([], [], [], fc='C1', ec='none', alpha=0.3, label='photometry')
    sub.fill_between([], [], [], fc='C2', ec='none', alpha=0.6, label='spectra+photometry')
    sub.legend(loc='upper left', handletextpad=0.3, fontsize=20)
    sub.axis('off')

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_ylabel(r'$\Delta_\theta$', labelpad=15, fontsize=30) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.1, hspace=0.4)

    ffig = os.path.join(dir_doc, 'etas.pdf')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def eta_l2_v2(method='opt'):
    ''' calculate bias as a function of galaxy properties
    '''
    lbls    = [r'$\log M_*$', r'$\log \overline{\rm SFR}_{\rm 1Gyr}$', r'$\log Z_{\rm MW}$', r'$t_{\rm age, MW}$', r'$\tau_{\rm ISM}$*']
    minmax  = [[9., 12.], [-3., 2], [-2.6, -1.], [0., 11.], [0., 2.]]

    # some subsamples to highlight
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    high_snr = (r_fiber < 20.5) 

    no_bulge = np.array(thetas['logM_bulge']) <= 0. #(10**(np.array(thetas['logM_bulge']) - np.array(thetas['logM_total'])) < 0.01) 

    def _eta_prop(prop_infer, prop_truth, prop_bin, method='opt'): 

        x_prop, eta_mu, eta_sig, nbins = [], [], [], [] 
        for ibin in range(len(prop_bin)-1): 
            inbin = (prop_truth > prop_bin[ibin]) & (prop_truth < prop_bin[ibin+1])

            if np.sum(inbin) > 1: 
                nbins.append(np.sum(inbin))
                x_prop.append(0.5 * (prop_bin[ibin] + prop_bin[ibin+1]))
                
                if method == 'opt': 
                    _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
                    eta_mu.append(_mu)
                    eta_sig.append(_sig)
                elif method == 'mcmc':  
                    _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                            niter=1000, burnin=500, thin=5)
                    _mu, _sig = _theta[:,0], _theta[:,1]
                    eta_mu.append(np.median(_mu))
                    eta_sig.append(np.median(_sig))
        return np.array(x_prop), np.array(eta_mu), np.array(eta_sig), np.array(nbins)

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(13, 8))

    #for ii, sample in enumerate(['S2', 'P2', 'SP2']): 
    for ii, sample, clr in zip(range(2), ['P2', 'SP2'], ['C1', 'C2']): 
        print('--- %s ---' % sample) 
        props_infer, props_truth = L2_chains(sample)
        
        # get eta for log M*, log SFR, and log Z_MW
        logM_bin    = np.arange(8, 13, 0.2)
        logSFR_bin  = np.arange(-4, 2.5, 0.5)
        logZMW_bin  = np.arange(-2.5, -1.5, 0.05)
        tageMW_bin  = np.arange(0.02, 13.2, 0.5)
        tauism_bin  = np.arange(0., 2., 0.1) 
        
        x_props, eta_mus, eta_sigs = [], [], []
        for jj, prop_infer, prop_truth, prop_bin in zip(range(len(props_infer)), props_infer, props_truth, [logM_bin, logSFR_bin, logZMW_bin, tageMW_bin, tauism_bin]): 

            x_prop, eta_mu, eta_sig, nbins = _eta_prop(prop_infer, prop_truth, prop_bin, method=method) 
            print()
            print(x_prop)
            print(nbins)
            print(eta_mu)
            print(eta_sig)
            x_props.append(x_prop)
            eta_mus.append(eta_mu)
            eta_sigs.append(eta_sig)

            if (sample == 'SP2') and (jj == 2): 
                x_prop_highsnr, eta_mu_highsnr, eta_sig_highsnr, _ = _eta_prop(
                        prop_infer[high_snr],
                        prop_truth[high_snr], 
                        prop_bin, method=method) 
            elif (sample == 'SP2') and (jj == 4): 
                x_prop_nobulge, eta_mu_nobulge, eta_sig_nobulge, _ = _eta_prop(
                        prop_infer[no_bulge],
                        prop_truth[no_bulge], 
                        prop_bin, method=method) 

        for i, x_prop, eta_mu, eta_sig in zip(range(len(x_props)), x_props, eta_mus, eta_sigs): 
            sub = fig.add_subplot(2, 3, i+1) 

            sub.plot(minmax[i], minmax[i], c='k', ls='--')
            sub.fill_between(x_prop, x_prop + eta_mu - eta_sig, x_prop + eta_mu + eta_sig,
                    fc=clr, ec='none', alpha=[0.3, 0.6][ii], label=['photometry', 'spectra+photometry'][ii]) 
            sub.scatter(x_prop, x_prop + eta_mu, c=clr, s=2) 
            sub.plot(x_prop, x_prop + eta_mu, c=clr)
            if ii == 0: sub.text(0.05, 0.95, lbls[i], ha='left', va='top', transform=sub.transAxes, fontsize=25)
            sub.set_xlim(minmax[i])
            sub.set_ylim(minmax[i])
            if i == 0: 
                sub.set_xticks([9., 10., 11., 12.]) 
                sub.set_yticks([9., 10., 11., 12.]) 
            elif i == 1: 
                sub.set_xticks([-2, -1, 0., 1., 2.]) 
                sub.set_yticks([-2, -1, 0., 1., 2.]) 
            elif i == 2: 
                sub.set_xticks([-2.5, -2.0, -1.5, -1.0]) 
                sub.set_yticks([-2.5, -2.0, -1.5, -1.0]) 
            elif i == 3: 
                sub.set_xticks([0., 2., 4., 6., 8., 10.]) 
                sub.set_yticks([0., 2., 4., 6., 8., 10.]) 


            if (sample == 'SP2') and (i == 2): 
                sub.fill_between(x_prop_highsnr, 
                        x_prop_highsnr + eta_mu_highsnr - eta_sig_highsnr, 
                        x_prop_highsnr + eta_mu_highsnr + eta_sig_highsnr, 
                        fc='k', ec='none', alpha=0.2, zorder=0) 
                sub.plot(x_prop_highsnr, 
                        x_prop_highsnr + eta_mu_highsnr, c='k', ls='-.', lw=1) 
                sub.plot(x_prop_highsnr, 
                        x_prop_highsnr + eta_mu_highsnr - eta_sig_highsnr,
                        c='k', ls='-.', lw=0.5) 
                sub.plot(x_prop_highsnr, 
                        x_prop_highsnr + eta_mu_highsnr + eta_sig_highsnr,
                        c='k', ls='-.', lw=0.5) 
                print('log Z_MW for high SNR') 
                print(eta_mu_highsnr) 
                print(eta_sig_highsnr) 

            elif (sample == 'SP2') and (i == 4): 
                sub.fill_between(x_prop_nobulge, 
                        x_prop_nobulge + eta_mu_nobulge - eta_sig_nobulge, 
                        x_prop_nobulge + eta_mu_nobulge + eta_sig_nobulge, 
                        fc='k', ec='none', alpha=0.2) 
                sub.plot(x_prop_nobulge, 
                        x_prop_nobulge + eta_mu_nobulge, c='k', ls=':', lw=1) 
                sub.plot(x_prop_nobulge, 
                        x_prop_nobulge + eta_mu_nobulge - eta_sig_nobulge, c='k', ls=':', lw=1) 
                sub.plot(x_prop_nobulge, 
                        x_prop_nobulge + eta_mu_nobulge + eta_sig_nobulge, c='k', ls=':', lw=1) 
                
                print('tau_ism for no bulge') 
                print(eta_mu_nobulge) 
                print(eta_sig_nobulge) 

    sub = fig.add_subplot(2, 3, 6) 
    plt1 = sub.fill_between([], [], [], fc='C2', ec='none', alpha=0.6, zorder=10, label='spectra+photometry')
    plt2 = sub.fill_between([], [], [], fc='C1', ec='none', alpha=0.3, zorder=10, label='photometry')
    plt3, = sub.plot([], [], c='k', ls='-.', lw=1, label=r'$r_{\rm fiber} < 20$', zorder=0)
    plt4, = sub.plot([], [], c='k', ls=':', lw=1, label=r'no bulge', zorder=0)
    sub.legend([plt1, plt2, plt3, plt4], 
            ['spectrophotometry', 'photometry', r'$r_{\rm fiber} < 20$', 'no bulge'],
            loc='upper left', handletextpad=0.3, fontsize=20)
    sub.axis('off')

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$\theta_{\rm true}$', labelpad=15, fontsize=30) 
    bkgd.set_ylabel(r'$\theta_{\rm true} + \Delta_\theta$', labelpad=15, fontsize=30) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    ffig = os.path.join(dir_doc, 'etas_v2.pdf')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def eta_photo_l2(method='opt'):
    ''' calculate bias as a function of galaxy properties
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    g_mag = 22.5 - 2.5 * np.log10(fluxes[:,0])
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 
    z_mag = 22.5 - 2.5 * np.log10(fluxes[:,2]) 

    g_r = g_mag - r_mag
    r_z = r_mag - z_mag 

    high_snr = (r_fiber < 20.5) 
    no_bulge = np.array(thetas['logM_bulge']) <= 0. #(10**(np.array(thetas['logM_bulge']) - np.array(thetas['logM_total'])) < 0.01) 

    def _eta_prop_photo(prop_infer, prop_truth, photo, photo_bin, method='opt'): 
        x_prop, eta_mu, eta_sig, nbins = [], [], [], [] 
        for ibin in range(len(photo_bin)-1): 
            inbin = (photo > photo_bin[ibin]) & (photo < photo_bin[ibin+1])

            if np.sum(inbin) > 1: 
                nbins.append(np.sum(inbin))
                x_prop.append(0.5 * (photo_bin[ibin] + photo_bin[ibin+1]))
                
                if method == 'opt': 
                    _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
                    eta_mu.append(_mu)
                    eta_sig.append(_sig)
                elif method == 'mcmc':  
                    _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                            niter=1000, burnin=500, thin=5)
                    _mu, _sig = _theta[:,0], _theta[:,1]
                    eta_mu.append(np.median(_mu))
                    eta_sig.append(np.median(_sig))

        return np.array(x_prop), np.array(eta_mu), np.array(eta_sig)

    proplbls = [r'\log M_*', r'\log {\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}']
    lbls    = [r'$r_{\rm fiber}$', r'$r$', r'$g-r$', r'$r-z$']
    minmax  = [[17.5, 22.], [15., 20], [0.2, 1.8], [0.1, 1.]]
    dbin    = [0.5, 0.5, 0.1, 0.1]

    photos  = [r_fiber, r_mag, g_r, r_z]

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 15))    

    props_infer, props_truth = L2_chains('SP2')
        
    for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 
        print(proplbls[ii]) 
        # get eta for different photometry bins 
        photo_bins = [np.arange(minmax[ii][0], minmax[ii][1]+dbin[ii], dbin[ii]) for ii in range(len(minmax))]
        
        for i, photo, photo_bin in zip(range(len(photos)), photos, photo_bins): 
            x_prop, eta_mu, eta_sig = _eta_prop_photo(prop_infer, prop_truth, photo, photo_bin, method=method)
            print(eta_mu)

            sub = fig.add_subplot(len(proplbls), len(photos), len(photos)*ii+i+1) 
            sub.plot(minmax[i], [0., 0.], c='k', ls='--')
            sub.fill_between(x_prop, eta_mu - eta_sig, eta_mu + eta_sig, fc='C2', ec='none', alpha=0.5) 
            sub.scatter(x_prop, eta_mu, c='C2', s=2) 
            sub.plot(x_prop, eta_mu, c='C2')
            if ii == 4: sub.set_xlabel(lbls[i], fontsize=25)
            else: sub.set_xticklabels([]) 
            sub.set_xlim(minmax[i])
            if i == 0: sub.set_ylabel(r'$\Delta_{%s}$' % proplbls[ii], fontsize=30)
            if i != 0: sub.set_yticklabels([]) 
            sub.set_ylim(-1., 1.) 
            ''' 
            if (ii == 2) and (i in [0, 2, 3]): 
                x_prop_highsnr, eta_mu_highsnr, eta_sig_highsnr = _eta_prop_photo(
                        prop_infer[high_snr],
                        prop_truth[high_snr], 
                        photo[high_snr], photo_bin, method=method) 
                sub.fill_between(x_prop_highsnr, 
                        eta_mu_highsnr - eta_sig_highsnr, 
                        eta_mu_highsnr + eta_sig_highsnr,
                        fc='k', hatch="///", alpha=0.2, zorder=0) 
                sub.plot(x_prop_highsnr, eta_mu_highsnr, c='k', ls='-.',
                        zorder=0)
            elif ii == 4: 
                x_prop_nobulge, eta_mu_nobulge, eta_sig_nobulge =  _eta_prop_photo(
                        prop_infer[no_bulge],
                        prop_truth[no_bulge], 
                        photo[no_bulge], photo_bin, method=method) 
                sub.fill_between(x_prop_nobulge, 
                        eta_mu_nobulge - eta_sig_nobulge, 
                        eta_mu_nobulge + eta_sig_nobulge,
                        fc='k', hatch="///", alpha=0.2, zorder=0) 
                        #fc='k', ec='k', alpha=0.3) 
                sub.plot(x_prop_nobulge, eta_mu_nobulge, c='k', ls='-.',
                        zorder=0)
            '''
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(dir_doc, 'etas_photo.pdf')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def eta_rfibmag_l2(method='opt', nmin=10): 
    ''' calculate bias as a function of g-r, r-z color 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 

    proplbls = [r'\log M_*', r'\log \overline{\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}']

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 8))

    nhist, xedge, yedge = np.histogram2d(r_mag, r_fiber, bins=20, range=[(14., 22.), (16., 24.)])
    has_gals = np.where(nhist > nmin)

    color_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    props_infer, props_truth = L2_chains('SP2')

    for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus         = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        eta_sigs        = np.zeros(nhist.shape)
        eta_sigs[:,:]   = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((r_mag > xedge[i]) & (r_mag <= xedge[i+1]) & (r_fiber > yedge[j]) & (r_fiber <= yedge[j+1])) 
            assert np.sum(inbin) > nmin

            if method == 'opt': 
                _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
            elif method == 'mcmc':  
                _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                        niter=1000, burnin=500, thin=5)

            eta_mus[i, j] = _mu
            eta_sigs[i, j] = _sig
        
        sub = fig.add_subplot(2, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, vmin=-1, vmax=1., cmap='coolwarm_r')
        sub.scatter(r_mag, r_fiber, c='k', s=0.1)

        sub.set_xlim(16., 21.)
        sub.set_ylim(18., 22.5)
        #sub.text(16.4, 22.6, r'$%s$' % proplbls[ii], ha='left', va='top', fontsize=20)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        sub.set_xticklabels([])
        if ii != 0: sub.set_yticklabels([])

        sub = fig.add_subplot(2, len(proplbls), len(proplbls)+ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs1 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_sigs.T, vmin=-1., vmax=1., cmap='coolwarm_r')
        sub.scatter(r_mag, r_fiber, c='k', s=0.1)

        sub.set_xlim(16., 21.)
        sub.set_ylim(18., 22.5)
        if ii != 0: sub.set_yticklabels([])

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    cbar_ax = fig.add_axes([0.91, 0.55, 0.01, 0.3])
    cbar = fig.colorbar(cs0, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$\mu_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.3])
    cbar = fig.colorbar(cs1, cax=cbar_ax, boundaries=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_ylabel(r'$\sigma_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$r$ magnitude', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'$r_{\rm fiber}$ magnitude', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_etas_rfibmag.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def eta_color_l2(method='opt', nmin=10): 
    ''' calculate bias as a function of g-r, r-z color 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    g_mag = 22.5 - 2.5 * np.log10(fluxes[:,0])
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 
    z_mag = 22.5 - 2.5 * np.log10(fluxes[:,2]) 

    g_r = g_mag - r_mag
    r_z = r_mag - z_mag 

    high_snr = (r_fiber < 20.5) 

    proplbls = [r'\log M_*', r'\log \overline{\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}']

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 8))

    nhist, xedge, yedge = np.histogram2d(g_r, r_z, bins=20, range=[(0., 2.), (0., 2.)])
    has_gals = np.where(nhist > nmin)

    color_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    props_infer, props_truth = L2_chains('SP2')

    for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus         = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        eta_sigs        = np.zeros(nhist.shape)
        eta_sigs[:,:]   = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((g_r > xedge[i]) & (g_r <= xedge[i+1]) & (r_z > yedge[j]) & (r_z <= yedge[j+1])) 
            assert np.sum(inbin) > nmin

            if method == 'opt': 
                _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
            elif method == 'mcmc':  
                _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                        niter=1000, burnin=500, thin=5)

            eta_mus[i, j] = _mu
            eta_sigs[i, j] = _sig
        
        sub = fig.add_subplot(2, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, vmin=-1, vmax=1., cmap='coolwarm_r')
        sub.scatter(g_r, r_z, c='k', s=0.1)

        sub.set_xlim(0., 1.8)
        sub.set_xticklabels([])
        sub.set_ylim(0., 1.2)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])
        sub.set_yticks([0., 0.4, 0.8, 1.2]) 

        sub = fig.add_subplot(2, len(proplbls), len(proplbls)+ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs1 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_sigs.T, vmin=-1., vmax=1., cmap='coolwarm_r')
        sub.scatter(g_r, r_z, c='k', s=0.1)

        sub.set_xlim(0., 1.8)
        sub.set_ylim(0., 1.2)
        if ii != 0: sub.set_yticklabels([])
        sub.set_yticks([0., 0.4, 0.8, 1.2]) 

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    cbar_ax = fig.add_axes([0.91, 0.55, 0.01, 0.3])
    cbar = fig.colorbar(cs0, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$\mu_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.3])
    cbar = fig.colorbar(cs1, cax=cbar_ax, boundaries=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_ylabel(r'$\sigma_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$g - r$ color', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'$r - z$ color', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_etas_color.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def eta_msfr_l2(method='opt', nmin=10): 
    ''' calculate bias as a function of g-r, r-z color 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 8))    

    props_infer, props_truth = L2_chains('SP2')

    mstar = props_truth[0,:]
    sfr = props_truth[1,:]
    
    nhist, xedge, yedge = np.histogram2d(mstar, sfr, bins=[20, 8], range=[(8., 12.5), (0., 2.)])
    _nhist, _xedge, _yedge = np.histogram2d(mstar, sfr, bins=[20, 6], range=[(8., 12.5), (-3., 0.)])

    nhist = np.concatenate([_nhist, nhist], axis=1)
    yedge = np.concatenate([_yedge, yedge[1:]])

    has_gals = np.where(nhist > nmin)

    msfr_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    proplbls = [r'\log M_*', r'\log \overline{\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}\ast']

    for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        eta_sigs        = np.zeros(nhist.shape)
        eta_sigs[:,:]   = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((mstar > xedge[i]) & (mstar <= xedge[i+1]) & (sfr > yedge[j]) & (sfr <= yedge[j+1])) 

            if method == 'opt': 
                _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
            elif method == 'mcmc':  
                _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                        niter=1000, burnin=500, thin=5)

            eta_mus[i, j] = _mu
            eta_sigs[i, j] = _sig

        sub = fig.add_subplot(2, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, vmin=-1, vmax=1., cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_xticklabels([])
        sub.set_ylim(-2., 2.)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])

        sub = fig.add_subplot(2, len(proplbls), len(proplbls)+ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs1 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_sigs.T, vmin=-1., vmax=1., cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_ylim(-2., 2.)
        if ii != 0: sub.set_yticklabels([])

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    cbar_ax = fig.add_axes([0.91, 0.55, 0.01, 0.3])
    cbar = fig.colorbar(cs0, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$\mu_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.3])
    cbar = fig.colorbar(cs1, cax=cbar_ax, boundaries=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_ylabel(r'$\sigma_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'true $\log M_*$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'true $\log \overline{\rm SFR}_{\rm 1Gyr}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, 'etas_msfr.pdf')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def SFH_demo(): 
    ''' figure that demonstrates
    '''
    # 0, 5 looks pretty good 
    #igal    = 5 
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'

    # read in parameters of L2 mocks 
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    lgal_sfr = thetas['sfr_1gyr'][:100] # this is because I dont' want to download 2200 mcmc chains
    lgal_ms = thetas['logM_total'][:100]
    lgal_logssfr = np.log10(lgal_sfr) - lgal_ms

    # pickle 1 SF 1 Green Valley 1 Quiescent galaxy 
    is_sf = (lgal_logssfr > -10.5)
    is_g  = (lgal_logssfr < -10.5) & (lgal_logssfr > -11.0)
    is_q  = (lgal_logssfr < -11.0)
    print(lgal_logssfr[5]) 

    i_sf    = 5 #np.random.choice(np.arange(100)[is_sf])
    i_g     = np.random.choice(np.arange(100)[is_g]) # 92
    i_q     = 68 # np.random.choice(np.arange(100)[is_q])

    #igals = [i_sf, i_g, i_q]
    #colrs = ['C0', 'C2', 'C1']
    #print(igals) 
    igals = [i_q, i_sf, i_q]
    colrs = ['C1', 'C0']
    
    # plot 
    fig = plt.figure(figsize=(12,5))
    sub0 = fig.add_subplot(121)
    sub1 = fig.add_subplot(122)
    
    for i, igal, colr in zip(range(len(igals)), igals, colrs):
        zred = thetas['redshift'][igal]

        tlb_true = thetas['t_lookback'][igal]
        sfh_true = (thetas['sfh_disk'][igal] + thetas['sfh_bulge'][igal]) / thetas['dt'][igal] / 1e9

        zh_true_disk    = thetas['Z_disk'][igal]
        zh_true_bulge   = thetas['Z_bulge'][igal]
        zh_mw_true      = (zh_true_disk * thetas['sfh_disk'][igal] + zh_true_bulge * thetas['sfh_bulge'][igal]) / (thetas['sfh_disk'][igal] + thetas['sfh_bulge'][igal])

        # read MCMC chain 
        chain = pickle.load(open(os.path.join(dat_dir, 'L2', 
            'SP2.provabgs.%i.chain.p' % igal), 'rb'))
        flat_chain = UT.flatten_chain(chain['mcmc_chain'][1500:,:,:])[:,:-1] # ignore f_fiber
        # calculate SFHs and ZHs
        m_nmf = Models.NMF(burst=True, emulator=True) # SPS model  
        
        _sfhs = [m_nmf.SFH(tt, zred=zred) for tt in flat_chain]

        tlb_edges = _sfhs[0][0] # lookback time bin edges

        sfhs = np.array([_sfh[1] for _sfh in _sfhs]) # star-formation histories 
        zhs  = np.array([m_nmf.ZH(tt, zred=zred)[1] for tt in flat_chain]) # metallicity histories

        # get quantiles of the SFHs and ZHs 
        q_sfhs  = np.quantile(sfhs, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0) 
        q_zhs   = np.quantile(zhs, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0) 
        
        sub0.plot(tlb_true, sfh_true, ls='--', c=colr)
        sub0.fill_between(0.5*(tlb_edges[1:] + tlb_edges[:-1]), q_sfhs[0], q_sfhs[-1], 
                alpha=0.2, color=colr, linewidth=0)
        sub0.fill_between(0.5*(tlb_edges[1:] + tlb_edges[:-1]), q_sfhs[1], q_sfhs[-2], 
                alpha=0.5, color=colr, linewidth=0)

        sub1.plot(tlb_true, zh_mw_true, c=colr, ls='--', label='LGAL (true)')
        sub1.fill_between(0.5*(tlb_edges[1:] + tlb_edges[:-1]), q_zhs[0], q_zhs[-1],
                alpha=0.2, color=colr, linewidth=0)
        sub1.fill_between(0.5*(tlb_edges[1:] + tlb_edges[:-1]), q_zhs[1], q_zhs[-2],
                alpha=0.5, color=colr, linewidth=0, label='PROVABGS inferred')

        if i == 0: sub1.legend(loc='lower left', fontsize=20, handletextpad=0.2)  
        
    sub0.set_ylabel(r'star formation history [$M_\odot/{\rm yr}$]', fontsize=25) 
    sub0.set_xlim(0., 12)
    sub0.set_yscale('log')

    sub1.set_ylabel(r'metallicity history', fontsize=25) 
    sub1.set_xlim(0., 12)
    sub1.set_yscale('log')

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$t_{\rm lookback}$', labelpad=10, fontsize=30) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.3)
    _ffig = os.path.join(dir_doc, 'sfh_demo.pdf')
    fig.savefig(_ffig, bbox_inches='tight') 
    return None 


# --- high SNR ---
def eta_l2_highSNR(method='opt'):
    ''' calculate bias as a function of galaxy properties
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    high_snr = (r_fiber < 20.) 

    lbls    = [r'$\log M_*$', r'$\log \overline{\rm SFR}_{\rm 1Gyr}$', r'$\log Z_{\rm MW}$', r'$t_{\rm age, MW}$', r'$\tau_{\rm ISM}$*']
    minmax  = [[9., 12.], [-3., 2], [-2.6, -1.], [0., 11.], [0., 2.]]

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(13, 8))

    for ii, sample, clr in zip(range(1), ['SP2'], ['C2']): 
        print('--- %s ---' % sample) 
        props_infer, props_truth = L2_chains(sample)
        props_infer = props_infer[:,high_snr,:]
        props_truth = props_truth[:,high_snr]
        
        # get eta for log M*, log SFR, and log Z_MW
        logM_bin    = np.arange(8, 13, 0.2)
        logSFR_bin  = np.arange(-4, 2.5, 0.5)
        logZMW_bin  = np.arange(-2.5, -1.5, 0.05)
        tageMW_bin  = np.arange(0.02, 13.2, 0.5)
        tauism_bin  = np.arange(0., 2., 0.1) 
        
        x_props, eta_mus, eta_sigs = [], [], []
        for prop_infer, prop_truth, prop_bin in zip(props_infer, props_truth, [logM_bin, logSFR_bin, logZMW_bin, tageMW_bin, tauism_bin]): 

            x_prop, eta_mu, eta_sig, nbins = [], [], [], [] 
            for ibin in range(len(prop_bin)-1): 
                inbin = (prop_truth > prop_bin[ibin]) & (prop_truth < prop_bin[ibin+1])
                if np.sum(inbin) > 1: 
                    nbins.append(np.sum(inbin))
                    x_prop.append(0.5 * (prop_bin[ibin] + prop_bin[ibin+1]))
                    
                    if method == 'opt': 
                        _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
                        eta_mu.append(_mu)
                        eta_sig.append(_sig)
                    elif method == 'mcmc':  
                        _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                                niter=1000, burnin=500, thin=5)
                        _mu, _sig = _theta[:,0], _theta[:,1]
                        eta_mu.append(np.median(_mu))
                        eta_sig.append(np.median(_sig))
            print()
            print(x_prop)
            print(nbins)
            print(eta_mu)
            print(eta_sig)
            x_props.append(np.array(x_prop))
            eta_mus.append(np.array(eta_mu))
            eta_sigs.append(np.array(eta_sig))
        
        for i, x_prop, eta_mu, eta_sig in zip(range(len(x_props)), x_props, eta_mus, eta_sigs): 
            sub = fig.add_subplot(2, 3, i+1) 

            sub.plot(minmax[i], minmax[i], c='k', ls='--')
            sub.fill_between(x_prop, x_prop + eta_mu - eta_sig, x_prop + eta_mu + eta_sig,
                    fc=clr, ec='none', alpha=[0.3, 0.6][ii], label=['photometry', 'spectra+photometry'][ii]) 
            sub.scatter(x_prop, x_prop + eta_mu, c=clr, s=2) 
            sub.plot(x_prop, x_prop + eta_mu, c=clr)
            if ii == 0: sub.text(0.05, 0.95, lbls[i], ha='left', va='top', transform=sub.transAxes, fontsize=25)
            sub.set_xlim(minmax[i])
            sub.set_ylim(minmax[i])
            if i == 0: 
                sub.set_xticks([9., 10., 11., 12.]) 
                sub.set_yticks([9., 10., 11., 12.]) 
            elif i == 1: 
                sub.set_xticks([-2, -1, 0., 1., 2.]) 
                sub.set_yticks([-2, -1, 0., 1., 2.]) 
            elif i == 2: 
                sub.set_xticks([-2.5, -2.0, -1.5, -1.0]) 
                sub.set_yticks([-2.5, -2.0, -1.5, -1.0]) 
            elif i == 3: 
                sub.set_xticks([0., 2., 4., 6., 8., 10.]) 
                sub.set_yticks([0., 2., 4., 6., 8., 10.]) 

    sub = fig.add_subplot(2, 3, 6) 
    sub.fill_between([], [], [], fc='C1', ec='none', alpha=0.3, label='photometry')
    sub.fill_between([], [], [], fc='C2', ec='none', alpha=0.6, label='spectra+photometry')
    sub.legend(loc='upper left', handletextpad=0.3, fontsize=20)
    sub.axis('off')

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$\theta_{\rm true}$', labelpad=15, fontsize=30) 
    bkgd.set_ylabel(r'$\theta_{\rm true} + \Delta_\theta$', labelpad=15, fontsize=30) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    ffig = os.path.join(dir_doc, '_etas_highSNR.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _eta_photo_highSNR(method='opt'):
    ''' calculate bias as a function of galaxy properties
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    g_mag = 22.5 - 2.5 * np.log10(fluxes[:,0])
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 
    z_mag = 22.5 - 2.5 * np.log10(fluxes[:,2]) 

    g_r = g_mag - r_mag
    r_z = r_mag - z_mag 

    high_snr = (r_fiber < 20.) 

    proplbls = [r'\log M_*', r'\log {\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}']
    lbls    = [r'$r_{\rm fiber}$', r'$r$', r'$g-r$', r'$r-z$']
    minmax  = [[17.5, 22.], [15., 20], [0.2, 1.8], [0.1, 1.]]
    dbin    = [0.5, 0.5, 0.1, 0.1]

    photos  = [r_fiber[high_snr], r_mag[high_snr], g_r[high_snr], r_z[high_snr]]

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 15))    

    props_infer, props_truth = L2_chains('SP2')
    props_infer = props_infer[:,high_snr,:]
    props_truth = props_truth[:,high_snr]
        
    for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 
        print(proplbls[ii]) 
        # get eta for different photometry bins 
        photo_bins = [np.arange(minmax[ii][0], minmax[ii][1]+dbin[ii], dbin[ii]) for ii in range(len(minmax))]
        
        x_props, eta_mus, eta_sigs = [], [], []
        for photo, photo_bin in zip(photos, photo_bins): 

            x_prop, eta_mu, eta_sig = [], [], []
            for ibin in range(len(photo_bin)-1): 
                #inbin = high_snr & (photo > photo_bin[ibin]) & (photo < photo_bin[ibin+1])
                inbin = (photo > photo_bin[ibin]) & (photo < photo_bin[ibin+1])
                if np.sum(inbin) > 3: 
                    x_prop.append(0.5 * (photo_bin[ibin] + photo_bin[ibin+1]))
                    
                    if method == 'opt': 
                        _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
                        eta_mu.append(_mu)
                        eta_sig.append(_sig)
                    elif method == 'mcmc':  
                        _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                                niter=1000, burnin=500, thin=5)
                        _mu, _sig = _theta[:,0], _theta[:,1]
                        eta_mu.append(np.median(_mu))
                        eta_sig.append(np.median(_sig))
            print(eta_mu)
            x_props.append(np.array(x_prop))
            eta_mus.append(np.array(eta_mu))
            eta_sigs.append(np.array(eta_sig))
        
        for i, x_prop, eta_mu, eta_sig in zip(range(len(x_props)), x_props, eta_mus, eta_sigs): 
            sub = fig.add_subplot(len(proplbls), len(x_props), len(x_props)*ii+i+1) 
            sub.plot(minmax[i], [0., 0.], c='k', ls='--')
            sub.fill_between(x_prop, eta_mu - eta_sig, eta_mu + eta_sig, fc='C2', ec='none', alpha=0.5) 
            sub.scatter(x_prop, eta_mu, c='C2', s=2) 
            sub.plot(x_prop, eta_mu, c='C2')
            if ii == 4: sub.set_xlabel(lbls[i], fontsize=25)
            else: sub.set_xticklabels([]) 
            sub.set_xlim(minmax[i])
            if i == 0: sub.set_ylabel(r'$\Delta_{%s}$' % proplbls[ii], fontsize=30)
            if i != 0: sub.set_yticklabels([]) 
            sub.set_ylim(-1., 1.) 
        #sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(dir_doc, '_etas_photo_highsnr.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _eta_color_highSNR(method='opt', nmin=10): 
    ''' calculate bias as a function of g-r, r-z color 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    g_mag = 22.5 - 2.5 * np.log10(fluxes[:,0])
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 
    z_mag = 22.5 - 2.5 * np.log10(fluxes[:,2]) 

    g_r = g_mag - r_mag
    r_z = r_mag - z_mag 

    high_snr = (r_fiber < 20.) 
    print('%i of %i high SNR' % (np.sum(high_snr), len(high_snr)))

    proplbls = [r'\log M_*', r'\log \overline{\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}']

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 8))

    nhist, xedge, yedge = np.histogram2d(g_r[high_snr], r_z[high_snr], bins=20, range=[(0., 2.), (0., 2.)])
    has_gals = np.where(nhist > nmin)

    color_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    props_infer, props_truth = L2_chains('SP2')
    props_infer = props_infer[:,high_snr,:]
    props_truth = props_truth[:,high_snr]

    for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus         = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        eta_sigs        = np.zeros(nhist.shape)
        eta_sigs[:,:]   = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((g_r[high_snr] > xedge[i]) & (g_r[high_snr] <= xedge[i+1]) & (r_z[high_snr] > yedge[j]) & (r_z[high_snr] <= yedge[j+1])) 
            assert np.sum(inbin) > nmin

            _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])

            eta_mus[i, j] = _mu
            eta_sigs[i, j] = _sig
        
        sub = fig.add_subplot(2, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, vmin=-1, vmax=1., cmap='coolwarm_r')
        sub.scatter(g_r[high_snr], r_z[high_snr], c='k', s=0.1)

        sub.set_xlim(0., 1.8)
        sub.set_xticklabels([])
        sub.set_ylim(0., 1.2)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])
        sub.set_yticks([0., 0.4, 0.8, 1.2]) 

        sub = fig.add_subplot(2, len(proplbls), len(proplbls)+ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs1 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_sigs.T, vmin=-1., vmax=1., cmap='coolwarm_r')
        sub.scatter(g_r[high_snr], r_z[high_snr], c='k', s=0.1)

        sub.set_xlim(0., 1.8)
        sub.set_ylim(0., 1.2)
        if ii != 0: sub.set_yticklabels([])
        sub.set_yticks([0., 0.4, 0.8, 1.2]) 

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    cbar_ax = fig.add_axes([0.91, 0.55, 0.01, 0.3])
    cbar = fig.colorbar(cs0, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$\mu_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.3])
    cbar = fig.colorbar(cs1, cax=cbar_ax, boundaries=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_ylabel(r'$\sigma_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$g - r$ color', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'$r - z$ color', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_etas_color_highSNR.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _eta_msfr_highSNR(method='opt', nmin=10): 
    ''' calculate bias as a function of g-r, r-z color 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    high_snr = (r_fiber < 20.) 
    print('%i of %i high SNR' % (np.sum(high_snr), len(high_snr)))

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 8))    

    props_infer, props_truth = L2_chains('SP2')
    props_infer = props_infer[:,high_snr,:]
    props_truth = props_truth[:,high_snr]

    mstar = props_truth[0,:]
    sfr = props_truth[1,:]
    
    nhist, xedge, yedge = np.histogram2d(mstar, sfr, bins=[20, 4], range=[(8., 12.5), (0., 2.)])
    _nhist, _xedge, _yedge = np.histogram2d(mstar, sfr, bins=[20, 3], range=[(8., 12.5), (-3., 0.)])

    nhist = np.concatenate([_nhist, nhist], axis=1)
    yedge = np.concatenate([_yedge, yedge[1:]])

    has_gals = np.where(nhist > nmin)

    msfr_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    proplbls = [r'\log M_*', r'\log \overline{\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}']

    for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        eta_sigs        = np.zeros(nhist.shape)
        eta_sigs[:,:]   = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((mstar > xedge[i]) & (mstar <= xedge[i+1]) & (sfr > yedge[j]) & (sfr <= yedge[j+1])) 

            if method == 'opt': 
                _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
            elif method == 'mcmc':  
                _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                        niter=1000, burnin=500, thin=5)

            eta_mus[i, j] = _mu
            eta_sigs[i, j] = _sig

        sub = fig.add_subplot(2, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, vmin=-1, vmax=1., cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_xticklabels([])
        sub.set_ylim(-2., 2.)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])

        sub = fig.add_subplot(2, len(proplbls), len(proplbls)+ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs1 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_sigs.T, vmin=-1., vmax=1., cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_ylim(-2., 2.)
        if ii != 0: sub.set_yticklabels([])

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    cbar_ax = fig.add_axes([0.91, 0.55, 0.01, 0.3])
    cbar = fig.colorbar(cs0, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$\mu_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.3])
    cbar = fig.colorbar(cs1, cax=cbar_ax, boundaries=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_ylabel(r'$\sigma_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'true $\log M_*$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'true $\log \overline{\rm SFR}_{\rm 1Gyr}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_etas_msfr_highsnr.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _eta_msfr_young(method='opt', nmin=10): 
    ''' 
    '''
    # eta as a function of galaxy properties 
    props_infer, props_truth = L2_chains('SP2')
    
    tage = props_truth[3,:]
    young = (tage < 6.)
    print('%i of %i have young stellar populations' % (np.sum(young), len(young)))

    mstar = props_truth[0,young]
    sfr = props_truth[1,young]
    
    nhist, xedge, yedge = np.histogram2d(mstar, sfr, bins=[20, 4], range=[(8., 12.5), (0., 2.)])
    _nhist, _xedge, _yedge = np.histogram2d(mstar, sfr, bins=[20, 3], range=[(8., 12.5), (-3., 0.)])

    nhist = np.concatenate([_nhist, nhist], axis=1)
    yedge = np.concatenate([_yedge, yedge[1:]])

    has_gals = np.where(nhist > nmin)

    msfr_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    proplbls = [r't_{\rm age, MW}']

    fig = plt.figure(figsize=(6, 8))    

    for ii, prop_infer, prop_truth in zip(range(1), [props_infer[3, young, :]], [props_truth[3, young]]): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        eta_sigs        = np.zeros(nhist.shape)
        eta_sigs[:,:]   = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((mstar > xedge[i]) & (mstar <= xedge[i+1]) & (sfr > yedge[j]) & (sfr <= yedge[j+1])) 

            if method == 'opt': 
                _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
            elif method == 'mcmc':  
                _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                        niter=1000, burnin=500, thin=5)

            eta_mus[i, j] = _mu
            eta_sigs[i, j] = _sig

        sub = fig.add_subplot(2, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, vmin=-1, vmax=1., cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_xticklabels([])
        sub.set_ylim(-2., 2.)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])

        sub = fig.add_subplot(2, len(proplbls), len(proplbls)+ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs1 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_sigs.T, vmin=-1., vmax=1., cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_ylim(-2., 2.)
        if ii != 0: sub.set_yticklabels([])

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    cbar_ax = fig.add_axes([0.91, 0.55, 0.01, 0.3])
    cbar = fig.colorbar(cs0, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$\mu_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.3])
    cbar = fig.colorbar(cs1, cax=cbar_ax, boundaries=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_ylabel(r'$\sigma_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'true $\log M_*$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'true $\log \overline{\rm SFR}_{\rm 1Gyr}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_etas_msfr_young.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _eta_msfr_nobulge(method='opt', nmin=10): 
    ''' calculate bias as a function of g-r, r-z color 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))

    small_bulge = (10**(np.array(thetas['logM_bulge']) - np.array(thetas['logM_total'])) < 0.01) 
    print('%i galaxies with small bulge' % np.sum(small_bulge))
    print(np.array(thetas['logM_bulge'])[small_bulge])

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(5, 8))    

    props_infer, props_truth = L2_chains('SP2')
    props_infer = props_infer[:,small_bulge,:]
    props_truth = props_truth[:,small_bulge]

    mstar = props_truth[0,:]
    sfr = props_truth[1,:]
    
    nhist, xedge, yedge = np.histogram2d(mstar, sfr, bins=[20, 4], range=[(8., 12.5), (0., 2.)])
    _nhist, _xedge, _yedge = np.histogram2d(mstar, sfr, bins=[20, 3], range=[(8., 12.5), (-3., 0.)])

    nhist = np.concatenate([_nhist, nhist], axis=1)
    yedge = np.concatenate([_yedge, yedge[1:]])

    has_gals = np.where(nhist > nmin)

    msfr_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    proplbls = [r'\tau_{\rm ISM}']

    for ii, prop_infer, prop_truth in zip(range(1), [props_infer[4]], [props_truth[4]]): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        eta_sigs        = np.zeros(nhist.shape)
        eta_sigs[:,:]   = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((mstar > xedge[i]) & (mstar <= xedge[i+1]) & (sfr > yedge[j]) & (sfr <= yedge[j+1])) 

            if method == 'opt': 
                _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
            elif method == 'mcmc':  
                _theta = PopInf.eta_Delta_mcmc(prop_infer[inbin,:] - prop_truth[inbin, None], 
                        niter=1000, burnin=500, thin=5)

            eta_mus[i, j] = _mu
            eta_sigs[i, j] = _sig

        sub = fig.add_subplot(2, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, vmin=-1, vmax=1., cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_xticklabels([])
        sub.set_ylim(-2., 2.)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])

        sub = fig.add_subplot(2, len(proplbls), len(proplbls)+ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs1 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_sigs.T, vmin=-1., vmax=1., cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_ylim(-2., 2.)
        if ii != 0: sub.set_yticklabels([])

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    cbar_ax = fig.add_axes([0.91, 0.55, 0.01, 0.3])
    cbar = fig.colorbar(cs0, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$\mu_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.3])
    cbar = fig.colorbar(cs1, cax=cbar_ax, boundaries=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_ylabel(r'$\sigma_{\Delta_\theta}$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'true $\log M_*$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'true $\log \overline{\rm SFR}_{\rm 1Gyr}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_etas_msfr_nobulge.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _color_photo(nmin=10): 
    ''' true properties as a function of color
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    g_mag = 22.5 - 2.5 * np.log10(fluxes[:,0])
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 
    z_mag = 22.5 - 2.5 * np.log10(fluxes[:,2]) 

    g_r = g_mag - r_mag
    r_z = r_mag - z_mag 
    
    props = [r_fiber, r_mag, g_r, r_z] 
    proplbls = [r'r_{\rm fiber}', r'r', r'g-r', r'r-z']

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 4))

    nhist, xedge, yedge = np.histogram2d(g_r, r_z, bins=20, range=[(0., 2.), (0., 2.)])
    has_gals = np.where(nhist > nmin)

    color_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    _, props_truth = L2_chains('SP2')

    for ii, prop_truth in zip(range(len(props)), props): 
        print(proplbls[ii]) 

        eta_mus         = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((g_r > xedge[i]) & (g_r <= xedge[i+1]) & (r_z > yedge[j]) & (r_z <= yedge[j+1])) 
            assert np.sum(inbin) > nmin

            eta_mus[i, j] = np.median(prop_truth[inbin]) 
        
        sub = fig.add_subplot(1, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, cmap='coolwarm_r')
        sub.scatter(g_r, r_z, c='k', s=0.1)

        sub.set_xlim(0., 1.8)
        sub.set_ylim(0., 1.2)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])
        sub.set_yticks([0., 0.4, 0.8, 1.2]) 
        fig.colorbar(cs0, ax=sub)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$g - r$ color', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'$r - z$ color', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_color_photo.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _color_trueprops(nmin=10): 
    ''' true properties as a function of color
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    g_mag = 22.5 - 2.5 * np.log10(fluxes[:,0])
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 
    z_mag = 22.5 - 2.5 * np.log10(fluxes[:,2]) 

    g_r = g_mag - r_mag
    r_z = r_mag - z_mag 

    proplbls = [r'\log M_*', r'\log \overline{\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}']

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(24, 4))

    nhist, xedge, yedge = np.histogram2d(g_r, r_z, bins=20, range=[(0., 2.), (0., 2.)])
    has_gals = np.where(nhist > nmin)

    color_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    _, props_truth = L2_chains('SP2')

    for ii, prop_truth in zip(range(len(props_truth)), props_truth): 
        print(proplbls[ii]) 

        eta_mus         = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan
        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((g_r > xedge[i]) & (g_r <= xedge[i+1]) & (r_z > yedge[j]) & (r_z <= yedge[j+1])) 
            assert np.sum(inbin) > nmin

            eta_mus[i, j] = np.median(prop_truth[inbin]) 
        
        sub = fig.add_subplot(1, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, cmap='coolwarm_r')
        sub.scatter(g_r, r_z, c='k', s=0.1)

        sub.set_xlim(0., 1.8)
        sub.set_xticklabels([])
        sub.set_ylim(0., 1.2)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])
        sub.set_yticks([0., 0.4, 0.8, 1.2]) 
        fig.colorbar(cs0, ax=sub)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$g - r$ color', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'$r - z$ color', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_color_trueprops.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _msfr_photo(nmin=3):
    ''' calculate bias as a function of galaxy properties
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'

    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    g_mag = 22.5 - 2.5 * np.log10(fluxes[:,0])
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 
    z_mag = 22.5 - 2.5 * np.log10(fluxes[:,2]) 

    g_r = g_mag - r_mag
    r_z = r_mag - z_mag 

    props = [r_fiber, r_mag, g_r, r_z] 

    _, props_truth = L2_chains('SP2')

    mstar = props_truth[0,:]
    sfr = props_truth[1,:]
    
    nhist, xedge, yedge = np.histogram2d(mstar, sfr, bins=[20, 8], range=[(8., 12.5), (0., 2.)])
    _nhist, _xedge, _yedge = np.histogram2d(mstar, sfr, bins=[20, 6], range=[(8., 12.5), (-3., 0.)])

    nhist = np.concatenate([_nhist, nhist], axis=1)
    yedge = np.concatenate([_yedge, yedge[1:]])

    has_gals = np.where(nhist > nmin)

    msfr_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    proplbls = [r'r_{\rm fiber}', r'r', r'g-r', r'r-z']

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 4))    

    for ii, prop in zip(range(len(props)), props): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((mstar > xedge[i]) & (mstar <= xedge[i+1]) & (sfr > yedge[j]) & (sfr <= yedge[j+1])) 
            eta_mus[i, j] = np.median(prop[inbin])

        sub = fig.add_subplot(1, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_ylim(-2., 2.)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])
        fig.colorbar(cs0, ax=sub)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'true $\log M_*$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'true $\log \overline{\rm SFR}_{\rm 1Gyr}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_msfr_photo.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _msfr_trueprops(nmin=3):
    ''' calculate bias as a function of galaxy properties
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'


    _, props_truth = L2_chains('SP2')

    mstar = props_truth[0,:]
    sfr = props_truth[1,:]

    props = props_truth
    
    nhist, xedge, yedge = np.histogram2d(mstar, sfr, bins=[20, 8], range=[(8., 12.5), (0., 2.)])
    _nhist, _xedge, _yedge = np.histogram2d(mstar, sfr, bins=[20, 6], range=[(8., 12.5), (-3., 0.)])

    nhist = np.concatenate([_nhist, nhist], axis=1)
    yedge = np.concatenate([_yedge, yedge[1:]])

    has_gals = np.where(nhist > nmin)

    msfr_bins = [(xedge[i], xedge[i+1], yedge[j], yedge[j+1]) for i, j in zip(has_gals[0], has_gals[1])]

    proplbls = [r'\log M_*', r'\log \overline{\rm SFR}_{\rm 1Gyr}', r'\log Z_{\rm MW}', r't_{\rm age, MW}', r'\tau_{\rm ISM}']
    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(25, 4))    

    for ii, prop in zip(range(len(props)), props): 
        print(proplbls[ii]) 
        # get eta for different color bins 
        
        eta_mus = np.zeros(nhist.shape)
        eta_mus[:,:]    = np.nan

        for i, j in zip(has_gals[0], has_gals[1]): 
            inbin = ((mstar > xedge[i]) & (mstar <= xedge[i+1]) & (sfr > yedge[j]) & (sfr <= yedge[j+1])) 
            eta_mus[i, j] = np.median(prop[inbin])

        sub = fig.add_subplot(1, len(proplbls), ii+1) 

        X, Y = np.meshgrid(xedge, yedge)
        cs0 = sub.pcolormesh(X, Y, #0.5*(xedge[:-1] + xedge[1:]), 0.5*(yedge[:-1] + yedge[1:]),
                eta_mus.T, cmap='coolwarm_r', rasterized=True)
        sub.scatter(mstar, sfr, c='k', s=0.1, rasterized=True)

        sub.set_xlim(9., 12.)
        sub.set_ylim(-2., 2.)
        sub.text(0.05, 0.95, r'$%s$' % proplbls[ii], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if ii != 0: sub.set_yticklabels([])
        fig.colorbar(cs0, ax=sub)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'true $\log M_*$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'true $\log \overline{\rm SFR}_{\rm 1Gyr}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ffig = os.path.join(dir_doc, '_msfr_trueprops.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _tauism_bulge(): 
    ''' Figure comparing inferred tau_ism for galaxies separated by bulge
    component contribution.
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas  = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    
    small_bulge = (10**(np.array(thetas['logM_bulge']) - np.array(thetas['logM_total'])) < 0.01) 
    print('%i galaxies with small bulge' % np.sum(small_bulge))
    print('%i galaxies with large bulge' % np.sum(~small_bulge))

    props_infer, props_truth = L2_chains('SP2', derived_properties=True)
    tau_infer = props_infer[-1]
    tau_truth = props_truth[-1]

    
    fig = plt.figure(figsize=(6,6))    
    sub = fig.add_subplot(111) 
   
    prop_bin  = np.arange(0., 2., 0.1) 
    for i, prop_infer, prop_truth in zip(range(2), [tau_infer[small_bulge], tau_infer[~small_bulge]], [tau_truth[small_bulge], tau_truth[~small_bulge]]): 

        x_prop, eta_mu, eta_sig, nbins = [], [], [], [] 
        for ibin in range(len(prop_bin)-1): 
            inbin = (prop_truth > prop_bin[ibin]) & (prop_truth < prop_bin[ibin+1])
            if np.sum(inbin) > 1: 
                nbins.append(np.sum(inbin))
                x_prop.append(0.5 * (prop_bin[ibin] + prop_bin[ibin+1]))
                
                _mu, _sig = PopInf.eta_Delta_opt(prop_infer[inbin,:] - prop_truth[inbin, None])
                eta_mu.append(_mu)
                eta_sig.append(_sig)
        x_prop = np.array(x_prop)
        eta_mu = np.array(eta_mu) 
        eta_sig = np.array(eta_sig) 

        sub.fill_between(x_prop, x_prop + eta_mu - eta_sig, x_prop + eta_mu + eta_sig,
                fc='C%i' % i, ec='none', alpha=[0.3, 0.6][i], label=['small bulge', 'large bulge'][i]) 
        sub.scatter(x_prop, x_prop + eta_mu, c='C%i' % i, s=2) 
        sub.plot(x_prop, x_prop + eta_mu, c='C%i' % i)

    sub.plot([0., 2.], [0., 2.], c='k', ls='--')
    sub.legend(loc='upper left', handletextpad=0.2, fontsize=25) 
    sub.set_xlabel(r'$\tau_{\rm ISM, true}$', fontsize=25) 
    sub.set_xlim(0., 2.)
    sub.set_ylabel(r'$\hat{\tau_{\rm ISM}}$', fontsize=25) 
    sub.set_ylim(0., 2.)
    sub.set_rasterization_zorder(10)

    ffig = os.path.join(dir_doc, '_tauism_bulge.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _l2_photo():
    ''' check the photometric properties of the L2 catalog 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 
    g_mag = 22.5 - 2.5 * np.log10(fluxes[:,0])
    r_mag = 22.5 - 2.5 * np.log10(fluxes[:,1]) 
    z_mag = 22.5 - 2.5 * np.log10(fluxes[:,2]) 

    g_r = g_mag - r_mag
    r_z = r_mag - z_mag 

    low_snr = r_fiber > 20.
    
    ranges  = [(18, 22), (16, 20), (0., 1.5), (0., 1)]
    lbls    = [r'$r_{\rm fiber}$', r'$r$', '$g - r$', '$r - z$']
    props   = [r_fiber, r_mag, g_r, r_z]

    fig = plt.figure(figsize=(16, 16))
    for i in range(4): 
        for j in range(4): 
            if j > i: 
                sub = fig.add_subplot(4, 4, 4*j+i+1)
                sub.scatter(props[i], props[j])
                sub.scatter(props[i][low_snr], props[j][low_snr])
                if j == 3: 
                    sub.set_xlabel(lbls[i], fontsize=25)
                else: 
                    sub.set_xticklabels([])
                if i == 0: 
                    sub.set_ylabel(lbls[j], fontsize=25) 
                else: 
                    sub.set_yticklabels([])
                sub.set_xlim(ranges[i])
                sub.set_ylim(ranges[j]) 
            elif j == i: 
                sub = fig.add_subplot(4, 4, 4*j+i+1)
                _ = sub.hist(props[i], range=ranges[i], bins=20)
                _ = sub.hist(props[i][low_snr], range=ranges[i], bins=20)
                sub.set_xlim(ranges[i])
                sub.set_xticklabels([])
                sub.set_yticklabels([])

    ffig = os.path.join(dir_doc, '_l2_photo.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _l2_props():
    ''' check the photometric properties of the L2 catalog 
    '''
    _, props_truth = L2_chains('SP2', derived_properties=True)

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.scatter(props_truth[0], props_truth[1]) 
    
    ffig = os.path.join(dir_doc, '_l2_prop.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def L2_chains(sample, derived_properties=True): 
    ''' read in posterior chains for L2 mock challenge and derive galaxy
    properties for the chains and the corresponding true properties 
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'

    if derived_properties: 
        f_chain = os.path.join(dat_dir, 'L2', '%s.prop_chains.npy' % sample)
        f_truth = os.path.join(dat_dir, 'L2', '%s.prop_truths.npy' % sample)
    
        prop_infs = np.transpose(np.load(f_chain), (2, 0, 1))
        prop_true = np.load(f_truth).T
        return prop_infs, prop_true
    else:
        f_chain = os.path.join(dat_dir, 'L2', '%s.flat_chains.npy' % sample)
        return np.load(f_chain) 


def model_prior(): 
    ''' figure illustrating model priors 
    '''
    # prior  on SPS parameters
    prior = Infer.load_priors([
        Infer.UniformPrior(7., 12.5, label='sed'),
        Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
        Infer.UniformPrior(0., 1., label='sed'), # burst fraction
        Infer.UniformPrior(1e-2, 13.27, label='sed'), # tburst
        Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-2., 1., label='sed')    # uniform priors on dust_index
    ])

    # declare SPS model  
    m_nmf = Models.NMF(burst=True, emulator=True)
    
    n_sample = 10000 # draw n_sample samples from the prior
    zred = 0.1
    
    # sample prior and evaluate SSFR and Z_MW 
    _thetas = np.array([prior.sample() for i in range(n_sample)])
    thetas = prior.transform(_thetas)
    
    logmstar        = thetas[:,0]
    logsfr_1gyr     = np.log10(m_nmf.avgSFR(thetas, zred=zred, dt=1.))
    logssfr_1gyr    = logsfr_1gyr - logmstar
    logz_mw         = np.log10(m_nmf.Z_MW(thetas, zred=zred))
    tage_mw         = m_nmf.tage_MW(thetas, zred=zred)
    
    #lbls = [r'$\log M_*$', r'$\log {\rm SFR}_{\rm 1 Gyr}$', r'$\log Z_{\rm MW}$', r'$t_{\rm age, MW}$']
    #fig = DFM.corner(
    #        np.array([logmstar, logsfr_1gyr, logz_mw, tage_mw]).T, 
    #        range=[(7., 12.5), (-5, 4.), (-5, -1), (0., 13.2)],
    #        labels=lbls,
    #        label_kwargs={'fontsize': 20},
    #        levels=[0.68, 0.95])#, smooth=True) 
    lbls = [r'$\log M_*$', r'$\log {\rm SSFR}_{\rm 1 Gyr}$', r'$\log Z_{\rm MW}$', r'$t_{\rm age, MW}$']
    fig = DFM.corner(
            np.array([logmstar, logssfr_1gyr, logz_mw, tage_mw]).T, 
            range=[(7., 12.5), (-14, -8.5), (-5, -1), (0., 13.2)],
            labels=lbls,
            label_kwargs={'fontsize': 20},
            levels=[0.68, 0.95])#, smooth=True) 

    ffig = os.path.join(dir_doc, 'model_prior.pdf')
    fig.savefig(ffig, bbox_inches='tight')
    return None 


def Nmock(): 
    props_infer, props_truth = L2_chains('SP2')
    print(props_infer.shape)
    return None 

if __name__=="__main__": 
    #Nmock()
    #BGS()

    #FM_photo()
    #FM_spec()
    
    #_NMF_bases() 

    #nmf_bases()

    #posterior_demo()

    #inferred_props()

    #eta_l2(method='opt')
    #eta_l2(method='mcmc')

    #eta_l2_v2(method='opt')

    #eta_photo_l2(method='opt') 
    #eta_photo_l2(method='mcmc') 

    #eta_rfibmag_l2(method='opt', nmin=10)   

    #eta_color_l2(method='opt', nmin=10)

    #eta_msfr_l2(method='opt', nmin=5)

    #SFH_demo()

    # extra figures 
    #_tauism_bulge()
    
    # high SNR only 
    #eta_l2_highSNR(method='opt')
    #_eta_photo_highSNR()
    #_eta_color_highSNR(method='opt', nmin=10)
    #_eta_msfr_highSNR()

    #_color_photo()
    #_color_trueprops()
    #_msfr_photo()
    #_msfr_trueprops()

    #_eta_msfr_young(method='opt', nmin=10)
    #_eta_msfr_nobulge(method='opt', nmin=10)

    #_l2_photo() 
    #_l2_props()

    model_prior()
