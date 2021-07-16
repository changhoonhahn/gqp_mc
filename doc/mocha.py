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
    fig = plt.figure(figsize=(15,5))
    gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.75)
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
    gs2 = fig.add_gridspec(nrows=1, ncols=1, wspace=0.05, left=0.75, right=0.95)
    sub = plt.subplot(gs2[0,0])
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
    sub.legend(plts, ['DESI', 'SDSS', 'GAMA'], loc='upper left', handletextpad=0.2, prop={'size': 20}) 
    ffig = os.path.join(dir_doc, 'bgs.pdf')
    fig.savefig(ffig, bbox_inches='tight')

    raise ValueError
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
    photo, meta = Data.Photometry(sim='lgal', noise='legacy', lib='fsps', sample='mini_mocha')
    flux_g = photo['flux'][:,0] * 1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs())
    flux_r = photo['flux'][:,1] * 1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs())
    flux_z = photo['flux'][:,2] * 1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()) # convert to 10^-17 ergs/s/cm^2/Ang
    ivar_g = photo['ivar'][:,0] * (1e-9 * 1e17 * UT.c_light() / 4750.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_r = photo['ivar'][:,1] * (1e-9 * 1e17 * UT.c_light() / 6350.**2 * (3631. * UT.jansky_cgs()))**-2.
    ivar_z = photo['ivar'][:,2] * (1e-9 * 1e17 * UT.c_light() / 9250.**2 * (3631. * UT.jansky_cgs()))**-2. # convert to 10^-17 ergs/s/cm^2/Ang

    # read noiseless Lgal spectroscopy 
    specs, _ = Data.Spectra(sim='lgal', noise='none', lib='fsps', sample='mini_mocha') 
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
            ['LGal SED',  'forward modeled photometry', 'broadband filter response'], 
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
            fmt='.C3', markersize=5)#, label='forward modeled DESI photometry') 
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
    # read noiseless Lgal spectroscopy 
    spec_s, meta    = Data.Spectra(sim='lgal', noise='none', lib='fsps', sample='mini_mocha') 
    spec_bgs, _     = Data.Spectra(sim='lgal', noise='bgs', lib='fsps', sample='mini_mocha') 
    
    fig = plt.figure(figsize=(12,4))
    sub = fig.add_subplot(111) 
    
    for i, band in enumerate(['b', 'r', 'z']): 
        if band == 'b': 
            _plt, = sub.plot(spec_bgs['wave_%s' % band], spec_bgs['flux_%s' % band][0],
                c='C%i' % i, lw=0.25) 
        else: 
            sub.plot(spec_bgs['wave_%s' % band], spec_bgs['flux_%s' % band][0],
                c='C%i' % i, lw=0.25) 
    
    _plt_lgal, = sub.plot(spec_s['wave'], spec_s['flux'][0,:], c='k', ls='--', lw=1) 
    _plt_lgal0, = sub.plot(spec_s['wave'], spec_s['flux_unscaled'][0,:], c='k', ls=':', lw=1) 
    
    leg = sub.legend(
            [_plt, _plt_lgal, _plt_lgal0], 
            ['forward modeled spectrum', 'fiber fraction scaled SED', 'LGal SED'],
            loc='upper right', handletextpad=0.3, fontsize=17) 
    sub.set_xlabel('wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3500, 1.05e4)
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=20) 
    sub.set_ylim(0., 3.*(spec_s['flux_unscaled'][0].max()))

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
            r'$\tau_{\rm ISM}$']
    minmax  = [[8., 12.], [-4., 1], [-2.5, -1.5], [0., 2.]]
    widths  = [0.04, 0.04, 0.01, 0.03]

    fig = plt.figure(figsize=(20, 15))    
    for i, sample in enumerate(['S2', 'P2', 'SP2']): 
        props_infer, props_truth = L2_chains(sample, derived_properties=True)
        
        for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 

            sub = fig.add_subplot(3,4,4*i+ii+1)
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
                violin.set_alpha(0.5)

            sub.plot(minmax[ii], minmax[ii], c='k', ls='--')
            sub.set_xlim(minmax[ii])
            sub.set_ylim(minmax[ii])
            if ii == 0: sub.set_yticks([8., 9., 10., 11., 12.])
            if ii == 1: sub.set_yticks([-4., -3., -2., -1., 0., 1.])
            if ii == 3: sub.set_yticks([0., 0.5, 1.0, 1.5, 2.]) 
            if ii == 0: sub.text(0.95, 0.05, ['spectra', 'photometry', 'spectra+photometry'][i], 
                    ha='right', va='bottom', transform=sub.transAxes, fontsize=25)
            if i != 2: sub.set_xticklabels([])
            if i == 0: sub.set_title(lbls[ii], pad=10, fontsize=25)

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
    lbls    = [r'$\log M_*$', r'$\log \overline{\rm SFR}_{\rm 1Gyr}$', r'$\log Z_{\rm MW}$', r'$\tau_{\rm ISM}$']
    minmax  = [[9., 12.], [-3., 1], [-2.3, -1.7], [0., 1.]]
    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 3))

    #for ii, sample in enumerate(['S2', 'P2', 'SP2']): 
    for ii, sample, clr in zip(range(2), ['P2', 'SP2'], ['C1', 'C2']): 
        props_infer, props_truth = L2_chains(sample)
        
        # get eta for log M*, log SFR, and log Z_MW
        logM_bin    = np.arange(8, 13, 0.2)
        logSFR_bin  = np.arange(-4, 1, 0.5)
        logZMW_bin  = np.arange(-2.3, -1.7, 0.05)
        tauism_bin  = np.arange(0., 1., 0.1) 
        
        x_props, eta_mus, eta_sigs = [], [], []
        for prop_infer, prop_truth, prop_bin in zip(props_infer, props_truth, [logM_bin, logSFR_bin, logZMW_bin, tauism_bin]): 

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
            sub = fig.add_subplot(1, len(x_props), i+1) 

            sub.plot(minmax[i], [0., 0.], c='k', ls='--')
            sub.fill_between(x_prop, eta_mu - eta_sig, eta_mu + eta_sig,
                    fc=clr, ec='none', alpha=[0.3, 0.6][ii], label=['photometry', 'spectra+photometry'][ii]) 
            sub.scatter(x_prop, eta_mu, c=clr, s=2) 
            sub.plot(x_prop, eta_mu, c=clr)
            if ii == 0: sub.set_xlabel(lbls[i], fontsize=25)
            sub.set_xlim(minmax[i])
            if i == 0: 
                sub.set_ylabel(r'$\Delta_\theta$ [dex]', fontsize=30)
            if i != 0: sub.set_yticklabels([]) 
            if i == 0: sub.legend(loc='upper left', handletextpad=0.2, fontsize=20)
            sub.set_ylim(-1., 1.) 
            #if i == 3: sub.text(0.95, 0.05, ['spectra', 'photometry', 'spectra + photometry'][ii], 
            #        ha='right', va='bottom', transform=sub.transAxes, fontsize=25)

        #sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(dir_doc, 'etas.pdf')
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

    proplbls = [r'\log M_*', r'\log {\rm SFR}_{1Gyr}', r'\log Z_{\rm MW}', r'\tau_{\rm ISM}']
    lbls    = [r'$r_{\rm fiber}$', r'$r$', r'$g-r$', r'$r-z$']
    minmax  = [[17., 22.], [15., 20], [0, 1.], [0., 1.]]
    dbin    = [0.5, 0.5, 0.1, 0.1]

    photos  = [r_fiber, r_mag, g_r, r_z]

    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 12))    

    props_infer, props_truth = L2_chains('SP2')
        
    for ii, prop_infer, prop_truth in zip(range(len(props_infer)), props_infer, props_truth): 
        print(proplbls[ii]) 
        # get eta for different photometry bins 
        photo_bins = [np.arange(minmax[ii][0], minmax[ii][1], dbin[ii]) for ii in range(len(minmax))]
        
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
            if ii == 3: sub.set_xlabel(lbls[i], fontsize=25)
            else: sub.set_xticklabels([]) 
            sub.set_xlim(minmax[i])
            if i == 0: sub.set_ylabel(r'$\Delta_{%s}$' % proplbls[ii], fontsize=30)
            if i != 0: sub.set_yticklabels([]) 
            sub.set_ylim(-1., 1.) 
        #sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(dir_doc, 'etas_photo.pdf')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def eta_l2_highSNR(method='opt'):
    ''' calculate bias as a function of galaxy properties for high SNR galaxies
    '''
    dat_dir = '/Users/chahah/data/gqp_mc/mini_mocha/'
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))
    fluxes = np.load(os.path.join(dat_dir, 'mocha_p2.flux.npy'))

    r_fiber = 22.5 - 2.5 * np.log10(thetas['f_fiber_meas'] * fluxes[:,1]) 

    high_snr = (r_fiber < 20.5) 
    print(np.sum(high_snr), len(r_fiber))

    lbls    = [r'$\log M_*$', r'$\log \overline{\rm SFR}_{\rm 1Gyr}$', r'$\log Z_{\rm MW}$', r'$\tau_{\rm ISM}$']
    minmax  = [[9., 12.], [-3., 1], [-2.3, -1.7], [0., 1.]]
    # eta as a function of galaxy properties 
    fig = plt.figure(figsize=(20, 3))

    #for ii, sample in enumerate(['S2', 'P2', 'SP2']): 
    for ii, sample, clr in zip(range(2), ['P2', 'SP2'], ['C1', 'C2']): 
        props_infer, props_truth = L2_chains(sample)
        
        # get eta for log M*, log SFR, and log Z_MW
        logM_bin    = np.arange(8, 13, 0.2)
        logSFR_bin  = np.arange(-4, 1, 0.5)
        logZMW_bin  = np.arange(-2.3, -1.7, 0.05)
        tauism_bin  = np.arange(0., 1., 0.1) 
        
        x_props, eta_mus, eta_sigs = [], [], []
        for prop_infer, prop_truth, prop_bin in zip(props_infer, props_truth, [logM_bin, logSFR_bin, logZMW_bin, tauism_bin]): 

            x_prop, eta_mu, eta_sig, nbins = [], [], [], [] 
            for ibin in range(len(prop_bin)-1): 
                inbin = high_snr & (prop_truth > prop_bin[ibin]) & (prop_truth < prop_bin[ibin+1])
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
            sub = fig.add_subplot(1, len(x_props), i+1) 

            sub.plot(minmax[i], [0., 0.], c='k', ls='--')
            sub.fill_between(x_prop, eta_mu - eta_sig, eta_mu + eta_sig,
                    fc=clr, ec='none', alpha=[0.3, 0.6][ii], label=['photometry', 'spectra+photometry'][ii]) 
            sub.scatter(x_prop, eta_mu, c=clr, s=2) 
            sub.plot(x_prop, eta_mu, c=clr)
            if ii == 0: sub.set_xlabel(lbls[i], fontsize=25)
            sub.set_xlim(minmax[i])
            if i == 0: 
                sub.set_ylabel(r'$\Delta_\theta$ [dex]', fontsize=30)
            if i != 0: sub.set_yticklabels([]) 
            if i == 0: sub.legend(loc='upper left', handletextpad=0.2, fontsize=20)
            sub.set_ylim(-1., 1.) 
            #if i == 3: sub.text(0.95, 0.05, ['spectra', 'photometry', 'spectra + photometry'][ii], 
            #        ha='right', va='bottom', transform=sub.transAxes, fontsize=25)

        #sub.legend(loc='upper right', fontsize=20, handletextpad=0.2) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(dir_doc, '_etas_highsnr.png')
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
    thetas = pickle.load(open(os.path.join(dat_dir, 'l2.theta.p'), 'rb'))

    # read in MCMC chains 
    if sample == 'S2': 
        f_chain = lambda i: os.path.join(dat_dir, 'L2', 'S2.provabgs_model.%i.chain.p' % i)
    elif sample == 'P2': 
        f_chain = lambda i: os.path.join(dat_dir, 'L2', 'P2.provabgs.%i.chain.p' % i)
    elif sample == 'SP2': 
        f_chain = lambda i: os.path.join(dat_dir, 'L2', 'SP2.provabgs.%i.chain.p' % i)

    
    igals, chains = [], []
    for i in range(100): 
        if os.path.isfile(f_chain(i)): 
            igals.append(i)
            chains.append(pickle.load(open(f_chain(i), 'rb')))
    print('%i of 100 galaxies in the mock challenge' % len(igals)) 
    
    # provabgs model 
    m_nmf = Models.NMF(burst=True, emulator=True)
    
    if derived_properties: 
        # derived 
        logMstar_true, logMstar_inf = [], [] 
        logSFR_true, logSFR_inf     = [], [] 
        logZ_MW_true, logZ_MW_inf   = [], []
        tau_ism_true, tau_ism_inf   = [], [] 

        for i, chain in zip(igals, chains): 
            if sample != 'SP2':
                flat_chain = UT.flatten_chain(chain['mcmc_chain'][1500:,:,:])       
            else: 
                flat_chain = UT.flatten_chain(chain['mcmc_chain'][1500:,:,:-1])

            z_obs = thetas['redshift'][i]
            
            if sample == 'S2': 
                logMstar_true.append(thetas['logM_fiber'][i])
            else: 
                logMstar_true.append(thetas['logM_total'][i])
            logMstar_inf.append(flat_chain[:,0])
            
            if sample == 'S2':
                logSFR_true.append(np.log10(thetas['sfr_1gyr'][i]) - (thetas['logM_total'][i] - thetas['logM_fiber'][i]))
            else: 
                logSFR_true.append(np.log10(thetas['sfr_1gyr'][i]))
            logSFR_inf.append(np.log10(m_nmf.avgSFR(flat_chain, zred=z_obs, dt=1.0)))
            
            logZ_MW_true.append(np.log10(thetas['Z_MW'])[i])
            logZ_MW_inf.append(np.log10(m_nmf.Z_MW(flat_chain, zred=z_obs)))

            tau_ism_true.append(thetas['tau_ism'][i])
            tau_ism_inf.append(flat_chain[:,-2])
            
        logMstar_true   = np.array(logMstar_true)
        logMstar_inf    = np.array(logMstar_inf)

        logSFR_true     = np.array(logSFR_true).flatten()
        logSFR_inf      = np.array(logSFR_inf)

        logZ_MW_true    = np.array(logZ_MW_true).flatten()
        logZ_MW_inf     = np.array(logZ_MW_inf)

        tau_ism_true    = np.array(tau_ism_true)
        tau_ism_inf     = np.array(tau_ism_inf)

        props_true      = np.array([logMstar_true, logSFR_true, logZ_MW_true, tau_ism_true])
        props_inf       = np.array([logMstar_inf, logSFR_inf, logZ_MW_inf, tau_ism_inf])

        return props_inf, props_true
    else:
        flat_chains = [] 
        for i, chain in zip(igals, chains): 
            flat_chain = UT.flatten_chain(chain['mcmc_chain'][1500:,:,:])       
            flat_chains.append(flat_chain)
        return np.array(flat_chains)


def model_prior(): 
    ''' figure illustrating model priors 
    '''
    # prior  on SPS parameters
    prior = Infer.load_priors([
        Infer.UniformPrior(7., 12.5, label='sed'),
        Infer.FlatDirichletPrior(4, label='sed'),   # flat dirichilet priors
        Infer.UniformPrior(0., 1., label='sed'), # burst fraction
        Infer.UniformPrior(1e-2, 13.27, label='sed'), # tburst
        Infer.LogUniformPrior(4.5e-5, 4.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.LogUniformPrior(4.5e-5, 4.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-3., 1., label='sed')    # uniform priors on dust_index
    ])

    # declare SPS model  
    m_nmf = Models.NMF(burst=True, emulator=True)
    
    n_sample = 10000 # draw n_sample samples from the prior
    zred = 0.1
    
    # sample prior and evaluate SSFR and Z_MW 
    _thetas = np.array([prior.sample() for i in range(n_sample)])
    thetas = prior.transform(_thetas)
    
    logmstar        = thetas[:,0]
    logssfr_1gyr    = np.log10(m_nmf.avgSFR(thetas, zred=zred, dt=1.)) - logmstar
    logz_mw         = np.log10(m_nmf.Z_MW(thetas, zred=zred))

    
    lbls = [r'$\log M_*$', r'$\log {\rm SSFR}_{\rm 1 Gyr}$', r'$\log Z_{\rm MW}$']
    fig = DFM.corner(
            np.array([logmstar, logssfr_1gyr, logz_mw]).T, 
            range=[(7., 12.5), (-14, -8.5), (-5, -1)],
            labels=[r'$\log M_*$', r'$\log \overline{\rm SSFR}_{\rm 1 Gyr}$', r'$\log Z_{\rm MW}$'], 
            label_kwargs={'fontsize': 20}) 

    ffig = os.path.join(dir_doc, 'model_prior.pdf')
    fig.savefig(ffig, bbox_inches='tight')
    return None 


if __name__=="__main__": 
    #BGS()

    #FM_photo()
    #FM_spec()
    
    #_NMF_bases() 

    #nmf_bases()

    #posterior_demo()

    #inferred_props()

    #eta_l2(method='opt')
    #eta_l2(method='mcmc')

    #eta_photo_l2(method='opt') 
    #eta_photo_l2(method='mcmc') 
    
    # eta for high SNR only 
    #eta_l2_highSNR(method='opt')

    #_l2_photo() 
    _l2_props()

    #model_prior()
