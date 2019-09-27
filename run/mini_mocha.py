'''

scripts to generate data for the mini Mock Challenge (mocha) 


'''
import os 
import h5py 
import numpy as np 
import corner as DFM 
from gqp_mc import util as UT 
from gqp_mc import data as Data 
# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def construct_sample(): 
    ''' construct the mini Mock Challenge photometry, spectroscopy 
    '''
    # select the galaxies of the mini mock challenge
    #Data._mini_mocha_galid() 
    # construct photometry and spectroscopy 
    Data.make_mini_mocha()  
    return None 


def validate_sample(): 
    ''' generate some plots to validate the mini Mock Challenge photometry
    and spectroscopy 
    '''
    # read photometry 
    photo, _ = Data.Photometry(noise='legacy', sample='mini_mocha') 
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
    ffig = os.path.join(UT.dat_dir(), 'mini_mocha', 'mini_mocha.photo.png')
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 

    # read spectra
    spec_s, _ = Data.Spectra(noise='none', sample='mini_mocha') 
    spec_bgs0, _ = Data.Spectra(noise='bgs0', sample='mini_mocha') 
    spec_bgs1, _ = Data.Spectra(noise='bgs1', sample='mini_mocha') 
    spec_bgs2, _ = Data.Spectra(noise='bgs2', sample='mini_mocha') 
        
    # read sky brightness
    _fsky = os.path.join(UT.dat_dir(), 'mini_mocha', 'bgs.exposure.surveysim.150s.v0p4.sample.hdf5') 
    fsky = h5py.File(_fsky, 'r') 
    wave_sky    = fsky['wave'][...] # sky wavelength 
    sbright_sky = fsky['sky'][...]
    
    # spectra validationg 
    fig = plt.figure(figsize=(10,5))     
    sub = fig.add_subplot(111)
    
    for i, spec_bgs in enumerate([spec_bgs2, spec_bgs0]): 
        wsort = np.argsort(spec_bgs['wave'][0,:]) 
        if i == 0: 
            _plt, = sub.plot(spec_bgs['wave'][0,wsort], spec_bgs['flux'][0,wsort], c='C%i' % i, lw=0.25) 
        else:
            sub.plot(spec_bgs['wave'][0,wsort], spec_bgs['flux'][0,wsort], c='C%i' % i, lw=0.25) 
    
    _plt_photo = sub.errorbar([4750, 6350, 9250], [flux_g[0], flux_r[0], flux_z[0]], 
            [ivar_g[0]**-0.5, ivar_r[0]**-0.5, ivar_z[0]**-0.5], fmt='.r') 

    _plt_lgal, = sub.plot(spec_s['wave'], spec_s['flux'][0,:], c='k', ls='-', lw=1) 
    _plt_lgal0, = sub.plot(spec_s['wave'], spec_s['flux_unscaled'][0,:], c='k', ls=':', lw=0.25) 

    leg = sub.legend(
            [_plt_lgal0, _plt_photo, _plt_lgal, _plt], 
            ['LGal spectrum', 'LGal photometry', 'LGal fiber spectrum', 'LGal BGS spectra'],
            loc='upper right', fontsize=17) 
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    sub.set_xlabel('Wavelength [$A$]', fontsize=20) 
    sub.set_xlim(3e3, 1e4) 
    sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$', fontsize=20) 
    sub.set_ylim(-2., 8.) 
    
    ffig = os.path.join(UT.dat_dir(), 'mini_mocha', 'mini_mocha.spectra.png')
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(ffig, pdf=True), bbox_inches='tight') 
    
    return None 


if __name__=="__main__": 
    #construct_sample()
    validate_sample()
