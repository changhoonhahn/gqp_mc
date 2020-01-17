'''

generate plots for the mock challenge paper 


'''
import os 
import h5py 
import numpy as np 
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
from gqp_mc import fitters as Fitters
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


dir_fig = os.path.join(UT.dat_dir(), 'mini_mocha') 


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


def mini_mocha_spec(noise='bgs0', method='ifsps'): 
    ''' Compare properties inferred from forward modeled photometry to input properties
    '''
    # read noiseless Lgal spectra of the spectral_challenge mocks
    specs, meta = Data.Spectra(sim='lgal', noise=noise, lib='bc03', sample='mini_mocha') 

    Mstar_input = meta['logM_fiber'][:97] # total mass 
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
    sub.set_xlabel(r'input $\log~M_{\rm fib.}$', fontsize=25)
    sub.set_xlim(9., 12.) 
    sub.set_ylabel(r'inferred $\log~M_{\rm fib.}$', fontsize=25)
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
    _ffig = os.path.join(dir_fig, 'mini_mocha.%s.specfit.vanilla.noise_%s.png' % (method, noise)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


def mini_mocha_specphoto(noise='bgs0_legacy', method='ifsps'): 
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
    _ffig = os.path.join(dir_fig, 'mini_mocha.%s.specphotofit.vanilla.noise_%s.png' % (method, noise)) 
    fig.savefig(_ffig, bbox_inches='tight') 
    fig.savefig(UT.fig_tex(_ffig, pdf=True), bbox_inches='tight') 
    return None 


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


def Fbestfit_photo(igal, noise='none', dust=False, method='ifsps'): 
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
    if dust: 
        model = 'vanilla'
    else: 
        model = 'dustless_vanilla'

    f_bf = os.path.join(UT.lgal_dir(), 'spectral_challenge', method, 
            'photo.noise_%s.dust_%s.%s.%i.hdf5' % (noise, ['no', 'yes'][dust], model, igal))
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
    #mock_challenge_photo(noise='none', dust=False, method='ifsps')
    #mock_challenge_photo(noise='none', dust=True, method='ifsps')
    #mock_challenge_photo(noise='legacy', dust=False, method='ifsps')
    #mock_challenge_photo(noise='legacy', dust=True, method='ifsps')

    mini_mocha_spec(noise='bgs0', method='ifsps')
    mini_mocha_spec(noise='bgs0', method='pfirefly')
    #mini_mocha_specphoto(noise='bgs0_legacy', method='ifsps')
