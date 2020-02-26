'''

script to test the iSpeculator implementation  


'''
import os 
import fsps 
import numpy as np 
from gqp_mc import util as UT 
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


def FSPS_sspLum(theta): 
    ''' FSPS wrapper that deals with NMF SFH and ZH basis and other parameters.

    :param theta:
        numpy array that specifies parameters.
        Indices 0 to Ncomp_sfh-1 specifies the SFH basis parameters.
        Indices Ncomp_sfh to Ncomp_sfh + Ncomp_zh specifies the ZH basis parameters.
        Index -1 specifes tau_ISM (dust)

    :return wave_rest
        rest-frame wavelength grid provided by FSPS

    :return lum_ssp:
        luminosity in uints of Lsun/AA of ssp. This can be converted to observed flux
        of units erg/s/cm^2/Angstrom by multiplying x Lsun/(4pi dlum^2)/(1+z)
    '''
    # initalize fsps object
    ssp = fsps.StellarPopulation(
        zcontinuous=1, # SSPs are interpolated to the value of logzsol before the spectra and magnitudes are computed
        sfh=0, # single SSP
        imf_type=1, # chabrier
        dust_type=2 # Calzetti (2000)
        )

    ispec = Fitters.iSpeculator() 
    t_lookback      = ispec._nmf_t_lookback
    nmf_sfh_basis   = ispec._nmf_sfh_basis
    nmf_zh_basis    = ispec._nmf_zh_basis 

    theta_sfh = theta[:4]
    theta_zh = theta[4:6]
    theta_dust = theta[-1] # dust parameter
    sfh = np.dot(theta_sfh, nmf_sfh_basis)
    zh = np.dot(theta_zh, nmf_zh_basis)

    for i, t, m, z in zip(range(len(t_lookback)), t_lookback, sfh, zh):
        if m <= 0: # no star formation in this bin
            continue
        ssp.params['logzsol'] = np.log10(z/0.0190) # log(Z/Zsun)
        ssp.params['dust2'] = theta_dust
        wave_rest, lum_i = ssp.get_spectrum(tage=t, peraa=True) # in units of Lsun/AA
        if i == 0: lum_ssp = np.zeros(len(wave_rest))
        lum_ssp += m * lum_i
    return wave_rest, lum_ssp


def Speculator_sspLum(theta): 
    '''
    '''
    ispec = Fitters.iSpeculator() 
    sspLum = ispec._emulator(theta)
    return ispec._emu_wave, sspLum


def compare_FSPS_Speculator(): 
    '''
    '''
    tt_i = np.array([0.25, 0.25, 0.25, 0.25, 1e-4, 1e-4, 0., 1.5])

    w0, lum0 = FSPS_sspLum(tt_i)
    w1, lum1 = Speculator_sspLum(tt_i) 
    wlim0 = (w0 > 3e3) 
    wlim1 = (w1 > 3e3) 
    print(np.sum(lum0))
    print(np.sum(lum1))
    print(lum0[wlim0]) 
    print(lum1[wlim1]) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.plot(w0, lum0, c='k', label='FSPS') 
    sub.plot(w1, lum1, c='C0', label='Speculator') 
    sub.set_xlim(3e3, 1e4) 
    ffig = os.path.join(UT.dat_dir(), '_fsps_speculator_comparison.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


if __name__=="__main__":
    compare_FSPS_Speculator()
