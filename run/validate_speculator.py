import os 
import pickle
import numpy as np
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

from speculator import SpectrumPCA
from speculator import Speculator


n_param = 10
n_pcas  = [50, 30, 30]
N_trains = [int(5e6), int(5e6), int(5e6)]
architectures = ['', '', '']

dat_dir = os.path.join(os.environ['GQPMC_DIR'], 'speculator') 

waves = np.load(os.path.join(dat_dir, 'wave_fsps.npy')) 
wave_bins = [(waves< 4500), (waves>= 4500) & (waves< 6500), (waves>= 6500)]


# plot the loss function 
floss = os.path.join(dat_dir, 'DESI_complexdust_model.Ntrain%i.wave_bin%i.pca%i.%sloss.dat' % (N_trains[0], 0, n_pcas[0], architectures[0]))
loss = np.loadtxt(floss) 

fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
sub.plot(np.arange(loss.shape[0]), loss[:,2], c='k')

for batchsize in np.unique(loss[:,0]):
  isbatch = (loss[:,0] == batchsize)
  sub.plot(np.arange(loss.shape[0])[isbatch], loss[:,2][isbatch], label='%i' % batchsize)
sub.legend(loc='upper right')
sub.set_ylabel('loss', fontsize=25)
sub.set_yscale('log')
sub.set_ylim(1e-3, 0.1)
sub.set_xlabel('Epochs', fontsize=25)
sub.set_xlim(0, loss.shape[0])
fig.savefig(floss.replace(".dat", '.png'), bbox_inches='tight') 


# read in training parameters and data
theta_test = np.load(os.path.join(dat_dir, 'DESI_complexdust.theta_test.npy')) 
logspectrum_test = np.load(os.path.join(dat_dir, 'DESI_complexdust.logspectrum_fsps_test.npy')) 
spectrum_test = 10**logspectrum_test

# read in trained speculator for all three wavebins 
fspeculator = [os.path.join(dat_dir, '_DESI_complexdust_model.Ntrain%i.wave_bin%i.pca%i.%slog' % (N_train, n_wave, n_pcas[n_wave], architectures[n_wave])) for n_wave, N_train in enumerate(N_trains)]
speculators = [Speculator(restore=True, restore_filename=_f) for _f in fspeculator] 


def combined_speculator(theta):
    spectrum_spec = []
    for iwave in range(len(wave_bins)):
        spectrum_spec.append(speculators[iwave].log_spectrum(theta))
    return np.concatenate(spectrum_spec, axis=1)

# speculator prediction of log(spectrum) 
logspectrum_spec = combined_speculator(theta_test.astype(np.float32))

# fractional error of the Speculator model
frac_dspectrum = 1. - 10**(logspectrum_spec - logspectrum_test)
frac_dspectrum_quantiles = np.nanquantile(frac_dspectrum, 
        [0.0005, 0.005, 0.025, 0.16, 0.84, 0.975, 0.995, 0.9995], axis=0)

fig = plt.figure(figsize=(15,5))
for iwave in range(len(wave_bins)):
    sub = fig.add_subplot(1,3,iwave+1)
    sub.fill_between(waves, 
            frac_dspectrum_quantiles[0],
            frac_dspectrum_quantiles[-1], fc='C0', ec='none', alpha=0.1, label='99.9%')
    sub.fill_between(waves, 
            frac_dspectrum_quantiles[1],
            frac_dspectrum_quantiles[-2], fc='C0', ec='none', alpha=0.2, label='99%')
    sub.fill_between(waves, 
            frac_dspectrum_quantiles[2],
            frac_dspectrum_quantiles[-3], fc='C0', ec='none', alpha=0.3, label='95%')
    sub.fill_between(waves, 
            frac_dspectrum_quantiles[3],
            frac_dspectrum_quantiles[-4], fc='C0', ec='none', alpha=0.5, label='68%')
    sub.plot(waves, 0.01 * np.ones(len(waves)), c='k', ls='--', lw=0.5)
    sub.plot(waves, -0.01 * np.ones(len(waves)), c='k', ls='--', lw=0.5)
    if iwave == len(wave_bins) - 1: sub.legend(loc='upper right', fontsize=20)
    if iwave == 0: sub.set_xlim(2.3e3, 4.5e3)
    elif iwave == 1:
        sub.set_xlabel('wavelength ($A$)', fontsize=25)
        sub.set_xlim(4.5e3, 6.5e3)
    elif iwave == 2:
        sub.set_xlim(6.5e3, 1e4)
    if iwave == 0: sub.set_ylabel(r'$(f_{\rm speculator} - f_{\rm fsps})/f_{\rm fsps}$', fontsize=25)
    sub.set_ylim(-0.1, 0.1)
fig.savefig(os.path.join(dat_dir,
    'DESI_complexdust_model.Ntrain%i.wave_bin0.pca%i.%sfrac_err.png' %
    (N_trains[0], n_pcas[0], architectures[0])), bbox_inches='tight')


dlogspectrum = logspectrum_spec - logspectrum_test
dlogspectrum_quantiles = np.nanquantile(dlogspectrum, 
        [0.0005, 0.005, 0.025, 0.16, 0.84, 0.975, 0.995, 0.9995], axis=0)

fig = plt.figure(figsize=(15,5))
for iwave in range(len(wave_bins)):
    sub = fig.add_subplot(1,3,iwave+1)
    sub.fill_between(waves,
            dlogspectrum_quantiles[0],
            dlogspectrum_quantiles[-1],
            fc='C0', ec='none', alpha=0.1, label='99.9%')
    sub.fill_between(waves,
            dlogspectrum_quantiles[1],
            dlogspectrum_quantiles[-2],
            fc='C0', ec='none', alpha=0.2, label='99%')
    sub.fill_between(waves,
            dlogspectrum_quantiles[2],
            dlogspectrum_quantiles[-3],
            fc='C0', ec='none', alpha=0.3, label='95%')
    sub.fill_between(waves,
            dlogspectrum_quantiles[3],
            dlogspectrum_quantiles[-4],
            fc='C0', ec='none', alpha=0.5, label='68%')
    if iwave == len(wave_bins) - 1: sub.legend(loc='upper right', fontsize=20)
    if iwave == 0: sub.set_xlim(2.3e3, 4.5e3)
    elif iwave == 1:
        sub.set_xlabel('wavelength ($A$)', fontsize=25)
        sub.set_xlim(4.5e3, 6.5e3)
    elif iwave == 2: sub.set_xlim(6.5e3, 1e4)
    if iwave == 0: sub.set_ylabel(r'$\log f_{\rm speculator} - \log f_{\rm fsps}$', fontsize=25)
    sub.set_ylim(-0.1, 0.1)
fig.savefig(os.path.join(dat_dir,
    'DESI_complexdust_model.Ntrain%i.wave_bin0.pca%i.%slog_err.png' %
    (N_trains[0], n_pcas[0], architectures[0])))


mean_frac_dspectrum = np.mean(np.abs(frac_dspectrum), axis=1)
quant = np.quantile(mean_frac_dspectrum, [0.68, 0.95, 0.99, 0.999])
fig = plt.figure(figsize=(8,6))
sub = fig.add_subplot(111)
for q, a in zip(quant[::-1], [0.1, 0.2, 0.3, 0.5]): 
  sub.fill_between([0., q], [0., 0.], [1., 1.], alpha=a, color='C0')
_ = sub.hist(mean_frac_dspectrum, 40, density=True, histtype='step', cumulative=True, color='k')
sub.set_xlabel(r'${\rm mean}_\lambda [ (f_{\rm speculator}  - f_{\rm fsps}) / f_{\rm fsps} ]$', fontsize=20)
sub.set_xlim(0., 0.03)
sub.set_ylabel('cumulative distribution', fontsize=20)
sub.set_ylim(0., 1.)
fig.savefig(os.path.join(dat_dir,
    'DESI_complexdust_model.Ntrain%i.wave_bin0.pca%i.%smean_frac_err_dist.png' %
    (N_trains[0], n_pcas[0], architectures[0])), bbox_inches='tight')

