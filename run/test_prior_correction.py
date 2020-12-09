import numpy as np 
import corner as DFM 
from scipy.stats import gaussian_kde as gkde
from sklearn.mixture import GaussianMixture as GMix
# --- provabgs --- 
from provabgs import infer as Infer
from provabgs import flux_calib as FluxCalib
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


n_sample = 50000

# redshift
z = 0.1


def bestfit_gmm(x, max_comp=10): 
    # fit GMMs with a range of components 
    ncomps = range(1, max_comp+1)
    gmms, bics = [], []  
    for i_n, n in enumerate(ncomps): 
        bics.append(gmm.bic(x.T)) # bayesian information criteria
        gmms.append(gmm)

    # components with the lowest BIC (preferred)
    i_best = np.array(bics).argmin()
    print(ncomps[i_best]) # number of components of the best-fit
    gbest = gmms[i_best] # best fit GMM 
    return gbest


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

# ------------------------------------------------------------
# get prior correction 
# 1. sample prior 
theta_prior = np.array([desi_mcmc.prior.sample() for i in range(n_sample)]) 

# 2. compute the derived properties we want to impose flat priors on  
logm_prior      = theta_prior[:,0] 
logsfr_prior    = np.log10(desi_mcmc.model.avgSFR(theta_prior, z, dt=1.))
logzmw_prior    = np.log10(desi_mcmc.model.Z_MW(theta_prior, z))
prop_prior      = np.array([logm_prior, logsfr_prior, logzmw_prior])

# 3. fit a joint distirbution of the derived properties 
kde_fit = gkde(prop_prior) 
gmm_fit = GMix(n_components=20)  
gmm_fit.fit(prop_prior.T)

kde_samples = kde_fit.resample(10000)
_gmm_samples, _ = gmm_fit.sample(10000)
gmm_samples = _gmm_samples.T

fig = DFM.corner(
    prop_prior.T,
    hist_kwargs={'density': True}
    ) 
_ = DFM.corner(
    gmm_samples.T, 
    color='C0',
    hist_kwargs={'density': True},
    fig=fig
    )
_ = DFM.corner(
    kde_samples.T, 
    color='C1',
    hist_kwargs={'density': True},
    fig=fig
    )
fig.savefig('test_prior_correction.fits.png', bbox_inches='tight') 
plt.close()


# test thetas
theta_prior_test = np.array([desi_mcmc.prior.sample() for i in range(n_sample)]) 
logm_prior_test     = theta_prior_test[:,0] 
logsfr_prior_test   = np.log10(desi_mcmc.model.avgSFR(theta_prior_test, z, dt=1.))
logzmw_prior_test   = np.log10(desi_mcmc.model.Z_MW(theta_prior_test, z))

# 4. calculate weights
prop_prior_test = np.array([logm_prior_test, logsfr_prior_test, logzmw_prior_test])

p_prop_kde = kde_fit.pdf(prop_prior_test)
p_prop_gmm = np.exp(gmm_fit.score_samples(prop_prior_test.T)) 
w_prior_corr_kde = 1./p_prop_kde
w_prior_corr_kde[p_prop_kde < 1e-4] = 0. 
w_prior_corr_gmm = 1./p_prop_gmm
w_prior_corr_gmm[p_prop_gmm < 1e-4] = 0. 

fig = DFM.corner(
    prop_prior_test.T,
    hist_kwargs={'density': True}
    ) 
_ = DFM.corner(
    prop_prior_test.T, 
    weights=w_prior_corr_gmm, 
    color='C0',
    hist_kwargs={'density': True},
    fig=fig
    )
_ = DFM.corner(
    prop_prior_test.T, 
    weights=w_prior_corr_kde, 
    color='C1',
    hist_kwargs={'density': True},
    fig=fig
    )
fig.savefig('test_prior_correction.png', bbox_inches='tight') 
plt.close()

