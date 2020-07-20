import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
from gqp_mc import data as Data 
from gqp_mc import fitters as Fitters
import warnings
warnings.filterwarnings("ignore")

specs,meta = Data.Spectra(sim='lgal',noise = 'none', lib = 'bc03', sample = 'mini_mocha')
# photo, _ = Data.Photometry(sim='lgal', noise= 'none', lib='bc03', sample='mini_mocha') 
True_logM_total = meta['logM_total']
True_logsfr_100myr = np.log10(meta['sfr_100myr'])
True_logsfr_1gyr = np.log10(meta['sfr_1gyr'])
True_logz_mw = np.log10(meta['Z_MW'])
True_logM_fiber = meta['logM_fiber']

class Doctor():
    def __init__(self):
        pass

    def _diagnose(self,mcmc_data,tt_names,param,gal_idx):
        if param == 'logmstar':
            truth = True_logM_total[gal_idx]
        elif param == 'logsfr.100myr':
            truth = True_logsfr_100myr[gal_idx]
        elif param == 'logsfr.1gyr':
            truth = True_logsfr_1gyr[gal_idx]
        elif param == 'logz.mw':
            truth = True_logz_mw[gal_idx]
        elif param == 'f_fiber':
            truth = (10**True_logM_fiber[gal_idx]) / (10**True_logM_total[gal_idx])
        else:
            raise ValueError

        data = mcmc_data[:,tt_names.index(param)]
        frac_err = (np.median(data)-truth)/truth
        err_up = (np.percentile(data,84.1)-truth)/truth- frac_err
        err_lo = frac_err - (np.percentile(data,15.9)-truth)/truth

        return frac_err,err_up,err_lo

    def diagnose(self,data_dir,sim,spec_or_photo,noise,model,igals,dummy_extra):
        for igal in igals:
            f_mcmc = f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{igal}.mcmc.hdf5'
            f_sfr = f_mcmc.replace('mcmc','postproc')
            f_dir = os.path.join(data_dir,f_sfr)

            if os.path.exists(f_dir):
                gal_idx = int(igal)
                f = h5py.File(f_dir,'r')
                tt_names = list(f['theta_names'][...].astype(str))
                mcmc_data = f['mcmc_chain'][...]
                fig, ax = plt.subplots(1,1, figsize = (7,7))
                logmstar_fracerr,logmstar_err_up, logmstar_err_lo = self._diagnose(mcmc_data,tt_names,'logmstar',gal_idx) 
                logsfr_100myr_fracerr,logsfr_100myr_err_up,logsfr_100myr_err_lo = self._diagnose(mcmc_data,tt_names,'logsfr.100myr',gal_idx) 
                logsfr_1gyr_fracerr,logsfr_1gyr_err_up,logsfr_1gyr_err_lo = self._diagnose(mcmc_data,tt_names,'logsfr.1gyr',gal_idx) 
                logz_mw_fracerr,logz_mw_err_up,logz_mw_err_lo = self._diagnose(mcmc_data,tt_names,'logz.mw',gal_idx)
                tt_fracerr = [logmstar_fracerr,logsfr_100myr_fracerr,logsfr_1gyr_fracerr,logz_mw_fracerr]
                tt_err_up = [logmstar_err_up,logsfr_100myr_err_up,logsfr_1gyr_err_up,logz_mw_err_up]
                tt_err_lo = [logmstar_err_lo,logsfr_100myr_err_lo,logsfr_1gyr_err_lo,logz_mw_err_lo]
                tick_labels = ['logmstar', 'logsfr.100myr','logsfr.1gyr','logz.mw']
                ticks = [0,1,2,3]

                if spec_or_photo =='specphoto':
                    f_fiber_fracerr,f_fiber_err_up,f_fiber_err_lo = self._diagnose(mcmc_data,tt_names,'f_fiber',gal_idx)
                    tt_fracerr.append(f_fiber_fracerr)
                    tt_err_up.append(f_fiber_err_up)
                    tt_err_lo.append(f_fiber_err_lo)
                    tick_labels.append('f_fiber')
                    ticks.append(4)
                print(tt_fracerr)

                ax.errorbar(np.arange(len(tt_fracerr)),tt_fracerr,yerr = (tt_err_up,tt_err_lo), fmt = 'ok', elinewidth = 1, capsize =2)
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels)
                ax.grid()
                save_base_dir = os.path.join(os.environ.get('GQPMC'),'doctor','data_list','diagnostic',f'{model}')
                if not os.path.exists(save_base_dir):
                    os.makedirs(save_base_dir)

                save_dir = os.path.join(save_base_dir,f_sfr.replace('postproc.hdf5','diagnostic.pdf'))
                ax.set_xlabel('Parameters')
                ax.set_ylabel('Fractional Error')
                ax.set_title('Parameter Diagonstics')
                fig.savefig(save_dir, format = 'pdf')

            else:
                print(f'igal{igal} not found')

    def _get_data(self,f_dir,param_idx,mcmc_postproc,all_data=False,perc=50):
        f = h5py.File(f_dir,'r')
        keys = list(f.keys())
        if mcmc_postproc == 'mcmc':
            num = -1
            
            for k in keys:
                if 'mcmc_chain' in k:
                    num += 1
            
            if not all_data:
                mcmc_data = f[f'mcmc_chain{num}'][...][:,:,param_idx]
            
            else:
                mcmc_data = np.array([])
                for idx in range(num+1):
                    mcmc_data = np.append(mcmc_data,f[f'mcmc_chain{idx}'][:,:,param_idx])

            val = np.percentile(mcmc_data,perc)

        else:
            mcmc_data = f['mcmc_chain'][...]
            val = np.percentile(mcmc_data,perc)
        f.close()

        return val

    def get_data(self,data_dir,sim,spec_or_photo,noise,model,igals,mcmc_postproc):
        if spec_or_photo == 'photo':
            param_list = np.array(['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh', 'beta4_sfh', 'gamma1_zh', 'gamma2_zh', 'tau'])
        else:
            param_list = np.array(['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh', 'beta4_sfh', 'gamma1_zh',
                                    'gamma2_zh', 'tau', 'f_fiber'])
        param_medians = {}
        param_up_sigmas = {}
        param_lo_sigmas = {}

        save_base_dir = os.path.join(os.environ.get('GQPMC'),'doctor','data_list','data_retrieval',f'{model}')
        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir)

        save_dir_median = os.path.join(save_base_dir,f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{mcmc_postproc}.median.npy')
        save_dir_upsig = os.path.join(save_base_dir,f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{mcmc_postproc}.upsig.npy')
        save_dir_losig = os.path.join(save_base_dir,f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{mcmc_postproc}.losig.npy')

        for n, param in enumerate(param_list):
            param_med = []
            param_upsig = []
            param_losig = []

            for igal in igals:
                f_name = os.path.join(data_dir,
                    f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{igal}.{mcmc_postproc}.hdf5')
                if os.path.exists(f_name):
                    med = self._get_data(f_name,n,mcmc_postproc,all_data=True,perc=50)
                    upsig = self._get_data(f_name,n,mcmc_postproc,all_data=True,perc=84.1)
                    losig = self._get_data(f_name,n,mcmc_postproc,all_data=True,perc=15.9)
                else:
                    print(f'igal{igal} not found')
                    med = 'N/A'
                    upsig = 'N/A'
                    losig = 'N/A'

                param_med.append(med)
                param_upsig.append(upsig)
                param_losig.append(losig)
            param_medians[param] = param_med
            param_up_sigmas[param] = param_upsig
            param_lo_sigmas[param] = param_losig

        np.save(save_dir_median,param_medians)
        np.save(save_dir_upsig,param_up_sigmas)
        np.save(save_dir_losig,param_lo_sigmas)

    def check_walkers(self,data_dir,sim,spec_or_photo,noise,model,igals,only_first):
        write_dir = os.path.join(os.environ.get('GQPMC'),'doctor','data_list','walker_log')
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        log_writeout = os.path.join(write_dir,f'{sim}.{spec_or_photo}.noise_{noise}.{model}.walker.log')
        f_log = open(log_writeout,'w')
        for igal in igals:
            f_log.write(f'----- igal {igal} -----\n')
            f_name = os.path.join(data_dir,f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{igal}.mcmc.hdf5')
            if os.path.exists(f_name):
                f = h5py.File(f_name,'r')
                prior_ranges = f['prior_range'][...]
                if only_first:
                    p0 = f['mcmc_chain0'][...][0,:,:]
                    for p_idx, p in enumerate(p0):
                        for prior_idx, prior in enumerate(prior_ranges):
                            up_lim = prior[1]
                            lo_lim = prior[0]
                            if not (p[prior_idx] < up_lim and p[prior_idx] > lo_lim):
                                txt = f'Search Scope: Initial position\nWalker index: {p_idx} Prior index: {prior_idx} Walker position: {p[prior_idx]}\nPrior range: ({lo_lim},{up_lim})\n'
                                f_log.write(txt)
                else:
                    num = -1
                    ndim = len(f['theta_names'][...])
                    for k in list(f.keys()):
                        if 'mcmc_chain' in k:
                            num+=1
                    flat_chain = []
                    for mcmc_num in range(num):
                        flat_chain.append(f[f'mcmc_chain{mcmc_num}'][...])
                    flat_chain = np.array(flat_chain)#
                    nwalker = flat_chain.shape[2]
                    ndim = flat_chain.shape[3]
                    flat_chain = flat_chain.reshape(-1,nwalker,ndim)
                    for prior_idx, prior in enumerate(prior_ranges):
                        up_lim = prior[1]
                        lo_lim = prior[0]
                        flat_param_chain = flat_chain[:,:,prior_idx]
                        if np.any(flat_param_chain < lo_lim) or np.any(flat_param_chain > up_lim):
                            f_log.write(f'Search scope: Entire chain\nBad walker found in the chain. Prior index: {prior_idx}\n')
            else:
                f_log.write('File not found\n')
                print(f'igal{igal} not found')


    def _plot_walkers(self,f,ax,param_idx,latest=True,inc=1):
        keys = list(f.keys())

        num = -1
        for k in keys:
            if 'mcmc_chain' in k:
                num += 1
        if not latest:
            mcmc_data = np.array([])
            for idx in range(num+1):
                mcmc_data = np.append(mcmc_data,f[f'mcmc_chain{idx}'][...][:,:,param_idx])
        else:
            mcmc_data = f[f'mcmc_chain{num}'][...][:,:,param_idx]

        num_walkers = mcmc_data.shape[1]
        length = mcmc_data.shape[0]//inc
        walker_mcmc_med, walker_mcmc_upsig, walker_mcmc_losig = np.zeros((num_walkers,length)), np.zeros((num_walkers,length)), np.zeros((num_walkers,length))

        for i in range(num_walkers):
            walker = mcmc_data[:,i]
            walker = walker[::inc]
            running_median, running_upsig, running_losig = [], [], []
            for ii in range(length):
                running_median.append(np.median(walker[:(ii+1)*inc]))
                running_upsig.append(np.percentile(walker[:(ii+1)*inc],84.1))
                running_losig.append(np.percentile(walker[:(ii+1)*inc],15.9))

            walker_mcmc_med[i] = running_median
            walker_mcmc_upsig[i] = running_upsig
            walker_mcmc_losig[i] = running_losig

        total_med = np.median(walker_mcmc_med, axis=0)
        total_upsig = np.median(walker_mcmc_upsig, axis = 0)
        total_losig = np.median(walker_mcmc_losig, axis = 0)
        ax.plot(np.arange(len(total_med))*inc*num_walkers, total_med, c = 'red', lw = 1.5)
        ax.fill_between(np.arange(len(total_med))*inc*num_walkers,y1 = total_losig,y2 = total_upsig, color = 'red', alpha = 0.4)

        prior = f['prior_range'][...]
        ax.axhline(prior[param_idx][0], c = 'k', lw = 1, ls = '--', alpha = 0.5)
        ax.axhline(prior[param_idx][1], c = 'k', lw = 1, ls = '--', alpha = 0.5)

        label = np.array(f['theta_names'][...]).astype(str)[param_idx]
        ax.set_ylabel(f'Prior Range & {label})')
        ax.set_xlabel('Iteration')

    def plot_walkers(self,data_dir,sim,spec_or_photo,noise,model,igals,latest):
        if spec_or_photo == 'photo':
            param_list = np.array(['logmstar', 'beta1_sfh', 'beta2_sfh', 'beta3_sfh', 'beta4_sfh', 'gamma1_zh', 'gamma2_zh', 'tau'])
        else:
            param_list = np.array(['logmstar' 'beta1_sfh' 'beta2_sfh' 'beta3_sfh' 'beta4_sfh' 'gamma1_zh'
                                    'gamma2_zh' 'tau' 'f_fiber'])
        n_param = len(param_list)

        save_base_dir = os.path.join(os.environ.get('GQPMC'),
            'doctor','data_list','param_plots')
        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir)

        for igal in igals:
            print(f'igal{igal}')
            f_name = os.path.join(data_dir,
                f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{igal}.mcmc.hdf5')
            if os.path.exists(f_name):
                fig, axs = plt.subplots(1,n_param, figsize = (7*n_param,6))
                f = h5py.File(f_name,'r')
                for n, param in enumerate(param_list):
                    self._plot_walkers(f,axs[n],n,latest=latest)
                fig.suptitle(f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{igal} parameter plots')
                fig.savefig(os.path.join(save_base_dir,
                    f'{sim}.{spec_or_photo}.noise_{noise}.{model}.{igal}.param_plots.pdf'), format = 'pdf', bbox_inches = 'tight')
                f.close()
            else:
                print(f'igal{igal} not found')

        pass

if __name__ == '__main__':
    test = sys.argv[1] # 'diagnose', 'get_data', 'check_walkers', 'plot_walkers'
    data_dir = sys.argv[2]
    sim = sys.argv[3]
    spec_or_photo = sys.argv[4]
    noise = sys.argv[5]
    model = sys.argv[6]
    igal = sys.argv[7]

    try:
        extra = sys.argv[8]
        if test == 'check_walkers' or test == 'plot_walkers':
            extra = extra == 'True'

        elif test == 'get_data':
            assert np.isin(extra,['mcmc','postproc'])
    except:
        pass
    doctor = Doctor()
    if test == 'diagnose':
        print('--- Diagnosing Parameter ---')
        work = doctor.diagnose
    elif test == 'get_data':
        print('--- Retrieving Data ---')
        if igal !='all':
            igal = 'all'
            print('Retrieval scope set to \'all\'')
        work = doctor.get_data
    elif test == 'check_walkers':
        print('--- Checking Walkers ---')
        work  = doctor.check_walkers
    elif test == 'plot_walkers':
        print('--- Plotting Walkers ---')
        work = doctor.plot_walkers
    else:
        raise ValueError('Select one of the operation from \'diagnose\', \'get_data\', \'check_walkers\', \'plot_walkers\'')

    if igal == 'all':
        igals = np.arange(97)
    else:
        igals = igal.split(',')

    work(data_dir,sim,spec_or_photo,noise,model,igals,extra)

        
