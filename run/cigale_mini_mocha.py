''' 

script to run photometry fit with CIGALE (mini mocha)

'''

import sys 
import os 
import h5py 
import numpy as np 
import corner as DFM 
from functools import partial
from multiprocessing.pool import Pool 
from pcigale.analysis_modules import get_module
import shutil

# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
from gqp_mc import fitters as Fitters
# --- cigale ---
from pcigale.session.configuration import Configuration
from pcigale.managers.parameters import ParametersManager

def cigale_dir(sample='mini_mocha'): 
    return os.path.join(UT.dat_dir(),sample,'cigale')
    
def cigaledata(sim='lgal',noise='none',lib='bc03', sample='mini_mocha', f_out='lgal.mini_mocha.noise.none.dat'):
    ''' generate data file for CIGALE (mini mocha)'''
    
    #reading data
    photo,meta=Data.Photometry(sim='lgal',noise=noise,lib='bc03', sample='mini_mocha')
    #dir_sample = os.path.join(UT.dat_dir(), sample) 
    #reading ids and redshift
    redshift=np.array(meta['redshift'][:])
    galid=np.array(meta['galid'][:])

    bands = ['g', 'r', 'z', 'w1', 'w2']#, 'w3', 'w4']

    for icol, band in enumerate(bands): 
        photo['flux'][:,icol] = photo['flux'][:,icol] * 3631 *  1e-6	# convert to mJansky [mJy]
        if noise == 'none': # assume 0.1% flux error
            sample=np.concatenate((galid.reshape(-1,1),redshift.reshape(-1,1) #id, redshift
                ,photo['flux'][:,0].reshape(-1,1),0.001*photo['flux'][:,0].reshape(-1,1) #flux_g,err_flux_g
                ,photo['flux'][:,1].reshape(-1,1),0.001*photo['flux'][:,1].reshape(-1,1) #flux_r,err_flux_r
                ,photo['flux'][:,2].reshape(-1,1),0.001*photo['flux'][:,2].reshape(-1,1) #flux_z,err_flux_z
                ,photo['flux'][:,3].reshape(-1,1),0.001*photo['flux'][:,3].reshape(-1,1) #flux_w1,err_flux_w1
                ,photo['flux'][:,4].reshape(-1,1),0.001*photo['flux'][:,4].reshape(-1,1) #flux_w2,err_flux_w2
                #,photo['flux'][:,5].reshape(-1,1),0.01*photo['flux'][:,5].reshape(-1,1) #flux_w3,err_flux_w3
                #,photo['flux'][:,6].reshape(-1,1),0.01*photo['flux'][:,6].reshape(-1,1) #flux_w4,err_flux_w4
                ),axis = 1)
        if noise != 'none': 
            photo['ivar'][:,icol] = photo['ivar'][:,icol]**-0.5 * 3631  *  1e-6 # inverse variance to error [mJy]
            sample=np.concatenate((galid.reshape(-1,1),redshift.reshape(-1,1) #id, redshift
                ,photo['flux'][:,0].reshape(-1,1),photo['ivar'][:,0].reshape(-1,1) #flux_g,err_flux_g
                ,photo['flux'][:,1].reshape(-1,1),photo['ivar'][:,1].reshape(-1,1) #flux_r,err_flux_r
                ,photo['flux'][:,2].reshape(-1,1),photo['ivar'][:,2].reshape(-1,1) #flux_z,err_flux_z
                ,photo['flux'][:,3].reshape(-1,1),photo['ivar'][:,3].reshape(-1,1) #flux_w1,err_flux_w1
                ,photo['flux'][:,4].reshape(-1,1),photo['ivar'][:,4].reshape(-1,1) #flux_w2,err_flux_w2
                #,photo['flux'][:,5].reshape(-1,1),photo['ivar'][:,5].reshape(-1,1) #flux_w3,err_flux_w3
                #,photo['flux'][:,6].reshape(-1,1),photo['ivar'][:,6].reshape(-1,1) #flux_w4,err_flux_w4
                ),axis = 1)
        np.savetxt(os.path.join(cigale_dir(),f_out), sample, delimiter = ' ', header='id redshift DECam_g DECam_g_err DECam_r DECam_r_err DECam_z DECam_z_err WISE1 WISE1_err WISE2 WISE2_err')# WISE3 WISE3_err WISE4 WISE4_err')

def cigaleini(config, f_out='lgal.mini_mocha.noise.none.dat',Ncores=13):
    ''' generate ini file for CIGALE'''

    if os.path.isfile(os.path.join(cigale_dir(),'pcigale.ini')):
        print('*** CAUTION: overwriting pcigale.ini ***')
    os.chdir(cigale_dir())
    config.create_blank_conf()
    method='pdf_analysis' #Available methods: pdf_analysis, savefluxes
    sed_modules='sfhdelayed, bc03, dustatt_modified_CF00, dl2014, redshifting'
    '''Available modules:
    	SFH: sfh2exp, sfhdelayed, sfhfromfile, sfhperiodic
    	SSP: bc03, m2005
   	Nebular emission: nebular
   	Dust attenuation: dustatt_calzleit, dustatt_powerlaw, dustatt_2powerlaws
    	Dust emission: casey2012, dale2014, dl2007, dl2014
    	AGN: dale2014, fritz2006
    	Radio: radio
    	Redshift: redshifting (mandatory!)
    '''

    inifile = open("pcigale.ini", "r")
    inifile = ''.join([i for i in inifile]) \
	.replace('data_file =', 'data_file = %s' %f_out) \
    	.replace('analysis_method =','analysis_method = %s' %method) \
	.replace('cores =','cores = %s' %Ncores) \
	.replace('sed_modules = ,','sed_modules = %s' %sed_modules)
    readyfile = open("pcigale.ini","w")
    readyfile.writelines(inifile)
    readyfile.close()


def cigalegenconf(config, simple='False'):
    ''' generate config file for CIGALE'''
    
    os.chdir(cigale_dir())
    if os.path.isfile(os.path.join(cigale_dir(),'pcigale.ini')):
        config.generate_conf()
        #sfhdelayed  
        tau_main='500, 2000, 4500' #Myr
        age_main='1000, 2000, 4500, 6000, 8000, 10000, 11500, 13000' #Myr
        tau_burst='10000' #Myr
        age_burst='70' #Myr
        f_burst='0, 0.001, 0.01, 0.03, 0.1' 
        #bc03  
        imf='0' #0 (Salpeter) or 1 (Chabrier)
        metallicity='0.004, 0.008, 0.02, 0.05' #0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05
        #dustatt_modified_CF00  
        Av_ISM='0, 0.1, 0.3, 0.8, 1.2, 1.7, 2.3, 2.8, 3.3, 3.8' #V-band attenuation in the ISM
        mu='0.3, 0.5, 0.8' #Av_ISM / (Av_BC+Av_ISM)
        slope_ISM='-0.7' #Power law slope of the attenuation in the ISM
        slope_BC='-0.7' #Power law slope of the attenuation in the birth cloud
        #dl2014  
        qpah='0.47, 1.12, 2.5, 3.9' #Mass fraction of PAH:0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 5.26, 5.95, 6.63, 7.32
        umin='5, 10, 25' #Minimum radiation field: 0.100, 0.120, 0.150,
        # 0.170, 0.200, 0.250, 0.300, 0.350, 0.400, 0.500, 0.600, 0.700, 0.800,
        # 1.000, 1.200, 1.500, 1.700, 2.000, 2.500, 3.000, 3.500, 4.000, 5.000,
        # 6.000, 7.000, 8.000, 10.00, 12.00, 15.00, 17.00, 20.00, 25.00, 30.00,
        # 35.00, 40.00, 50.00.
        alpha='2.0' #Powerlaw slope dU/dM propto U^alpha: 1.0, 1.1,
        # 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
        # 2.6, 2.7, 2.8, 2.9, 3.0.
        gamma='0.02' #Fraction illuminated from Umin to Umax: range(0,1).
        #analysis_params  
        variables='stellar.m_star, stellar.metallicity, sfh.sfr10Myrs, sfh.tau_main, sfh.age_main, sfh.age_burst, sfh.f_burst, dust.luminosity, dust.mass, dust.qpah, dust.umin, attenuation.V_B90, attenuation.FUV, attenuation.Av_BC, attenuation.mu,attenuation.slope_BC, attenuation.slope_ISM' 
        save_best_sed='True'
        save_chi2='True'
        mock_flag='True'
        if simple !='True':
            inifile = open("pcigale.ini", "r")
            inifile = ''.join([i for i in inifile]) \
	        .replace('tau_main = 2000.0', 'tau_main = %s' %tau_main) \
	        .replace('age_main = 5000','age_main = %s' %age_main) \
	        .replace('tau_burst = 50.0','tau_burst = %s' %tau_burst) \
	        .replace('age_burst = 20','age_burst = %s' %age_burst) \
	        .replace('f_burst = 0.0','f_burst = %s' %f_burst) \
	        .replace('imf = 0','imf = %s' %imf) \
	        .replace('metallicity = 0.02','metallicity = %s' %metallicity) \
	        .replace('Av_ISM = 1.0','Av_ISM = %s' %Av_ISM) \
	        .replace('mu = 0.44','mu = %s' %mu) \
	        .replace('slope_ISM = -0.7','slope_ISM = %s' %slope_ISM) \
	        .replace('slope_BC = -1.3','slope_BC = %s' %slope_BC) \
	        .replace('qpah = 2.5','qpah = %s' %qpah) \
	        .replace('umin = 1.0','umin = %s' %umin) \
	        .replace('alpha = 2.0','alpha = %s' %alpha) \
	        .replace('gamma = 0.1','gamma = %s' %gamma) \
	        .replace('variables = sfh.sfr, sfh.sfr10Myrs, sfh.sfr100Myrs','variables = %s' %variables) \
	        .replace('save_best_sed = False','save_best_sed = %s' %save_best_sed) \
	        .replace('save_chi2 = False','save_chi2 = %s' %save_chi2) \
	        .replace('mock_flag = False','mock_flag = %s' %mock_flag)
            readyfile = open("pcigale.ini","w")
            readyfile.writelines(inifile)
            readyfile.close()
    else:
            print('*** You need to generate CIGALE ini file! ***')

def cigalecheck(config):
    ''' check config file for CIGALE'''

    os.chdir(cigale_dir())
    if os.path.isfile(os.path.join(cigale_dir(),'pcigale.ini')):
        configuration = config.configuration
        if configuration:
            print("With this configuration cigale will compute {} "
                  "models.".format(ParametersManager(configuration).size))
    else:
        print('*** You need to generate CIGALE ini file! ***')

         
def cigalerun(config, noise='none',overwrite='True'):
    ''' run CIGALE'''

    os.chdir(cigale_dir())
    configuration = config.configuration
    if os.path.isfile(os.path.join(cigale_dir(),'pcigale.ini')):
        if configuration:
            analysis_module = get_module(configuration['analysis_method'])
            analysis_module.process(configuration)
            if os.path.exists('noise_%s' %noise):
                if overwrite !='True':
                    print('*** CAUTION: Result folder already exists ***')
                    shutil.rmtree('out')
                else: 
                    shutil.rmtree('noise_%s' %noise)
                    os.rename('out','noise_%s' %noise)
            else:
                os.rename('out','noise_%s' %noise)
    else:
        print('*** You need to generate CIGALE ini file! ***')	

if __name__=="__main__": 

    if len(sys.argv) < 11:
        print('')
        print('*** Usage: sample noise cores generateData generateIniFile generateConfFile generateConfFileTest checkModels run overwrite ***') 
        print('')
        print(' sample:           mini_mocha')
        print(' noise:            none or legacy')
        print(' genereateData:    create input file in mJansky')
        print(' genereateIniFile: generate CIGALE ini file with sed modules')
        print(' genreateConfFile: update CIGALE ini file with parameters ranges')
        print('                   if Test update with basic parameter ranges (to make dummy run)')
        print(' checkModels:      estimate number of models to generate')
        print(' run:              run CIGALE')
        print(' overwrite:        overwrite CIGALE resulting folder')
         
    else:
        sample = sys.argv[1]
        noise = sys.argv[2]
        cores = sys.argv[3]
        data = sys.argv[4] # generate input data
        init = sys.argv[5] # generate init file
        genconf = sys.argv[6] # generate conf file
        simple_config = sys.argv[7] # generate simple config file to test if everything is ok, only if genconf is True
        check = sys.argv[8] # check number of models
        run = sys.argv[9]
        overwrite = sys.argv[10]

    
        fileName='lgal.%s.noise.%s.dat' % (sample,noise)
        #if str_overwrite == 'True': overwrite=True
        #elif str_overwrite == 'False': overwrite=False

        if data == 'True':
    	    print('Generating CIGALE input file for %s sample with noise: %s' %(sample,noise))
    	    cigaledata(noise=noise,sample=sample,f_out=fileName)

        if init == 'True':
            print("Generating CIGALE init file")
            config=Configuration()
            cigaleini(config,f_out=fileName,Ncores=cores)

        if genconf == 'True':
    	    print("Generating CIGALE config file")
    	    config=Configuration()
    	    cigalegenconf(config,simple=simple_config)	  

        if check == 'True':
    	    config=Configuration()
    	    cigalecheck(config)  

        if run == 'True':
    	    config=Configuration()
    	    cigalerun(config,noise=noise,overwrite=overwrite)  

