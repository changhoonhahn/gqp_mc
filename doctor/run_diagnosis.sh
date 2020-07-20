#!bin/bash
#conda activate gqp

data_dir=/global/cscratch1/sd/kgb0255/gqp_mc/mini_mocha/ispeculator/james/fsps/
sim=lgal
spec_or_photo=specphoto
noise=bgs0_legacy
model=fsps
igal=all

#Retreive medians, 84.1 percentile, 15.9 percentile for mcmc files
python doctor.py get_data $data_dir $sim $spec_or_photo $noise $model $igal mcmc

#Retreive medians, 84.1 percentile, 15.9 percentile for postproc files
python doctor.py get_data $data_dir $sim $spec_or_photo $noise $model $igal postproc

#Get fractional error plots for 3 (photo) or 4 (specphoto) parameters - logM_total log SFR 100Myr logSFR 1Gyr (f_fiber)
python doctor.py diagnose $data_dir $sim $spec_or_photo $noise $model $igal None 

#Inspect the initial distribution of walkers and generate a log file
python doctor.py check_walkers $data_dir $sim $spec_or_photo $noise $model $igal True

#Inspect the distribution of walkers throughout the chains and generate a log file
#python doctor.py check_walkers $data_dir $sim $spec_or_photo $noise $model $igal False

#Plot the median vs iteration of walkers in the last chain (might take substantially long - able to adjust thinning in the doctor.py)
#python doctor.py plot_walkers $data_dir $sim $spec_or_photo $noise $model $igal True

#Plot the median vs iteration of walkers in the entire chain (might take substantially long - able to adjust thinning in the doctor.py))
#python doctor.py plot_walkers $data_dir $sim $spec_or_photo $noise $model $igal True
