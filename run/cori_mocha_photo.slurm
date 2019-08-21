#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J photofit_leg_dust
#SBATCH -o _photofit_leg_dust.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

module load python
source activate gqp
source ~/.bashrc.ext 

export HDF5_USE_FILE_LOCKING=FALSE

i0=0
i1=96
n_cpu=32

echo 'fitting photometry of galaxies # '$i0' to '$i1
######################################################
# no noise no dust 
######################################################
#srun -n 1 -c $n_cpu python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/spectral_challenge.py photo $i0 $i1 none False $n_cpu 100 100 1000 False 
######################################################
# no noise yes dust 
######################################################
#srun -n 1 -c $n_cpu python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/spectral_challenge.py photo $i0 $i1 none True $n_cpu 100 100 1000 False 

######################################################
# yes noise no dust 
######################################################
#srun -n 1 -c $n_cpu python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/spectral_challenge.py photo $i0 $i1 legacy False $n_cpu 100 100 1000 False 
######################################################
# yes noise yes dust 
######################################################
srun -n 1 -c $n_cpu python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/spectral_challenge.py photo $i0 $i1 legacy True $n_cpu 100 100 1000 False 

now=$(date +"%T")
echo "end time ... $now"
