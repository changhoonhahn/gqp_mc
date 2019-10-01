#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J minimocha_spec_none
#SBATCH -o _minimocha_spec_none.o
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

echo 'fitting spectra of galaxies # '$i0' to '$i1
######################################################
# no noise
######################################################
srun -n 1 -c $n_cpu python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 none $n_cpu 100 100 1000 False
######################################################
# yes noise
######################################################
#srun -n 1 -c $n_cpu python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 bgs1 $n_cpu 100 100 1000 False

now=$(date +"%T")
echo "end time ... $now"
