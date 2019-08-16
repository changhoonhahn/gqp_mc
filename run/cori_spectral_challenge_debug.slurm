#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --constraint=knl
#SBATCH -N 1
#SBATCH -J spec_challenge
#SBATCH -o _spec_challenge.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

module load python
source activate gqp
source ~/.bashrc.ext 

export HDF5_USE_FILE_LOCKING=FALSE

i0=11
i1=13
n_cpu=3

echo 'debugging: fitting spectra of galaxies # '$i0' to '$i1
srun -n 1 -c $n_cpu python /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py spec $i0 $i1 none False $n_cpu 100 100 1000 

#echo 'debugging: fitting photometry of galaxies # '$i0' to '$i1
#srun -n 1 -c $n_cpu python -W ignore /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py photo $i0 $i1 none False $n_cpu 100 100 1000

now=$(date +"%T")
echo "end time ... $now"
