#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --constraint=haswell
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

srun -n 1 -c 1 python /global/homes/c/chahah/projects/gqp_mc/run/spectral_challenge.py 'photo' 0

now=$(date +"%T")
echo "end time ... $now"
