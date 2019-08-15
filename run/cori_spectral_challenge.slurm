#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=03:00:00
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

for i in {0..4}; do
    echo 'spectral fitting '$i 
    srun -n 1 -c 1 python /global/homes/c/chahah/projects/gqp_mc/run/spectral_challenge.py spec $i none False
done
for i in {0..4}; do
    echo 'photometric fitting '$i 
    srun -n 1 -c 1 python /global/homes/c/chahah/projects/gqp_mc/run/spectral_challenge.py photo $i none False
done 

now=$(date +"%T")
echo "end time ... $now"
