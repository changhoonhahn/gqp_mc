#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J provabgs_mocks_noisy
#SBATCH -o ofiles/provabgs_mocks_noisy_cp.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp 

i0=0
i1=99
niter=2500
n_cpu=32

export OMP_NUM_THREADS=1

python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mocha_provabgs_noisy.py \
    $i0 $i1 $niter $n_cpu False
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mocha_provabgs_noisy.py \
#    $i0 $i1 $niter $n_cpu True

now=$(date +"%T")
echo "end time ... $now"
