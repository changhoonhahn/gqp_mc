#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J p1_tau
#SBATCH -o ofiles/mocha_p1_tau.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp 
module unload PrgEnv-intel
module load PrgEnv-gnu

export OMP_NUM_THREADS=1

i0=0
i1=99
niter=2500
n_cpu=32

python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mocha_P1_tau.py \
    $i0 $i1 $niter $n_cpu 

now=$(date +"%T")
echo "end time ... $now"
