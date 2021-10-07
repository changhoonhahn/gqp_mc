#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J p2_provabgs
#SBATCH -o ofiles/mocha_p2_provabgs.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp 
module unload PrgEnv-intel
module load PrgEnv-gnu

i0=0
i1=2124
niter=2500
n_cpu=32

export OMP_NUM_THREADS=1

python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mocha_P2_provabgs.py \
    $i0 $i1 $niter $n_cpu 

now=$(date +"%T")
echo "end time ... $now"
