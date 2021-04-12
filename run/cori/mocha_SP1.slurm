#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J mocha_sp1
#SBATCH -o ofiles/mocha_sp1.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp 

i0=0
i1=32
niter=2500
n_cpu=32

export OMP_NUM_THREADS=1

python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mocha_SP1_provabgs.py \
    $i0 $i1 $niter $n_cpu 

now=$(date +"%T")
echo "end time ... $now"
