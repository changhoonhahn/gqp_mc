#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J lgal_ispec_specphoto_post
#SBATCH -o ofiles/lgal_ispec_specphoto_post.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp

sim=lgal
i0=0
i1=96
n_cpu=32

export OMP_NUM_THREADS=1

# run MCMC with adaptive convergence and maxiter=50000 
python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
    specphoto $sim $i0 $i1 bgs0_legacy ispeculator emulator $n_cpu 40 200 \
    adaptive 50000 True False 

# postprocess the MC chains 
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ispeculator emulator $n_cpu 40 200 \
#    adaptive 100000 False True 


now=$(date +"%T")
echo "end time ... $now"
