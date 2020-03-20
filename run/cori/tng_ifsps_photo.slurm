#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --constraint=knl
#SBATCH -N 1
#SBATCH -J tng_ifsps_photo
#SBATCH -o ofiles/tng_ifsps_photo.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp

sim=tng
i0=0
i1=96
n_cpu=64

export OMP_NUM_THREADS=1

######################################################
# ifsps  
######################################################
python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
    photo $sim $i0 $i1 legacy ifsps vanilla $n_cpu 10 100 1000 True 

now=$(date +"%T")
echo "end time ... $now"
