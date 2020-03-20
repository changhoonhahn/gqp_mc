#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --constraint=knl
#SBATCH -N 1
#SBATCH -J lgal_ifsps_specphoto
#SBATCH -o ofiles/lgal_ifsps_specphoto.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp

sim=lgal
i0=0
i1=96
n_cpu=64

export OMP_NUM_THREADS=1

######################################################
# ifsps  
######################################################
python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
    specphoto $sim $i0 $i1 bgs0_legacy ifsps vanilla $n_cpu 20 100 1000 True

now=$(date +"%T")
echo "end time ... $now"
