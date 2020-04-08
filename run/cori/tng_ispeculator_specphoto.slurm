#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=knl
#SBATCH -N 1
#SBATCH -J tng_ispec_specphoto
#SBATCH -o ofiles/tng_ispec_specphoto.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp

sim=tng
i0=0
i1=96
n_cpu=64

export OMP_NUM_THREADS=1

python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
    specphoto $sim $i0 $i1 bgs0_legacy ispeculator emulator $n_cpu 40 200 4000 True

now=$(date +"%T")
echo "end time ... $now"
