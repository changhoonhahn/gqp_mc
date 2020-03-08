#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J MM_iSpec 
#SBATCH -o _MM_iSpec.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

module load python
source activate gqp
source ~/.bashrc.ext 

sim=tng
i0=0
i1=97
n_cpu=32

export OMP_NUM_THREADS=1

######################################################
# ifsps  
######################################################
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    spec $sim $i0 $i1 bgs0 ifsps vanilla $n_cpu 10 100 1000 True 
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    photo $sim $i0 $i1 legacy ifsps vanilla $n_cpu 10 100 1000 True 
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ifsps vanilla $n_cpu 20 100 1000 True

######################################################
# speculator  
######################################################
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    spec $sim $i0 $i1 bgs0 ispeculator NA $n_cpu 20 100 1000 True 
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    photo $sim $i0 $i1 legacy ispeculator NA $n_cpu 20 100 1000 True
#srun -n 1 -c $n_cpu\
python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
    specphoto $sim $i0 $i1 bgs0_legacy ispeculator emulator $n_cpu 40 200 4000 True

now=$(date +"%T")
echo "end time ... $now"
