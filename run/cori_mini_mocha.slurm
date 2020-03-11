#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J MM_lgal_iSpec_photo
#SBATCH -o _MM_lgal_iSpec_photo.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

module load python
source activate gqp
source ~/.bashrc.ext 

sim=lgal
i0=0
i1=96
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
#    specphoto $sim $i0 $i1 bgs0_legacy ifsps vanilla $n_cpu 20 100 1000 False

######################################################
# speculator  
######################################################
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    spec $sim $i0 $i1 bgs0 ispeculator NA $n_cpu 20 100 1000 True 
python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
    photo $sim $i0 $i1 legacy ispeculator emulator $n_cpu 20 200 4000 False
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ispeculator emulator $n_cpu 40 200 4000 False
######################################################
# speculator w/ fsps  
######################################################
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ispeculator fsps $n_cpu 40 200 4000 True
#python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ispeculator fsps_complexdust $n_cpu 40 200 4000 True

now=$(date +"%T")
echo "end time ... $now"
