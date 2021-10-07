#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J s2_provabgs
#SBATCH -o ofiles/mocha_s2_1.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

conda activate gqp 
module unload PrgEnv-intel
module load PrgEnv-gnu

# [x] 0 - 99 
# [ ] 100 199 
# [ ] 200 299 
# [ ] 300 399 
# [ ] 400 499 
# [ ] 500 599 
# [ ] 600 699 
# [ ] 700 799 
# [ ] 800 899 
# [ ] 900 999 
# [ ] 1000 1099 
i0=0
i1=99
niter=2500
n_cpu=32

export OMP_NUM_THREADS=1

python -W ignore /global/homes/c/chahah/projects/gqp_mc/run/mocha_S2_provabgs.py \
    $i0 $i1 $niter $n_cpu 

now=$(date +"%T")
echo "end time ... $now"
