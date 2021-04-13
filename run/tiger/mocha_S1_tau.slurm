#!/bin/bash
#SBATCH -J s1_tau
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --output=o/s1_tau.o
#SBATCH --mail-type=all
#SBATCH --mail-user=chhahn@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module load anaconda3
conda activate gqp 

python /home/chhahn/projects/gqp_mc/run/mocha_S1_tau.py 0 4 10 5

now=$(date +"%T")
echo "end time ... $now"
