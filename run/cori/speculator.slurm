#!/bin/bash 
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J speculator 
#SBATCH -o ofiles/speculator.o
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

conda activate gqp

# generate training set for simpledust 
python /global/homes/c/chahah/projects/gqp_mc/run/speculator_training.py train simpledust 100 
python /global/homes/c/chahah/projects/gqp_mc/run/speculator_training.py train complexdust 100 

# train pca 
python /global/homes/c/chahah/projects/gqp_mc/run/speculator_pca.py simpledust 0 100 20
python /global/homes/c/chahah/projects/gqp_mc/run/speculator_pca.py complexdust 0 100 30

now=$(date +"%T") 
echo "end time ... $now"
