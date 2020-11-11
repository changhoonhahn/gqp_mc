'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 


def deploy_trainingset_job(ibatch, model='simpledust', ncpu=1): 
    ''' create slurm script and then submit 
    '''
    cntnt = '\n'.join(["#!/bin/bash", 
        "#SBATCH -J train%i" % ibatch,
        "#SBATCH --exclusive",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=40",
        "#SBATCH --partition=general",
        "#SBATCH --time=00:14:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=ofiles/train_%s%i.o" % (model[0], ibatch), 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=changhoon.hahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load anaconda3", 
        "conda activate gqp", 
        "",
        "python /home/chhahn/projects/gqp_mc/run/speculator_training.py train %s %i %i" % (model, ibatch, ncpu), 
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_train.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _train.slurm')
    os.system('rm _train.slurm')
    return None 


def deploy_trainpca_job(ibatch0, ibatch1, n_pcas, model='simpledust'):
    ''' create slurm script and then submit 
    '''
    n_pca0, n_pca1, n_pca2 = n_pcas
    cntnt = '\n'.join(["#!/bin/bash", 
        "#SBATCH -J pca%i%i%i_%i_%i" % (n_pca0, n_pca1, n_pca2, ibatch0, ibatch1),  
        "#SBATCH --exclusive",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=40",
        "#SBATCH --partition=general",
        "#SBATCH --time=01:59:59", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=ofiles/train_pca%i%i%i_%s_%i_%i.o" % (n_pca0, n_pca1, n_pca2, model[0], ibatch0, ibatch1), 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=changhoon.hahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load anaconda3", 
        "conda activate gqp", 
        "",
        "python /home/chhahn/projects/gqp_mc/run/speculator_pca.py train %s %i %i %i %i %i" % (model, ibatch0, ibatch1, n_pca0, n_pca1, n_pca2),
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_pca.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _pca.slurm')
    os.system('rm _pca.slurm')
    return None 


def deploy_trainspeculator_job(model, i_wave, n_pcas, Ntrain, Nlayer, Nunits):
    ''' create slurm script for training speculator and then submit 
    '''
    cntnt = '\n'.join(["#!/bin/bash", 
        "#SBATCH -J train_spec_%s_%i_%i_%i" % (model, i_wave, n_pcas, Ntrain),  
        "#SBATCH --exclusive",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        '#SBATCH --gres=gpu:1', 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=ofiles/train_spec_%s_%i_%i_%i.o" % (model, i_wave, n_pcas, Ntrain), 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load anaconda3", 
        "conda activate tf2-gpu", 
        "",
        "python /home/chhahn/projects/gqp_mc/run/train_speculator.py %s %i %i %i %i %i" % (model, i_wave, n_pcas, Ntrain, Nlayer, Nunits),
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_train.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _train.slurm')
    os.system('rm _train.slurm')
    return None 

job_type = sys.argv[1]

if job_type == 'trainingset': 
    ibatch0 = int(sys.argv[2])
    ibatch1 = int(sys.argv[3])
    model = sys.argv[4]
    ncpu = int(sys.argv[5]) 
    for ibatch in range(ibatch0, ibatch1+1): 
        print('submitting %s batch %i' % (model, ibatch))
        deploy_trainingset_job(ibatch, model=model, ncpu=ncpu)
elif job_type == 'trainpca': 
    ibatch0 = int(sys.argv[2])
    ibatch1 = int(sys.argv[3])
    model = sys.argv[4]
    n_pca0 = int(sys.argv[5]) 
    n_pca1 = int(sys.argv[6]) 
    n_pca2 = int(sys.argv[7]) 
    n_pcas = [n_pca0, n_pca1, n_pca2]
    print('submitting [%i, %i, %i] component pca %s training' % (n_pca0, n_pca1, n_pca2, model))
    deploy_trainpca_job(ibatch0, ibatch1, n_pcas, model=model)
elif job_type == 'trainspeculator': 
    model = sys.argv[2]
    i_wave = int(sys.argv[3]) 
    n_pcas = int(sys.argv[4]) 
    Ntrain = int(sys.argv[5]) 
    Nlayer = int(sys.argv[6]) 
    Nunits = int(sys.argv[7]) 
    deploy_trainspeculator_job(model, i_wave, n_pcas, Ntrain, Nlayer, Nunits)
else: 
    raise ValueError
