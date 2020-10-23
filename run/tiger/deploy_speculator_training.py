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


def deploy_trainpca_job(ibatch0, ibatch1, n_pca, model='simpledust', train_or_transform='transform'): 
    ''' create slurm script and then submit 
    '''
    cntnt = '\n'.join(["#!/bin/bash", 
        "#SBATCH -J pca%i_%i_%i" % (n_pca, ibatch0, ibatch1),  
        "#SBATCH --exclusive",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=40",
        "#SBATCH --partition=general",
        "#SBATCH --time=%s" % {"train": "01:59:59", "transform": "01:00:01"}[train_or_transform],
        "#SBATCH --export=ALL",
        "#SBATCH --output=ofiles/%s_pca%i_%s_%i_%i.o" % (train_or_transform, n_pca, model[0], ibatch0, ibatch1), 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=changhoon.hahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load anaconda3", 
        "conda activate gqp", 
        "",
        "python /home/chhahn/projects/gqp_mc/run/speculator_pca.py %s %i %i %i %s" % (model, ibatch0, ibatch1, n_pca, train_or_transform),
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

job_type = sys.argv[1]

ibatch0 = int(sys.argv[2])
ibatch1 = int(sys.argv[3])

if job_type == 'trainingset': 
    model = sys.argv[4]
    ncpu = int(sys.argv[5]) 
    for ibatch in range(ibatch0, ibatch1+1): 
        print('submitting %s batch %i' % (model, ibatch))
        deploy_trainingset_job(ibatch, model=model, ncpu=ncpu)
elif job_type == 'trainpca': 
    model = sys.argv[4]
    n_pca = int(sys.argv[5]) 
    train_or_transform = sys.argv[6]
    print('submitting %i component pca %s for %sing' % (n_pca, model, train_or_transform))
    deploy_trainpca_job(ibatch0, ibatch1, n_pca, model=model, train_or_transform=train_or_transform)
else: 
    raise ValueError
