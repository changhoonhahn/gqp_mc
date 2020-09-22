'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 


def deploy_trainingset_job(ibatch, model='simpledust'): 
    ''' create slurm script and then submit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH --qos=regular", 
        "#SBATCH --time=01:00:00", 
        "#SBATCH --constraint=haswell", 
        "#SBATCH -N 1", 
        "#SBATCH -J train%i" % ibatch,  
        "#SBATCH -o ofiles/train_%s%i.o" % (model[0], ibatch), 
        "#SBATCH -L SCRATCH,project", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "conda activate gqp", 
        "",
        "python /global/homes/c/chahah/projects/gqp_mc/run/speculator_training.py %s %i" % (model, ibatch), 
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


def deploy_trainpca_job(ibatch0, ibatch1, model='simpledust'): 
    ''' create slurm script and then submit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH --qos=regular", 
        "#SBATCH --time=01:00:00", 
        "#SBATCH --constraint=haswell", 
        "#SBATCH -N 1", 
        "#SBATCH -J pca%i_%i" % (ibatch0, ibatch1),  
        "#SBATCH -o ofiles/pca_%s_%i_%i.o" % (model[0], ibatch0, ibatch1), 
        "#SBATCH -L SCRATCH,project", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "conda activate gqp", 
        "",
        "python /global/homes/c/chahah/projects/gqp_mc/run/speculator_pca.py %s %i %i" % (model, ibatch0, ibatch1),
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
    for ibatch in range(ibatch0, ibatch1+1): 
        print('submitting %s batch %i' % (model, ibatch))
        deploy_trainingset_job(ibatch, model=model)
elif job_type == 'trainpca': 
    model = sys.argv[4]
    print('submitting pca training for %s' % model) 
    deploy_trainpca_job(ibatch0, ibatch1, model=model)
else: 
    raise ValueError
