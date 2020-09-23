# running `mini_mocha.py`

```bash
python mini_mocha.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${4} ${4} ${4}
```
1. data type: 'photo', 'spec', or 'specphoto'
2. simulation: 'lgal' or 'tng'
3. first galaxy index 
4. last galaxy index
5. noise model for the data
    - use 'legacy' for photo 
    - 'bgs0', 'bgs1', ... for spec
    - 'bgs0_legacy' for spec photo
6. fitting method: 'ispeculator'
7. model name: 'emulator' 
8. number of threads 
9. number of walkers
10. number of iterations for burn in
11. number of iterations: either specify a number of put `adaptive`
12. overwrite? If `True` existing file will be overwritten
13. postprocess SFR calculation from the MC chains


**examples**:
```bash
python -W ignore mini_mocha.py photo lgal 0 10 legacy ispeculator emulator 1 20 20 40 True True 
python -W ignore mini_mocha.py specphoto lgal 0 10 bgs0_legacy ispeculator emulator 1 20 20 40 True True 
```
---
# training `Speculator`
## generating training SED data using FSPS 
`Speculator` is trained on SEDs generated using FSPS. The script
`run/speculator_training.py` can be used to generate extra batches of 
training SED data. The script samples 10000 parameters from the prior and 
runs them through FSPS to generate SEDs. The script is meant to be run on 
NERSC cori. 

```bash
python speculator_training.py train {1} {2}
```
1. model name: 'simpledust' or 'complexdust' 
2. batch number (batches 0 to 99 have already been generated) 

This will generate two `.npy` files. One that contains the parameter values
sampled from the prior. 

```python
'DESI_%s.theta_train.%i.seed%i.npy' % (model, ibatch, seed))
```

Another that contains the corresponding SED values. 

```python
'DESI_%s.logspectrum_fsps_train.%i.seed%i.npy' % (model ibatch, seed))
```

## training the PCA
Before you can use the training SED data to train `Speculator`, you first have
to train a PCA and decompose the training SED data into PCA componenets. PCA
components are what `Speculator` ultmiately predict. 

The script `run/speculator_pca.py`can be used to train the PCA over multiple
batches of SED training data created from the section above. The script is 
meant to be run on NERSC cori. 

```bash
python speculator_pca.py {1} {2} {3} {4} 
```
1. model name: 'simpledust' or 'complexdust'
2. batch0: first batch number 
3. batch1: last batch number
4. n_pcas: number of pca components 

This will train a PCA with n_pcas components for specified model name over 
batches batch0 to batch1. 

## running on cori
To run either of the above scripts on cori, you can modify and use the slurm 
script `run/cori/speculator.slurm`. 

