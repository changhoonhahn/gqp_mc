#GQP Mock Challenge 

Python package for the GQP mock challenge. Currently includes functions
that make it easy to read in forward modeled Lgal spectra and photometry. 


## Table of Conents 
[Installiation](#installation)<br>
[To do](#to-do)<br>

## Installation

### on NERSC

Below is one way to setup the package on nersc. 

First, define `$GQPMC_DIR` environment in your `~/.bashrc.ext`---
i.e. add 
```
export GQPMC_DIR="\SOME_LOCAL_DIRECTORY\" 
```
to your `~/.bashrc.ext` file. 

Then run 
```
source ~/.bashrc.ext
```
on the command line 

Now we're going to symlink to the spectral_challenge directory 
in the desi project directory so that we have access to the data
```
# go to $GQPMC_DIR
cd $GQPMC_DIR
ln -s /global/projecta/projectdirs/desi/mocks/LGal_spectra/ Lgal
```

With the data all set up, we can now install the package: 
```bash 
# load python 
module load python 

# create conda environment 
conda create -n gqp python=3.7

# install dependencies
pip install h5py 
pip install pytest 
pip install astropy 

# clone the repo 
git clone https://github.com/changhoonhahn/gqp_mc.git 

# go to project directory
cd gqp_mc 

# install package
python setup.py install --user 

# test the package
pytest 
```

## To do 

* `speclite` package is currently used for photometry, but this should be phased out for the filters in fsps.
