# GQP Mock Challenge 

Python package for the GQP mock challenge. Currently includes functions
that make it easy to read in forward modeled Lgal spectra and photometry. 


## Table of Conents 
[Installiation](#installation)<br>
[To do](#to-do)<br>
[Installing FSPS](#installing-fsps)<br> 

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

You need to install FSPS if you want to use the iFSPS fitter ([https://github.com/cconroy20/fsps](https://github.com/cconroy20/fsps)). See below for some notes on installing FSPS on NERSC

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
pip install emcee 
pip install speclite 
pip install fsps

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

* add read-in for no-noise photometry in 'gqp_mc.data.Photometry'
* `speclite` package is currently used for photometry, but this should be phased out for the filters in fsps.

## Installing FSPS 
To compile FSPS using ifor modify the `src/Makefile` and comment out 
> F90 = gfortran

and uncomment

> F90 = ifort 

Also, under "Compiler Optimizations" comment out 

> F90FLAGS = -O -cpp 

and add  

> F90FLAGS = -O3 -cpp -fPIC

Alternatively, use

> module swap PrgEnv-intel PrgEnv-gnu

and add

> F90FLAGS = -O -cpp -fPIC
