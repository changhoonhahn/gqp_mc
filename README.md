# GQP Mock Challenge 

Python package for the GQP mock challenge. Currently includes functions
that make it easy to read in forward modeled Lgal spectra and photometry. 
Please feel free to add in implementations of your favorite spectral or
photometry fitter and submit a pull request! 


## Table of Conents 
[Installiation](#installation)<br>
[Fitting Spectra/Photometry](#fitting-spectra-or-photometry)<br> 
[To do](#to-do)<br>
[Installing FSPS](#installing-fsps)<br> 

## Installation

### on NERSC

Below is one way to setup the package on nersc. See also Rita's notebook:
[start_example.ipynb](https://github.com/ritatojeiro/desi_gqp/blob/master/nb/start_example.ipynb)

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

Now we're going to symlink to the LGal directory and the directory with the 
mini-Mock Challenge (mini_mocha) in the desi project directory so that we
have access to the data
```
# go to $GQPMC_DIR
cd $GQPMC_DIR
ln -s /global/cfs/cdirs/desi/mocks/LGal_spectra/ Lgal
ln -s /global/cfs/cdirs/desi/mocks/TNG_spectra/ tng 
ln -s /global/cfs/cdirs/desi/mocks/gqp_mini_mocha/ mini_mocha 
```

You need to install FSPS if you want to use the iFSPS fitter ([https://github.com/cconroy20/fsps](https://github.com/cconroy20/fsps)). See below for some notes on installing FSPS on NERSC
Yoy need also to install CIGALE if you want to use CIGALE photometry fitter ([https://cigale.lam.fr](https://cigale.lam.fr). See below some notes on installing CIGALE on NERSC

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
pip install multiprocessing
pip install configobj
pip install sqlalchemy

# clone the repo 
git clone https://github.com/changhoonhahn/gqp_mc.git 

# go to project directory
cd gqp_mc 

# install package
python setup.py install --user 

# test the package
pytest 
```

## Fitting Spectra or Photometry

One of the main goals of the mock challenge is to fit spectra and photometry of 
the mock challenge galaxies. For examples on reading in the mock challenge data
and fitting them see `run/spectral_challenge.py`. 

## To do 

* ~~add read-in for no-noise photometry in 'gqp_mc.data.Photometry'~~
* implement joint fitting of spectra and photometry 
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

## Installing CIGALE

Before compiling add DECam* filters to the CIGALE filter directory 

> cigale/database_builder/filters

Compile CIGALE (tested on v2018)

> python setup.py build
> python setup.py develop

Add CIGALE to your python path

> export PYTHONPATH='${PYTHONPATH}:/yoour_directory/cigalev2018/'
