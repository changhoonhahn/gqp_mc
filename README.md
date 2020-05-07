# GQP Mock Challenge 

Python package for the GQP mock challenge. The package makes it easy to read in forward modeled DESI-like spectra and photometry and run SED fitting. The package is being actively developed so there will be constant updates! Also, we want this package to be developed as openly as possible so please feel free to contribute via pull requests. 
[toc]

## Updates
**04/30/2020**: LGal data updated. Now with version flag. We're starting with `v1.0`! As long as you git pull and install updates, there shouldn't be any issues. 

## Installation

### on `NERSC`
#### setting up anaconda

```bash
module load python
conda init
```

This will append several lines into your `~/.bashrc` file. After you've done this *once* you do not need to run this again and you can directly activate the conda environment. For details see the nersc [documentation](https://docs.nersc.gov/programming/high-level-environments/python/#using-conda-activate)


#### install the `gqp_mc` package
First, add the following to your `~/.bashrc` or `~/.bashrc.ext` : 

```
export GQPMC_DIR="\SOME_LOCAL_DIRECTORY\" 
export HDF5_USE_FILE_LOCKING=FALSE
```
This defines the `$GQPMC_DIR` environment, which is used in the package and address an `HDF5` i/o issue on `NERSC`. Then run `source ~/.bashrc` or `source ~/.bashrc.ext`on the command line so that the changes take effect. 

Now we're going to symlink to the LGal directory and the directory with the mini-Mock Challenge (`mini_mocha`) in the desi project directory so that we have access to the data.

```
# go to $GQPMC_DIR
cd $GQPMC_DIR
ln -s /global/cfs/cdirs/desi/mocks/LGal_spectra/ Lgal
ln -s /global/cfs/cdirs/desi/mocks/TNG_spectra/ tng 
ln -s /global/cfs/cdirs/desi/mocks/gqp_mini_mocha/ mini_mocha 
```

Your symlinks should point to the proper directory. If the symlinks are bad, fix the symlink, referring to this:
[updating_symlink](https://github.com/kgb0255/GQPMC_v2_JAMES/blob/6da67f918cfadfb17eaa163ddfb25e63dc9b3c53/Documentation/NERSC_Installation/outdated_symlink.md)

You need to install `FSPS` if you want to use the `iFSPS` fitter: [https://github.com/cconroy20/fsps](https://github.com/cconroy20/fsps). See [below](#installing-fsps) for some notes on installing `FSPS` on `NERSC`. (You probably want to install `FSPS`)

With the data all set up, we can now install the package: 
```bash 
# create conda environment 
conda create -n gqp python=3.7 jupyter ipython pip

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
pip install corner

# clone the repo 
git clone https://github.com/changhoonhahn/gqp_mc.git 

# go to project directory
cd gqp_mc 

# install package
python setup.py install

# test the package
pytest 

# create ofiles directory
cd run/cori
mkdir ofiles
```

You're all set. Now you can activate the conda environment by 

```python
conda activate gqp
```
Above is one way to setup the package on nersc. See also Rita's [notebook](https://github.com/ritatojeiro/desi_gqp/blob/master/nb/start_example.ipynb), which details how to install the package. 

If you want to use the CIGALE photometry fitter, you need also to install CIGALE [https://cigale.lam.fr](https://cigale.lam.fr). See below some notes on installing CIGALE on NERSC.

### common/known issues

- Multiprocessing installation might raise following error:
	```python
  ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
  ```

  You can neglect this error, as multiprocessing package has been integrated to python default packages for python 3.X.

- If you encounter following error while installing the packages:

  ```bash
  ERROR: Could not install packages due to an EnvironmentError:
  [Errno 30] Read-only file system: 
  '/global/common/cori_cle7/software/python/3.7-anaconda-2019.10/lib/python3.7/site-packages/...'
  ```

  follow this [link](https://github.com/kgb0255/GQPMC_v2_JAMES/blob/f5e9ec3064c91775e09679a92a67a19ffb80d1c3/Documentation/NERSC_Installation/pacakge_error.md).


## Fitting Spectra or Photometry

*this section needs more detail*

One of the main goals of the mock challenge is to simultaneously fit DESI-like spectra and photometry of the mock challenge galaxies. For examples on reading in the mock challenge data and fitting them see `run/spectral_challenge.py`. 

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


If you modified the `src/Makefile` correctly, it should look like this: [Makefile_example](https://github.com/kgb0255/GQPMC_v2_JAMES/blob/6da67f918cfadfb17eaa163ddfb25e63dc9b3c53/Documentation/NERSC_Installation/Makefile)

## Installing CIGALE

Before compiling add DECam* filters to the CIGALE filter directory 

> cigale/database_builder/filters

Compile CIGALE (tested on v2018)

> python setup.py build
> python setup.py develop

Add CIGALE to your python path

> export PYTHONPATH='${PYTHONPATH}:/your_directory/cigalev2018/'
