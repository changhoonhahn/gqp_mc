# GQP Mock Challenge 

Python package for the GQP mock challenge. The package makes it easy to read in forward modeled DESI-like spectra and photometry and run SED fitting. The package is being actively developed so there will be constant updates! Also, we want this package to be developed as openly as possible so please feel free to contribute via pull requests. 
[toc]

## Updates
**04/30/2020**: LGal data updated. Now with version flag. We're starting with `v1.0`! As long as you git pull and install updates, there shouldn't be any issues. 

## Fitting Spectra or Photometry

The main goals of the mock challenge is to simultaneously fit DESI-like spectra and photometry of the mock challenge galaxies. For examples running SED fits for photometry or spectra + photometry, check out `run/eg.sh`. 

For examples of submitting SED fit jobs to `NERSC` see slurm scripts in `run/cori/`. We are coordinating job submissions using this [spreadsheet](https://docs.google.com/spreadsheets/d/1pwfZjUi8eREd6YxM0rLaK8LVsVTUdrJ1ctBUfOlZN_8/edit?usp=sharing). 

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
pip install multiprocessing
pip install configobj
pip install sqlalchemy
pip install corner

# install python-fsps from github because gqp_mc repo uses
# the development version 0.3.0 (not the stable PIP version) 
git clone https://github.com/dfm/python-fsps.git
cd python-fsps
python setup.py install

# clone the repo 
git clone https://github.com/changhoonhahn/gqp_mc.git 

# go to project directory
cd gqp_mc 

# install package
pip install -e . 

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

## Installing FSPS 
Follow instructions in https://github.com/cconroy20/fsps/ and https://github.com/cconroy20/fsps/blob/master/doc/INSTALL except when compiling the code: 

- modify the `src/Makefile` and comment out line 10
  ```bash
  F90=gfortran
  ```
  and uncomment line 13
  ```bash
  F90=ifort
  ```
  
- If you get the following error message:
	> autosps.f90(21): error #6353: A RETURN statement is invalid in the main program.
	>    RETURN
	
	modify `RETURN` in [line 21](https://github.com/cconroy20/fsps/blob/master/src/autosps.f90#L21) of `autosps.f90`  to `STOP` and rerun `make`
## Installing CIGALE

Before compiling add DECam* filters to the CIGALE filter directory 

> cigale/database_builder/filters

Compile CIGALE (tested on v2018)

> python setup.py build
> python setup.py develop

Add CIGALE to your python path

> export PYTHONPATH='${PYTHONPATH}:/your_directory/cigalev2018/'
