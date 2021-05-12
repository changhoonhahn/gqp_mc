#!/bin/bash

pip uninstall fsps
# install with different spectral or isochrone library 
#FFLAGS="-DMIST=0 -DPADOVA=1 -DPARSEC=0 -DBASTI=0 -DGENEVA=0 -DMILES=1 -DBASEL=0" python -m pip install fsps --no-binary fsps
#FFLAGS="-DMIST=0 -DPADOVA=0 -DPARSEC=1 -DBASTI=0 -DGENEVA=0 -DMILES=1 -DBASEL=0" python -m pip install fsps --no-binary fsps
#FFLAGS="-DMIST=0 -DPADOVA=0 -DPARSEC=0 -DBASTI=1 -DGENEVA=0 -DMILES=1 -DBASEL=0" python -m pip install fsps --no-binary fsps
FFLAGS="-DMIST=0 -DPADOVA=0 -DPARSEC=0 -DBASTI=0 -DGENEVA=1 -DMILES=1 -DBASEL=0" python -m pip install fsps --no-binary fsps
#FFLAGS="-DMIST=1 -DPADOVA=0 -DPARSEC=0 -DBASTI=0 -DGENEVA=0 -DMILES=0 -DBASEL=1" python -m pip install fsps --no-binary fsps

# generate SED with this version of FSPS 
python /Users/chahah/projects/gqp_mc/run/sed_other_spec_iso_lib.py

# return FSPS back to original setup 
pip uninstall fsps
FFLAGS="-DMIST=1 -DPADOVA=0 -DPARSEC=0 -DBASTI=0 -DGENEVA=0 -DMILES=1 -DBASEL=0" python -m pip install fsps --no-binary fsps
