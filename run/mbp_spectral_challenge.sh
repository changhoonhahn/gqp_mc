#!/bin/bash
source activate gqp

i0=11
i1=13

#echo 'fitting spectra of galaxies # '$i0' to '$i1
#python /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py spec $i0 $i1 none False 3 100 100 1000 

echo 'fitting photometry of galaxies # '$i0' to '$i1
python -W ignore /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py photo $i0 $i1 none False 3 100 100 1000 
