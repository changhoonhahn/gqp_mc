#!/bin/bash
#source activate gqp

i0=0
i1=1

#echo 'fitting spectra of galaxies # '$i0' to '$i1
# no noise; no dust 
#python /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py spec $i0 $i1 none False 3 100 100 1000 False True
# no noise; yes dust 
#python /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py spec $i0 $i1 none True 3 100 100 1000 False 

echo 'fitting photometry of galaxies # '$i0' to '$i1
# no noise; no dust 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py photo $i0 $i1 none False 3 100 100 1000 False
# no noise; yes dust 
python -W ignore /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py photo $i0 $i1 none True 3 100 100 1000 False
