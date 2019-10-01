#!/bin/bash
#source activate gqp

i0=0
i1=0

#echo 'fitting spectra of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 none 1 10 100 1000 False True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 bgs0 1 10 100 1000 False True

#echo 'fitting photometry of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py photo $i0 $i1 none 1 10 100 1000 False True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py photo $i0 $i1 legacy 1 10 100 1000 False True

#echo 'fitting spectrophotometry of galaxies # '$i0' to '$i1
python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py specphoto $i0 $i1 none 1 20 100 1000 False True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py specphoto $i0 $i1 bgs0_legacy 1 20 100 1000 False True
