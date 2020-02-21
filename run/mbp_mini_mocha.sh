#!/bin/bash
#source activate gqp

i0=0
i1=96

#echo 'fitting spectra of galaxies # '$i0' to '$i1

# --- iFSPS fitting --- 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 none ifsps 1 10 100 1000 False True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 bgs0 ifsps 1 10 100 1000 False True

#echo 'fitting photometry of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py photo $i0 $i1 none ifsps 1 10 100 1000 False True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py photo $i0 $i1 legacy ifsps 1 100 100 1000 False True

#echo 'fitting spectrophotometry of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py specphoto $i0 $i1 none ifsps 1 20 100 1000 False True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py specphoto $i0 $i1 bgs0_legacy ifsps 1 20 100 1000 False True

# --- pseudoFirefly fitting --- 
python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 none pfirefly 1 10 100 1000 False True
python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 bgs0 pfirefly 1 10 100 1000 False True
