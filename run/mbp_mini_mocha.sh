#!/bin/bash
#source activate gqp

sim='tng'
i0=0
i1=0

python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py construct $sim
# --- iFSPS fitting --- 
#echo 'fitting spectra of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 bgs0 ifsps 1 10 100 1000 True
#echo 'fitting photometry of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py photo $i0 $i1 legacy ifsps 1 10 100 1000 True
#echo 'fitting spectrophotometry of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py specphoto $i0 $i1 bgs0_legacy ifsps 1 20 100 1000 True

# --- iSpeculator fitting --- 
#echo 'iSpeculator fitting spectra of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 bgs0 ispeculator 1 20 100 1000 True
#echo 'fitting photometry of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py photo $i0 $i1 legacy ispeculator 1 20 100 1000 True
#echo 'fitting spectrophotometry of galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py specphoto $i0 $i1 bgs0_legacy ispeculator 1 40 100 1000 True


# --- pseudoFirefly fitting --- 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 none pfirefly 1 10 100 1000 False True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 bgs0 pfirefly 1 10 100 1000 False True
