#!/bin/bash
#source activate gqp

sim='lgal'
i0=2
i1=2

#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py construct $sim

echo 'fitting galaxies # '$i0' to '$i1
#python -W ignore $HOME/projects/gqp_mc/run/mini_mocha.py \
#    photo $sim $i0 $i1 legacy 1 30 1000 1000 10000 True True

#python -W ignore $HOME/projects/gqp_mc/run/mini_mocha.py \
#    spec $sim $i0 $i1 bgs0 1 30 10 10 100 True True

python -W ignore $HOME/projects/gqp_mc/run/mini_mocha.py \
    specphoto $sim $i0 $i1 bgs0_legacy 1 30 10 10 100 False True
