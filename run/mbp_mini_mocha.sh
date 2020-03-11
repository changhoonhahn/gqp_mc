#!/bin/bash
#source activate gqp

sim='lgal'
i0=0
i1=96

echo 'fitting galaxies # '$i0' to '$i1
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py construct $sim

# --- iFSPS fitting --- 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    spec $sim $i0 $i1 bgs0 ifsps vanilla 1 10 100 1000 True 
python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
    photo $sim $i0 $i1 legacy ifsps vanilla 1 10 100 1000 False 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ifsps vanilla 1 20 100 1000 True

# --- iSpeculator fitting w/ emulator --- 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    spec $sim $i0 $i1 bgs0 ispeculator NA 1 20 100 1000 True 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    photo $sim $i0 $i1 legacy ispeculator emulator 1 20 100 1000 False
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ispeculator NA 1 40 100 1000 True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ispeculator emulator 1 40 200 4000 True 

# --- iSpeculator fitting w/ fsps --- 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ispeculator fsps 1 40 100 1000 True

# --- iSpeculator fitting w/ fsps complex dust --- 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ispeculator fsps_complexdust 1 40 100 1000 True



# --- pseudoFirefly fitting --- 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 none pfirefly 1 10 100 1000 False True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py spec $i0 $i1 bgs0 pfirefly 1 10 100 1000 False True

# --- iFSPS vanilla_complexdust 
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    spec $sim $i0 $i1 bgs0 ifsps vanilla_complexdust 1 20 100 1000 True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    photo $sim $i0 $i1 legacy ifsps vanilla_complexdust 1 20 100 1000 True
#python -W ignore /Users/ChangHoon/projects/gqp_mc/run/mini_mocha.py \
#    specphoto $sim $i0 $i1 bgs0_legacy ifsps vanilla_complexdust 1 20 100 1000 True
