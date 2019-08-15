#!/bin/bash
source activate gqp

for i in {0..4}; do
    echo 'spectral fitting '$i 
    python /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py spec $i none False
done
for i in {0..4}; do
    echo 'photometric fitting '$i 
    python /Users/ChangHoon/projects/gqp_mc/run/spectral_challenge.py photo $i none False
done 
