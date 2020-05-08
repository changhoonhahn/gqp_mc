#!/bin/bash
# example script of running SED fits on the spectra and photometry 
# 
# python run/mini_mocha.py data sim i0 i1 noise method model nthread nwalker burnin niter maxiter overwrite postprocess 
#   args: 
#       1. data: 'photo'=photometry, 'specphoto'=spectra + photometry
#       2. sim: specify simulation 'lgal'
#       3. i0: first galaxy index  
#       4. i1: last galaxy index 
#       5. noise: 'legacy' for photometry; 'bgs0_legacy' for specphoto
#       6. method: SED fitting method 'ispeculator'
#       7. model: specify model 'emulator' 
#       8. nthread: number of cpu 
#       9. nwalker: nubmer of walkers in MCMC 
#       10. burnin: number of iterations for burnin 
#       11. niter: number of iterations. If niter='adaptive' then it uses an
#           adaptive method. 
#       12. maxiter: max number of iterations for MCMC
#       13. overwrite: True/False overwrite MCMC file. 
#       14. postprocess: True/False calculate SFR, MW Z from parameters.

dir="$(dirname "$0")"
# SED fitting for photometry 
python -W ignore $dir/mini_mocha.py photo lgal 0 0 legacy ispeculator emulator 1 20 200 adaptive 3000 True False 
# SED fitting for spectra + photometry 
python -W ignore $dir/mini_mocha.py specphoto lgal 0 0 bg0_legacy ispeculator emulator 1 20 200 adaptive 3000 True False 

# postprocess the chains generated above 
python -W ignore $dir/mini_mocha.py photo lgal 0 0 legacy ispeculator emulator 1 20 200 adaptive 3000 False True 
python -W ignore $dir/mini_mocha.py specphoto lgal 0 0 bg0_legacy ispeculator emulator 1 20 200 adaptive 3000 False True 
