`doctor` can run 4 diagnosis.

1. Retrieve medians, 84.1 and 15.9 percentile data of specified galaxies. Can be run with `mcmc` files or `postproc` files.
2. Generate fractional error plots for 3 (photo) or 4 (specphoto) parameters - logM_total log SFR 100Myr logSFR 1Gyr (and f_fiber if specphoto). This requires `postproc` files.
3. Inspect walker distribution to see if any walker has gone out of the piror range. Or, inspect the initial walker distribution whether the walkers are within prior ranges. This generates a log files for entire galaxy set, and flags any galaxy that contains bad walkers.
4. Generate median vs. iteration plots for sampled parameters. This might take substantially long. The thinning parameter can be adjusted within the `doctor.py` file.
