#### `doctor.py`

4 diagnoses available.

1. Retrieve medians, 84.1 and 15.9 percentile data of specified galaxies from `.mcmc` files or `.postproc` files. The output will be stored at `~/doctor/data_list/data_retrieval/{model}/{sim}.{spec_or_photo}.noise_{noise}.{model}.{median or upsig or losig}.npy`, where the last entry will have one of `median`, `upsig` (+1 sigma), or `losig` (-1 sigma).

2. Generate fractional error plots for 3 (photo) or 4 (specphoto) parameters - logM_total log SFR 100Myr logSFR 1Gyr (and f_fiber if `spec_or_photo=specphoto`). This requires `postproc` files. The output will be stored at `~/doctor/data_list/diagnostic/{model}/{sim}.{spec_or_photo}.noise_{noise}.{model}.{igal}.diagnostic.pdf`.

3. Inspect walker distribution to see if any walker has gone out of the piror range. Or, inspect the initial walker distribution whether the walkers are within prior ranges. This generates a log files for entire galaxy set, and flags any galaxy that contains bad walkers. The log file will be stored at `~/doctor/data_list/walker_log/{sim}.{spec_or_photo}.noise_{noise}.{model}.walker.log`.

4. Generate median vs. iteration plots for sampled parameters. This might take substantially long. The thinning parameter can be adjusted within the `doctor.py` file. The output will be stored at `~/doctor/data_list/param_plots/{model}/{sim}.{spec_or_photo}.noise_{noise}.{model}.{igal}.param_plots.pdf`.

#### `run_diagnosis.sh`

```bash
data_dir=/global/cscratch1/sd/kgb0255/gqp_mc/mini_mocha/ispeculator/james/fsps/
sim=lgal
spec_or_photo=specphoto
noise=bgs0_legacy
model=fsps
igal=all
```

You can specify igal as comma-separated integers (with no space; e.g. 1,2,3,4) or as all (from 0 to 96). Comment out the operation you don't need. 
