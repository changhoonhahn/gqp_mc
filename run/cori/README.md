In order for these slurm files to be properly submitted (and to track the status of job that is being run),
`ofiles` directory is **required**. i.e.,

```bash
## Go to ~/gqp_mc/run/cori
mkdir ofiles
```

NERSC will store the output message generated during submitted jobs in this directory as `FILENAME.o`. You can 

```
tail -200 FILENAME.o
```

to view to stored output messages.
