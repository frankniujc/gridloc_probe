# GridLoc Probe Checkpoints
This tarball contains all the probe model checkpoints used in the paper _Does BERT Rediscover the Classical NLP Pipeline?_

The plots and results used in the paper is available at https://doi.org/10.5683/SP3/PCZHN4.
This Dataverse repository should contain the following files:
- [`gridloc_checkpoints.tar.gz`](https://borealisdata.ca/file.xhtml?fileId=378579)[1.8GB] contains all the probe model checkpoints and training logs.
- [`gridloc_plots.tar.gz`](https://borealisdata.ca/file.xhtml?fileId=378580)[2.5GB] contains all the plots generated for the paper.

The organisation of this directory is simple:
```bash
gridloc_checkpoints/<TASK>/<SEED>/
```

## `best.json`
Within each seed's directory, there is a `best.json` file showing the training log of the best performing epoch.

## `epoch_x.json`
The JSON file is the training log of the model after the epoch concluded.

## `epoch_x.pt`
The `.pt` file is the checkpoint of the model after the epoch concluded.  You can load the model using the `gridloc.gridloc_probe.analysis.Analysis.load_checkpoint` function.