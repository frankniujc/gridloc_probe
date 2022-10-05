# GridLoc Probe Plots
This tarball contains all the plots generated for the paper _Does BERT Rediscover a Classical NLP Pipeline?_

The plots and results used in the paper are available at https://doi.org/10.5683/SP3/PCZHN4.
This Dataverse repository should contain the following files:
- [`gridloc_checkpoints.tar.gz`](https://borealisdata.ca/file.xhtml?fileId=378579)[1.8GB] contains all the probe model checkpoints and training logs.
- [`gridloc_plots.tar.gz`](https://borealisdata.ca/file.xhtml?fileId=378580)[2.5GB] contains all the plots generated for the paper.

## `gridloc_plots/<TASK>/<SEED>/`

### `layer_weight_center_epoch_X.svg`
The vector file of the layer attention weights plot.

### `layer_weight_center_epoch_X.csv`
The exact attention weights of the 12 layers.

### `layer_weight_center_epoch_X.npy`
The serialised NumPy file containing the layer attention weights of every sentence in the task's test set.

### `top_layer_distribution_X.svg`
The vector file of the plot of the top layer distributions.

## `gridloc_plots/<TASK>/sentences/`
The directory contains the token-position attention weights heat map `sentence_plot_XXXXXX.svg` of the first 20 sentences in the SentEval test set. The sentence ID in the file name directly corresponds to the line number of SentEval data.