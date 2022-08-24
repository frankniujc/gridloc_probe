# Does BERT Rediscover the Classical NLP Pipeline?
Code, data, and results of the [COLING 2022](coling2022.org) paper _Does BERT Rediscover the Classical NLP Pipeline?_

**Abstract**: Does BERT store surface knowledge in its bottom layers, syntactic knowledge in its middle layers, and semantic knowledge in its upper layers? In re-examining Jawahar et al. (2019) and Tenney et al.'s (2019) probes into the structure of BERT, we have found that the pipeline-like separation that they were seeking lacks conclusive empirical support. BERT's structure is, however, linguistically grounded, although perhaps in a way that is more nuanced than can be explained by layers alone. We introduce a novel probe, called _GridLoc_, through which we can also take into account token positions, training rounds, and random seeds. Using GridLoc, we are able to detect other, stronger regularities that suggest that pseudo-cognitive appeals to layer depth may not be the preferred mode of explanation for BERT's inner workings.

## Plots and Results

The plots and results used in the paper is available at https://doi.org/10.5683/SP3/PCZHN4.
- `gridloc_checkpoints.zip`[1.8GB] contains all the probe model checkpoints and training logs.

## Quick Start

### Install Dependencies
Create a virtual environment and install the required dependencies.
```bash
virtualenv env -p python3
source env/bin/activate
pip install -r requirements.txt
```
Install [PyTorch](https://pytorch.org/get-started/locally/).
Please follow PyTorch's official installation guide.
This paper is implemented using PyTorch version `1.10.2+cu102`.

Install [Stanza](https://stanfordnlp.github.io/stanza/) English model. (Required only for TreeDepth analysis.)
```
python -c "import stanza; stanza.download('en')"
```

### Data Preparation

Get [SentEval](https://github.com/facebookresearch/SentEval) data.
```bash
mkdir data
cd data/
git clone https://github.com/facebookresearch/SentEval
cd ..
```

### Train Probes
Initiate the training process by running the script `scripts/train_probes.py`.
```bash
python scripts/train_probes.py
```

### Plot Layer Attention Weight
The script `scripts/plot_layer_weight_centers.py` contains the code to compute the layer attention weight distributions used in section 5.1 and 5.2 of the paper.
```bash
python scripts/plot_layer_weight_centers.py
```
The script will generate 3 files for each epoch of the probe: an `.svg` file, a `.csv` file, and an `.npy` file.
- `layer_weight_center_epoch_X.svg`: The vector file of the layer attention weights plot.
- `layer_weight_center_epoch_X.csv`: The exact attention weights of the 12 layers.
- `layer_weight_center_epoch_X.npy`: The serialised NumPy file containing the layer attention weights of every sentence in the task's test set.  See `scripts/read_npy.py` for more details on how to load and read the results.

Similarly, the script `scripts/plot_top_layer_distribution.py` contains the code to compute the top layer distributions used in section 5.1 and section 5.2 of the paper.
```bash
python scripts/plot_layer_weight_centers.py
```
The script will only generate a `top_layer_distribution_X.svg` file containing the plot of the top layer distributions.

### Plot Sentences
The script `scripts/plot_layer_weight_centers.py` contains the code to plot token-position attention weights of sentences.  These plots are used in section 5.3 and section 5.4 of the paper.
```bash
python scripts/plot_sentences.py
```

The script will generate a `sentence_plot_XXXXXX.svg` file containing the token-position attention weights heat map.  The sentence ID in the file name directly corresponds to the line number of SentEval data.

### TreeDepth Correlation Analysis
The script `scripts/tree_depth_analysis.py` contains the code to compute the average layer of different POSs and the correlation analysis of section 5.4 of the paper.

You may need to change the paths in the script to match your directory organisation.

```bash
python scripts/tree_depth_analysis.py
```

## Citation

## Contact
Email: `{niu,luwenjie,gpenn}@cs.toronto.edu`