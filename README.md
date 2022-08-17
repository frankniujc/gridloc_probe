# Does BERT Rediscover the Classical NLP Pipeline?
Code, data, and results of the [COLING 2022](coling2022.org) paper _Does BERT Rediscover the Classical NLP Pipeline?_

Does BERT store surface knowledge in its bottom layers, syntactic knowledge in its middle layers, and semantic knowledge in its upper layers? In re-examining Jawahar et al. (2019) and Tenney et al.'s (2019) probes into the structure of BERT, we have found that the pipeline-like separation that they were seeking lacks conclusive empirical support. BERT's structure is, however, linguistically grounded, although perhaps in a way that is more nuanced than can be explained by layers alone. We introduce a novel probe, called _GridLoc_, through which we can also take into account token positions, training rounds, and random seeds. Using GridLoc, we are able to detect other, stronger regularities that suggest that pseudo-cognitive appeals to layer depth may not be the preferred mode of explanation for BERT's inner workings.

## Quick Start

### Install Dependencies
Create a virtual environment and install the required dependencies.
```bash
virtualenv env -p python3
source env/bin/activate
pip install -r requirement
```
Install [PyTorch](https://pytorch.org/get-started/locally/).
Please follow PyTorch's official installation guide.
This paper is implemented using PyTorch version `1.10.2+cu102`.

### Data Preparation

Get [SentEval](https://github.com/facebookresearch/SentEval) data.
```
cd data/
git clone https://github.com/facebookresearch/SentEval
cd ..
```

### 

## Citation

## Contact
Email: `{niu,luwenjie,gpenn}@cs.toronto.edu`