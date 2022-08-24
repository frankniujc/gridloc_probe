import torch

from tqdm import tqdm

from transformers import BertModel, BertTokenizer

from ..bases import ProbeDataset, ProbeSplitDataset


class SentEvalDataset(ProbeSplitDataset):
    def __init__(self, data):
        self.data = data
        self.feature_extracted = False

    def tokenize(self, tokenizer):
        encodings = tokenizer(self.data['text'], padding=True, return_length=True)
        self.encodings = encodings
        self.label_ids = [self.label2idx[label] for label in self.data['labels']]


class SentEval(ProbeDataset):
    dataset_class = SentEvalDataset

    def split(self):
        splits = {
            'tr': {'text':[], 'labels':[], 'senteval_ids':[]}, # train
            'va': {'text':[], 'labels':[], 'senteval_ids':[]}, # validation
            'te': {'text':[], 'labels':[], 'senteval_ids':[]}, # test
        }

        with open(self.path) as open_file:
            i = 0
            for line in open_file:
                if line:
                    split, label, text = line.strip().split('\t')
                    splits[split]['text'].append(text)
                    splits[split]['labels'].append(label)
                    splits[split]['senteval_ids'].append(i)
                    i += 1

        tr_dataset = self.dataset_class(splits['tr'])
        va_dataset = self.dataset_class(splits['va'])
        te_dataset = self.dataset_class(splits['te'])

        self.train = tr_dataset
        self.valid = va_dataset
        self.test = te_dataset

        self.datasets = {
            'train': tr_dataset,
            'valid': va_dataset,
            'test': te_dataset,
        }


class SentEvalDatasetFast(SentEvalDataset):
    def tokenize(self, tokenizer):
        encodings = tokenizer([x.split() for x in self.data['text']], padding=True, return_length=True, return_offsets_mapping=True, is_split_into_words=True)
        self.encodings = encodings
        self.label_ids = [self.label2idx[label] for label in self.data['labels']]


class SentEvalFast(SentEval):
    dataset_class = SentEvalDatasetFast