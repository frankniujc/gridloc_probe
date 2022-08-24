import json
import sys
import random
from types import SimpleNamespace
from pathlib import Path

import torch
import numpy as np

from tqdm import tqdm

class TrainingRecord:
    # Epoch starts with 0
    def __init__(self, compare_metric:str, splits=['valid', 'test'], compare_split='valid', print_on_record=False, **kwargs):
        self.training_record = {}
        for split in splits:
            self.training_record[split] = []

        self.compare_metric = compare_metric
        self.compare_split = compare_split

        self.print_on_record = print_on_record

    def record(self, split, epoch_metrics:dict, epoch=None) -> None:
        if epoch == None:
            epoch = len(self.training_record[split])

        if 'epoch' in epoch_metrics:
            assert epoch_metrics['epoch'] == epoch
        else:
            epoch_metrics['epoch'] = epoch

        self.training_record[split].append(epoch_metrics)

        if self.print_on_record:
            self.log_metric(split, epoch_metrics)

    def log_metric(self, split, metric):
        print(f'{split.title()} Result:')
        print(json.dumps(metric, indent=4))

    def is_best_epoch(self) -> bool:
        epoch, _ = max(enumerate(self.training_record[self.compare_split]), key=lambda x: x[1][self.compare_metric])
        return epoch == len(self.training_record[self.compare_split]) - 1

    def get_last_epoch(self) -> dict:
        epoch = {}
        for split in self.training_record:
            epoch[split] = self.training_record[split][-1]
        return epoch

    def get_best_epoch(self) -> dict:
        epoch, _ = max(enumerate(self.training_record[self.compare_split]), key=lambda x: x[1][self.compare_metric])
        best_epoch_result = {}
        for split in self.training_record:
            best_epoch_result[split] = self.training_record[split][epoch]
        best_epoch_result['epoch'] = epoch
        return best_epoch_result

    def print_best_epoch(self) -> None:
        best_epoch = self.get_best_epoch()
        print(f'Best Epoch: {best_epoch["epoch"]}')
        print(json.dumps(best_epoch, indent=4))


class ProbeDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.split()
        self.prepare_label_dict()

    def tokenize(self, tokenizer):
        for dataset in self.datasets.values():
            dataset.tokenize(tokenizer)

    def extract_features(self, model, batch_size, device):
        for dataset in self.datasets.values():
            dataset.extract_features(model, batch_size, device)

    def prepare_label_dict(self):
        labels = set()
        for dataset in self.datasets.values():
            labels = labels.union(dataset.data['labels'])
        label2idx = {}
        idx2label = {}
        for i, label in enumerate(labels):
            label2idx[label] = i
            idx2label[i] = label

        self.label2idx = label2idx
        self.idx2label = idx2label

        for dataset in self.datasets.values():
            dataset.set_label_dict(label2idx, idx2label)

    @property
    def output_size(self):
        return len(self.label2idx)


class ProbeSplitDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.label_ids[idx])
        item['idx'] = idx
        item['senteval_id'] = self.data['senteval_ids'][idx]
        item['text'] = self.data['text'][idx]
        return item

    def set_label_dict(self, label2idx, idx2label):
        self.label2idx = label2idx
        self.idx2label = idx2label


class Experiment:

    def __init__(self, config_file_path, **override_config):
        self.load_config(config_file_path)
        for name, value in override_config.items():
            self.config_dict[name] = value
        self.config = SimpleNamespace(**self.config_dict)
        self.setup_experiment()

    def set_device(self, device=None):
        self.device = torch.device(self.config.device)

    def load_config(self, config_file_path):
        with open(config_file_path) as open_file:
            self.config_dict = json.load(open_file)
            self.config = SimpleNamespace(**self.config_dict)

    def create_save_directory(self):
        output_path = Path(self.config.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

    def save_model(self, suffix='', epoch=None):

        self.create_save_directory()

        output_path = Path(self.config.output_directory) / Path(suffix + '.pt')
        torch.save(self.model.state_dict(), output_path)

        if not epoch:
            epoch = self.training_record.get_best_epoch()
        model_info = {
            'config': self.config_dict,
            'epoch': epoch
        }
        output_path = Path(self.config.output_directory) / Path(suffix + '.json')
        with open(output_path, 'w') as open_file:
            json.dump(model_info, open_file)

    def save_model_info(self, suffix=''):
        best_epoch = self.training_record.get_best_epoch()
        model_info = {
            'config': self.config_dict,
            'best_epoch': best_epoch
        }
        output_path = Path(self.config.output_directory) / Path(suffix + '.json')
        with open(output_path, 'w') as open_file:
            json.dump(model_info, open_file)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def setup_experiment(self):
        self.set_device()
        self.setup_data()
        self.setup_model()
        self.setup_logging()

    def setup_data(self):
        raise NotImplementedError

    def setup_model(self):
        raise NotImplementedError

    def setup_logging(self):
        self.training_record = TrainingRecord('accuracy', **self.config_dict)

    def probe(self):
        raise NotImplementedError

    def train_epoch(self, layer, dataloader):
        raise NotImplementedError

    def evaluate(self, layer, dataset='valid'):
        raise NotImplementedError