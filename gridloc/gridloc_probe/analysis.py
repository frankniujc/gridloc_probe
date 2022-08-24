from pathlib import Path
import json
import itertools

import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizerFast

from scipy import stats
from scipy.stats import combine_pvalues, pearsonr, pointbiserialr
from scipy.optimize import curve_fit

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .experiments import GridLocProbeExperiment
from ..sent_eval.datasets import SentEvalFast


class Analysis:

    def load_checkpoint(self, path):
        path = Path(path)
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        with open(path.with_suffix('.json')) as open_file:
            self.checkpoint_info = json.load(open_file)

    def expand_batch(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        token_type_ids = batch['token_type_ids'].to(self.device)
        label = batch['label'].to(self.device)
        lengths = batch['length']

        return input_ids, attention_mask, token_type_ids, label, lengths


class GridLocProbeExperimentAnalysis(GridLocProbeExperiment, Analysis):

    def top_layer_distribution(self, split='test', plot_directory='plots', name='top_layer_distribution'):
        self.model.eval()
        dataloader = DataLoader(self.dataset.datasets[split], batch_size=self.config.batch_size, shuffle=True)

        modals = []

        for i, batch in enumerate(tqdm(dataloader)):
            input_ids, attention_mask, token_type_ids, label, lengths = self.expand_batch(batch)

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            bert_full_hidden = torch.stack(bert_output['hidden_states'][1:], dim=1).detach()
            token_weights, layer_weights, output = self.model(bert_full_hidden, lengths)

            modals += torch.argmax(layer_weights.squeeze(), dim=1).tolist()

        fig, ax = plt.subplots()
        ax.set_title(f'average: {np.average(modals)} std: {np.std(modals)}')
        sns.barplot(ax=ax, y=[modals.count(x) for x in range(12)], x=np.arange(1,13))
        Path(plot_directory).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(plot_directory) / Path(f'{name}.svg'), format='svg')

        return np.average(modals), np.std(modals)

    def layer_weights_center(self, split='test', plot_directory='plots', name='layer_weight_center'):
        self.model.eval()
        dataloader = DataLoader(self.dataset.datasets[split], batch_size=self.config.batch_size, shuffle=True)

        total_weights = torch.zeros(12).to(self.device)

        Path(plot_directory).mkdir(parents=True, exist_ok=True)
        npy_file = open(Path(plot_directory) / Path(f'{name}.npy'), 'wb')

        for i, batch in enumerate(tqdm(dataloader)):
            input_ids, attention_mask, token_type_ids, label, lengths = self.expand_batch(batch)

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            bert_full_hidden = torch.stack(bert_output['hidden_states'][1:], dim=1).detach()
            token_weights, layer_weights, output = self.model(bert_full_hidden, lengths)

            total_weights += layer_weights.squeeze().sum(dim=0).detach()

            np.save(npy_file, layer_weights.squeeze().detach().cpu().numpy())

        total_weights /= len(self.dataset.datasets[split])
        npy_file.close()

        with open(Path(plot_directory) / Path(f'{name}.csv'), 'w') as data_file:
            data_file.write(','.join(str(x) for x in total_weights.tolist()))

        fig, ax = plt.subplots()

        sns.barplot(ax=ax, y=total_weights.cpu().numpy(), x=np.arange(1,13))
        ax.set_xlabel('Layer')
        ax.set_ylabel('Attention Weight')
        epoch_info = self.checkpoint_info['epoch']
        ax.set_title(f'performance: valid:{epoch_info["valid"]["accuracy"]:.3f}({epoch_info["valid"]["f1"]:.3f}) test:{epoch_info["test"]["accuracy"]:.3f}({epoch_info["test"]["f1"]:.3f})')
        fig.savefig(Path(plot_directory) / Path(f'{name}.svg'), format='svg', bbox_inches='tight')


    def plot_sentences(self, num_sentences, multiplied=False,
            split='test', seed=1234, plot_directory='plots'):
        self.model.eval()

        dataloader = DataLoader(self.dataset.datasets[split], batch_size=min(self.config.batch_size, num_sentences), shuffle=False)

        i = 0

        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, label, lengths = self.expand_batch(batch)

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            bert_full_hidden = torch.stack(bert_output['hidden_states'][1:], dim=1).detach()
            token_weights, layer_weights, output = self.model(bert_full_hidden, lengths)

            for j, sent_layer_weight in enumerate(layer_weights.squeeze().tolist()):
                token = (
                    token_weights.view(
                        layer_weights.shape[0],
                        layer_weights.shape[1],
                        -1)[j],
                    layer_weights[j],
                    input_ids[j],
                    lengths[j],
                )
                Path(plot_directory).mkdir(parents=True, exist_ok=True)
                eval_sent_id = batch['senteval_id'][j].item()
                self.plot_weights('multi_'*multiplied + f'sentence_plot_{eval_sent_id}', token, plot_directory=plot_directory, multiplied=multiplied, eval_sent_id=eval_sent_id)

                i += 1
                if i >= num_sentences:
                    return

    def plot_weights(self, name, token, plot_directory='plots', multiplied=False, eval_sent_id=0):
        fig, ax = plt.subplots(figsize=(13,10))
        token_weights, layer_weights, input_ids, lengths, *_ = token
        if multiplied:
            weights = token_weights * layer_weights
            weights = weights[:,:lengths].flip(0) * 100
        else:
            weights = token_weights[:,:lengths].flip(0) * 100
        sns.heatmap(
            weights.detach().cpu(),
            ax=ax, annot=True, fmt='.3f',
            yticklabels=np.arange(12,0,-1),
            cbar=False,
            vmin=0., vmax=100.)

        tokens = self.bert_tokenizer.convert_ids_to_tokens(input_ids[:lengths])
        ax.set_xticks(np.arange(len(tokens)) + 0.5, labels=tokens)
        ax.xaxis.set_tick_params(labelsize=max(10, 50 / np.sqrt(lengths.item())))
        ax.set_ylabel('Layer')
        ax.set_title(f'token position attention: sentence {eval_sent_id}')

        fig.savefig(Path(plot_directory) / Path(f'{name}.svg'), format='svg', bbox_inches='tight')
        print(f"figure saved at {Path(plot_directory) / Path(f'{name}.svg')}")
        plt.close(fig)


class TreeDepthAnalysis(GridLocProbeExperiment, Analysis):

    dataset_class = SentEvalFast
    tokenizer_class = BertTokenizerFast

    def prepare_stanza(self):
        import stanza
        self.stanza = stanza.Pipeline('en', tokenize_pretokenized=True)

    def plot_tree_depth(self, num_sentences, name='tree_depth_plot',
            split='test', seed=1234, plot_directory='plots'):
        self.model.eval()

        if 'stanza' not in self.__dict__:
            self.prepare_stanza()

        dataloader = DataLoader(self.dataset.datasets[split], batch_size=self.config.batch_size, shuffle=False)

        i = 0

        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, label, _ = self.expand_batch(batch)

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            bert_full_hidden = torch.stack(bert_output['hidden_states'][1:], dim=1).detach()
            lengths = torch.sum(batch['attention_mask'], dim=1)
            token_weights, layer_weights, output = self.model(bert_full_hidden, lengths)

            for j, sent_layer_weight in enumerate(layer_weights.squeeze().tolist()):
                input_id = input_ids[j]
                length = lengths[j]
                offset_mapping = batch['offset_mapping'][j]

                stanza_doc = self.stanza(batch['text'][j])
                words = []  # stanza called it word
                for sentence in stanza_doc.sentences:
                    for word in sentence.words:
                        words.append(word)

                def find_depth(word, words):
                    if word.deprel == 'root':
                        return 0
                    else:
                        return 1 + find_depth(words[word.head-1], words)

                for word in words:
                    word.depth = find_depth(word, words)

                offsets = np.ones(len(offset_mapping),dtype=int) * -100
                offsets[(offset_mapping[:,0]==0) & (offset_mapping[:,1] != 0)] = torch.arange(len(words))
                sentence_token_weights = token_weights.view(
                    layer_weights.shape[0],
                    layer_weights.shape[1],
                    -1)[j]

                data = (sentence_token_weights, layer_weights[j], input_ids[j], length, offsets, words)
                senteval_id = batch['senteval_id'][j].item()

                self.plot_weights(name + f'_{senteval_id}', data, plot_directory=plot_directory)
                i += 1

                if i >= num_sentences:
                    return

    def analyse_tree_depth(self, split='test'):
        self.model.eval()

        if 'stanza' not in self.__dict__:
            self.prepare_stanza()

        dataloader = DataLoader(self.dataset.datasets[split], batch_size=self.config.batch_size, shuffle=False)

        pos = []
        depths = {}

        for bb, batch in enumerate(tqdm(dataloader)):
            input_ids, attention_mask, token_type_ids, label, _ = self.expand_batch(batch)

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            bert_full_hidden = torch.stack(bert_output['hidden_states'][1:], dim=1).detach()
            lengths = torch.sum(batch['attention_mask'], dim=1)
            token_weights, layer_weights, output = self.model(bert_full_hidden, lengths)

            for j, sent_layer_weight in enumerate(layer_weights.squeeze().tolist()):
                input_id = input_ids[j]
                length = lengths[j]
                offset_mapping = batch['offset_mapping'][j]

                stanza_doc = self.stanza(batch['text'][j])
                words = []  # stanza called it word
                for sentence in stanza_doc.sentences:
                    for word in sentence.words:
                        words.append(word)

                offsets = np.ones(len(offset_mapping),dtype=int) * -100
                offsets[(offset_mapping[:,0]==0) & (offset_mapping[:,1] != 0)] = torch.arange(len(words))
                sentence_token_weights = token_weights.view(
                    layer_weights.shape[0],
                    layer_weights.shape[1],
                    -1)[j]

                data = (sentence_token_weights, layer_weights[j], input_ids[j], length, offsets, words)
                max_layers = torch.argmax(sentence_token_weights[:,:length], dim=0)
                prev = None

                for max_layer, offset in zip(max_layers[1:-1], offsets[1:-1]):
                    # Excluding [CLS] and [SEP]
                    if offset >= 0:
                        if words[offset].upos not in depths:
                            depths[words[offset].upos] = []
                        depths[words[offset].upos].append(max_layer.item())

        for pos, lst in depths.items():
            print(pos, len(lst), np.average([x for x in lst]))

        for pos1, pos2 in itertools.combinations(depths.keys(), 2):
            pos1_depths = [x for x in depths[pos1]]
            pos2_depths = [x for x in depths[pos2]]

            pbr =  pointbiserialr(pos1_depths+pos2_depths, [0]*len(pos1_depths) + [1]*len(pos2_depths))

            print(pos1, pos2, pbr[0], pbr[1])