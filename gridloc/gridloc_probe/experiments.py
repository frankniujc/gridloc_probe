import sys
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast
from sklearn.metrics import f1_score

from tqdm import tqdm

from ..bases import Experiment
from .classifiers import RNN_MLP
from ..sent_eval.datasets import SentEval


class GridLocProbeExperiment(Experiment):

    classifier_class = RNN_MLP
    dataset_class = SentEval
    tokenizer_class = BertTokenizer

    def setup_data(self):
        self.dataset = self.dataset_class(self.config.data_path)
        self.bert_tokenizer = self.tokenizer_class.from_pretrained(self.config.bert_version)
        self.dataset.tokenize(self.bert_tokenizer)

    def setup_model(self):
        local_bert_config_path = Path('configs') / (self.config.bert_version + '.json')
        self.bert_model = BertModel(BertConfig.from_json_file(local_bert_config_path)).to(self.device)
        # Use a local BertConfig to avoid connection errors
        for param in self.bert_model.parameters():
            param.requires_grad = False

        input_size = self.bert_model.embeddings.word_embeddings.embedding_dim
        output_size = self.dataset.output_size
        self.model = self.classifier_class(
            rnn_type='rnn',
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            output_size=output_size,
            dropout=self.config.dropout).to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss()
        optimizers = {
            'adam': torch.optim.Adam,
        }
        self.optimizer = optimizers[self.config.optimizer](self.model.parameters(), lr=self.config.learning_rate)

    def probe(self):
        dataloader = DataLoader(self.dataset.train, batch_size=self.config.batch_size, shuffle=True)

        for i in range(self.config.epochs):
            self.train_epoch(dataloader)
            self.training_record.record('valid', self.evaluate('valid'))
            self.training_record.record('test', self.evaluate('test'))

            self.save_model(suffix=f'epoch_{i}', epoch=self.training_record.get_last_epoch())

        self.save_model_info('best')
        self.training_record.print_best_epoch()

    def train_epoch(self, dataloader):
        self.model.train()

        for i, batch in enumerate(tqdm(dataloader, file=sys.stderr)):

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            label = batch['label'].to(self.device)
            lengths = batch['length']

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            bert_full_hidden = torch.stack(bert_output['hidden_states'][1:], dim=1).detach()
            token_weights, layer_weights, output = self.model(bert_full_hidden, lengths)

            loss = self.loss_function(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, split='valid'):
        dataloader = DataLoader(self.dataset.datasets[split], batch_size=self.config.batch_size, shuffle=True)

        self.model.eval()

        correct = 0
        y_true, y_pred = [], []

        for batch in dataloader:

            gold = batch['label'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            lengths = batch['length']

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            bert_full_hidden = torch.stack(bert_output['hidden_states'][1:], dim=1).detach()
            token_weights, layer_weights, output = self.model(bert_full_hidden, lengths)

            pred = output.argmax(dim=1)
            correct += torch.sum(gold == pred).item()

            y_true.append(gold)
            y_pred.append(pred)

        accuracy = correct / len(self.dataset.datasets[split])
        f1 = f1_score(torch.cat(y_true).cpu(), torch.cat(y_pred).cpu(), average='macro')
        return {'accuracy':accuracy, 'f1':f1}