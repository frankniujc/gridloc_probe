import torch
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from sklearn.metrics import f1_score

from tqdm import tqdm

from ..bases import Experiment
from .classifiers import MLP
from .datasets import SentEval


class SentEvalExperiment(Experiment):

    def setup_data(self):
        self.dataset = SentEval(self.config.data_path)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.config.bert_version)
        self.dataset.tokenize(self.bert_tokenizer)

    def setup_model(self):
        self.bert_model = BertModel.from_pretrained(self.config.bert_version).to(self.device)
        input_size = self.bert_model.embeddings.word_embeddings.embedding_dim
        output_size = self.dataset.output_size
        self.model = MLP(
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
        for layer in range(1, 13):
            self.probe_layer(layer)

    def probe_layer(self, layer):
        # The number of layer starts with 1. The 0th layer is the embeddings output.

        dataloader = DataLoader(self.dataset.train, batch_size=self.config.batch_size, shuffle=True)
        for i in range(self.config.epochs):
            self.train_epoch(layer, dataloader)
            self.training_record.record('valid', self.evaluate(layer, 'valid'))
            self.training_record.record('test', self.evaluate(layer, 'test'))

        print(f'Layer {layer}:')
        self.training_record.print_best_epoch()

    def train_epoch(self, layer, dataloader):

        for i, batch in enumerate(tqdm(dataloader, file=sys.stderr)):

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            label = batch['label'].to(self.device)

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            cls_layer_output = bert_output['hidden_states'][layer][:,0,:].detach()

            output = self.model(cls_layer_output)
            loss = self.loss_function(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, layer, split='valid'):
        dataloader = DataLoader(self.dataset.datasets[split], batch_size=self.config.batch_size, shuffle=True)

        self.model.eval()

        correct = 0
        y_true, y_pred = [], []

        for batch in dataloader:

            gold = batch['label'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)

            bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            cls_layer_output = bert_output['hidden_states'][layer][:,0,:].detach()

            output = self.model(cls_layer_output)
            pred = output.argmax(dim=1)
            correct += torch.sum(gold == pred).item()

            y_true.append(gold)
            y_pred.append(pred)

        accuracy = correct / len(self.dataset.datasets[split])
        f1 = f1_score(torch.cat(y_true).cpu(), torch.cat(y_pred).cpu())
        return {'accuracy':accuracy, 'f1':f1}
