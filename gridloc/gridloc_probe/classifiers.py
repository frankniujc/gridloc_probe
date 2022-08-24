import torch
import torch.nn as nn

from ..sent_eval.classifiers import MLP

class RNN_MLP(torch.nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, output_size, dropout):
        super().__init__()

        rnn_class = {
            'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN,
        }

        self.rnn = rnn_class[rnn_type](
            input_size,
            hidden_size,
            num_layers = 1,
            dropout=dropout,
        )

        self.token_weights_decoder = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.layer_weights_decoder = nn.Linear(input_size, 1)

        self.mlp = MLP(input_size, hidden_size, output_size, dropout)

    def forward(self, input_hidden, sentence_lengths, h_pad=-float('inf')):

        # Batch, Layer, Sequence length, bert Hidden
        B, L, S, H = input_hidden.shape

        input_hidden = input_hidden.view(B*L, S, H)

        x = torch.nn.utils.rnn.pack_padded_sequence(
            input_hidden,
            sentence_lengths.repeat_interleave(L),
            enforce_sorted=False, batch_first=True)
        token_weights_hidden, _ = self.rnn(x)
        token_weights_hidden = torch.nn.utils.rnn.pad_packed_sequence(
            token_weights_hidden, batch_first=True)[0]
        token_weights_logit = self.token_weights_decoder(token_weights_hidden).squeeze()

        pad_mask = torch.arange(token_weights_logit.shape[1], device=token_weights_logit.device)
        pad_mask = pad_mask.repeat(token_weights_logit.shape[0], 1)
        pad_mask = pad_mask >= sentence_lengths.repeat_interleave(L).unsqueeze(-1).to(token_weights_logit.device)
        token_weights_logit = token_weights_logit.masked_fill(pad_mask, -float('inf'))
        token_weights = self.softmax(token_weights_logit)
        
        input_hidden = input_hidden[:, :token_weights.shape[1], :]
        layers = input_hidden * token_weights.unsqueeze(-1)
        layers = torch.sum(layers, dim=1).view(B, L, H)

        layer_weights = self.layer_weights_decoder(layers)
        layer_weights = self.softmax(self.layer_weights_decoder(layers))
        final = torch.sum(layers * layer_weights, dim=1)

        return token_weights, layer_weights, self.mlp(final)