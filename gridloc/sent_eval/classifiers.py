import torch
import torch.nn as nn

def MLP(input_size, hidden_size, output_size, dropout):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Dropout(p=dropout),
        nn.Sigmoid(),
        nn.Linear(hidden_size, output_size),
    )
