"""
LSTM and GRU temporal models will be used here
"""

import torch
import torch.nn as nn
from torchvision import models


class LSTMModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=256, num_layers=3, dropout=0.5):
        super(LSTMModel, self).__init__()
        """since Cholec80 is a small dataset, we want to bottleneck project
         into a lower-dim latent space in order for model to focus on key features"""

         # note later, if model is underfitting try to open up hidden layer size a little to allow more complexity to be captured
        self.project = nn.Linear(input_size, hidden_size)
        self.model = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, 
                             num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.project(x)
        # list of hidden representations at timestep t, final hidden representation, and cell states
        output, (h, cn) = self.model(x)
        output = self.dropout(output)
        return output


class GRUModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=256, num_layers=3, dropout=0.5):
        super(GRUModel, self).__init__()
        self.project = nn.Linear(input_size, hidden_size)
        self.model = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.project(x)
        output, hn = self.model(x)
        output = self.dropout(output)
        return output
