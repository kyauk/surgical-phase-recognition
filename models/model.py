import torch
import torch.nn as nn
from encoder import ResNet50Encoder
from temporal import LSTMModel, GRUModel 

class SequencingModel(nn.Module):
    def __init__(self):
        super(SequencingModel, self).__init__()
        self.encoder = ResNet50Encoder()
        self.temporal = LSTMModel()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 7 total phase classes
            nn.Linear(128,7)
        )

    def forward(self, x):
        """
        x-dim: (batch_size, seq_len, H, C, W)
        x-dim post encoding: (batch_size, seq_len, -1)
        x-dim post temporal: (batch_size, seq_len, num_phases)
        """

        x = self.encoder(x)
        x = self.temporal(x)
        # raw logits
        logits = self.classifier(x)
        return logits

