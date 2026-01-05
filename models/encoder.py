"""
Building the encoder for the images. Using ResNet50 as the encoder.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder,self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # remove FC layer to get flattened embeddings
        self.model.fc = nn.Identity()
        
        # Freeze the backbone
        for param in self.model.parameters():
            param.requires_grad = False
       

    def forward(self, x):
        # explicitly get x shape
        batch_size, seq_len, C, H, W = x.shape

        # since resnet takes in (N,C,H,W), squash the batch size of seq_len, where squashing is row-major
        x = x.view(batch_size * seq_len, C, H, W)
        """ Get embeddings computation"""
        embeddings = self.model(x)
        embeddings = embeddings.view(batch_size, seq_len, -1)

        return embeddings
        
        
        
    


