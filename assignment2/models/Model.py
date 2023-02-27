import torch
import math
import torch.nn as nn
from models.Embedding import Embedding
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        filter_sizes = [2,3,4]
        self.embedding = Embedding()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=3000,
                      out_channels=100,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.layer = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(100))
        self.fc1 = nn.Linear(len(filter_sizes) * 100, 100)
        self.fc2 = nn.Sequential(nn.Linear(100, 6), nn.Softmax(dim=1))

    def forward(self, text):

        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)

        conved = [self.layer(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)

        fc1 = self.fc1(cat)
        out = self.fc2(fc1)


        return out


