from model.msg3d import Model as ActionClassifier
from model.gender_classifier import Model as GenderClassifier
from utils import import_class, count_params
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import sys
sys.path.insert(0, '')


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, embedding_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(embedding_dim, 100)
        self.fc2 = nn.Linear(100, 500)
        self.fc3 = nn.Linear(500, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Anonymizer(nn.Module):
    def __init__(self,
                 num_point,
                 num_person,
                 in_channels=3):
        super().__init__()

        num_features = num_person * num_point * in_channels

        self.encoder = Encoder(num_features, 10)
        self.decoder = Decoder(10, num_features)

    def fix_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        N, C, T, V, M = x.size()  # (batch, xyz, frames, joints, #people)

        # (batch, frames, xyz, joints, #people)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C * V * M)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # (batch, xyz, frames, joints, #people)
        out = decoded.view(N, T, C, V, M).permute(0, 2, 1, 3, 4)

        return out
