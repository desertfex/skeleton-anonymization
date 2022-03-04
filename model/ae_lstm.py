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
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(
            input_size=output_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return self.fc(output), hidden, cell


class Anonymizer(nn.Module):
    def __init__(self,
                 num_point,
                 num_person,
                 in_channels=3):
        super().__init__()

        num_features = num_person * num_point * in_channels

        self.encoder = Encoder(num_features, 100)
        self.decoder = Decoder(100, num_features)
        self.teacher_force = False

    def fix_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        N, C, T, V, M = x.size()  # (batch, xyz, frames, joints, #people)

        # (batch, frames, xyz, joints, #people)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C * V * M)
        outputs = torch.zeros(N, T, C * V * M).cuda(x.get_device())
        hidden, cell = self.encoder(x)
        input = torch.zeros(N, 1, C * V * M).cuda(x.get_device())
        for i in range(1, T):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, i, :] = output[:, 0, :]
            input = x[:, i:i+1, :] if self.teacher_force else output

        # (batch, xyz, frames, joints, #people)
        out = outputs.view(N, T, C, V, M).permute(0, 2, 1, 3, 4)

        return out
