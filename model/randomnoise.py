import torch
import torch.nn as nn


class Anonymizer(nn.Module):
    def __init__(self, mean=0., std=1.):
        super(Anonymizer, self).__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        return x + torch.randn(x.size()).cuda(x.device) * self.std + self.mean
