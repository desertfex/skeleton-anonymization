from model.msg3d import MultiWindow_MS_G3D, MS_GCN, MS_TCN
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


class Anonymizer(nn.Module):
    def __init__(self,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super().__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 48
        c2 = 48  # c1 * 2     # 192
        # c1 = 96
        # c2 = 96
        # c3 = 3

        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(
            3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(
            c1, c2, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)
        self.gcn3 = MS_GCN(num_gcn_scales, c2, 3,
                           A_binary, disentangled_agg=True)

    def forward(self, x):
        N, C, T, V, M = x.size()  # (batch, xyz, frames, joints, #people)

        # (batch, people * joints * xyz, frames)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        # (batch * people, joints, xyz, frames) -> (batch * people, xyz, frames, joints)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)

        x = self.gcn3(x)
        # print(x.shape)

        out = x
        # (batch, people, xyz, frames, joints)
        out = out.view(N, M, 3, 664, -1)
        out = out.permute(0, 2, 3, 4, 1)

        return out


class Model(nn.Module):
    def __init__(self,
                 num_gender_class,
                 num_action_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super().__init__()

        print(num_gender_class, num_action_class,
              num_point,
              num_person,
              num_gcn_scales,
              num_g3d_scales,
              graph,
              in_channels)
        self.gender_classifier = GenderClassifier(
            num_gender_class, num_point, num_person,
            num_gcn_scales, num_g3d_scales,
            graph, in_channels)

        self.action_classifier = ActionClassifier(
            num_action_class, num_point, num_person,
            num_gcn_scales, num_g3d_scales,
            graph, in_channels)

        self.anonymizer = Anonymizer(
            num_point, num_person,
            num_gcn_scales, num_g3d_scales,
            graph, in_channels)

    def load_action_classifier(self, path):
        self.action_classifier.load_state_dict(torch.load(path))

    def load_gender_classifier(self, path):
        self.gender_classifier.load_state_dict(torch.load(path))

    def forward(self, x):
        anonymized = self.anonymizer(x)

        gender = self.gender_classifier(anonymized)
        action = self.action_classifier(anonymized)

        return [anonymized, gender, action]
