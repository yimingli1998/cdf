# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yiming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activate):
        super(NN, self).__init__()
        self.layer_in = nn.Linear(input_dim, hidden_dim[0])
        self.layer_hidden = nn.ModuleList(
            [nn.Linear(hidden_dim[i], hidden_dim[i + 1]) for i in range(len(hidden_dim) - 1)])
        self.layer_out = nn.Linear(hidden_dim[-1], output_dim)
        self.activate = activate

    def forward(self, x):
        # x = (x - torch.tensor([-torch.pi, -torch.pi, -torch.pi, -torch.pi], dtype=torch.float32, device='cuda')) \
        #     / torch.tensor([2*torch.pi, 2*torch.pi, 2*torch.pi, 2*torch.pi], dtype=torch.float32, device='cuda')
        x = self.activate(self.layer_in(x))
        for i in range(len(self.layer_hidden)):
            x = self.activate(self.layer_hidden[i](x))
        x = self.layer_out(x)
        x = torch.sigmoid(x)

        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


