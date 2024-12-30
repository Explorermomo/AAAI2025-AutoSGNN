#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import torch.nn.functional as F

class LLM_FILTER(torch.nn.Module):
    def __init__(self, in_channel,out_channel,Model_layer):
        super(LLM_FILTER, self).__init__()
        in_channels = in_channel
        out_channels = out_channel
        self.conv1 = Model_layer(64, 64)
        self.conv2 = Model_layer(64, 64)
        self.lin0 = torch.nn.Linear(in_channel,64)
        self.lin00 = torch.nn.Linear(in_channels, out_channels)
        self.lin1 = torch.nn.Linear(64, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin0(x))
        x_raw = x 
        x1 = self.conv1(x, x_raw, edge_index)
        x2 = self.conv2(x1, x_raw, edge_index) 
        x = F.dropout(x2, p=0.5, training=self.training)
        x = self.lin1(x)

        return F.log_softmax(x, dim=1)
    
