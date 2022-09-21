# -*- coding: utf-8 -*-
"""
@Time : 2022/1/7 15:14 
@Author : Chao Zhu
"""
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, device, rnn_type, input_size, hidden_size=64, num_layers=1, dropout=0, bidirectional=False):
        super().__init__()
        self.device = device
        self.rnn_type = rnn_type
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        if self.rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError('Unknown RNN type: {}'.format(self.rnn_type))

        self.fc = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=1)

    def forward(self, batch, device):
        src = batch['park_x'].to(device)  # [batch_size, seq_len, feature_dim]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim]
        # h_0 = [layers * num_directions, batch_size, hidden_size]
        h_0 = torch.zeros(self.layers * self.num_directions, src.shape[1], self.hidden_size).to(device)
        if self.rnn_type == 'LSTM':
            c_0 = torch.zeros(self.layers * self.num_directions, src.shape[1], self.hidden_size).to(device)
            out, (hn, cn) = self.rnn(src, (h_0, c_0))
            # out = [seq_len, batch_size, hidden_size * num_directions]
            # hn/cn = [layers * num_directions, batch_size, hidden_size]
        else:
            out, hn = self.rnn(src, h_0)
            # out = [seq_len, batch_size, hidden_size * num_directions]
            # hn = [layers * num_directions, batch_size, hidden_size]

        out_last = out[-1].unsqueeze(0).permute(1, 0, 2)  # [batch_size, out_len = 1, hidden_size * num_directions]
        out = self.fc(out_last)  # out = [batch_size, 1, 1]

        return out
