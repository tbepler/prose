from __future__ import print_function,division

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

import os
from prose.utils import get_project_root


class SkipLSTM(nn.Module):
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0, bidirectional=True):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = nin
        for i in range(num_layers):
            f = nn.LSTM(dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
            self.layers.append(f)
            if bidirectional:
                dim = 2*hidden_dim
            else:
                dim = hidden_dim

        n = hidden_dim*num_layers + nin
        if bidirectional:
            n = 2*hidden_dim*num_layers + nin

        self.proj = nn.Linear(n, nout)

    @staticmethod
    def load_pretrained(path='prose_dlm'):
        if path is None or path == 'prose_dlm':
            root = get_project_root()
            path = os.path.join(root, 'saved_models', 'prose_dlm_3x1024.sav')

        model = SkipLSTM(21, 21, 1024, 3)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def to_one_hot(self, x):
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def transform(self, x):
        one_hot = self.to_one_hot(x)
        hs =  [one_hot] # []
        h_ = one_hot
        for f in self.layers:
            h,_ = f(h_)
            hs.append(h)
            h_ = h
        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
        return h

    def forward(self, x):
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot

        for f in self.layers:
            h,_ = f(h_)
            hs.append(h)
            h_ = h

        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
            z = self.proj(h.view(-1,h.size(2)))
            z = z.view(x.size(0), x.size(1), -1)

        return z
