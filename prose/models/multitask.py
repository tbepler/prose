from __future__ import print_function,division

import numpy as np
import torch
import torch.nn as nn

import os
from prose.utils import get_project_root


class ProSEMT(nn.Module):
    def __init__(self, embedding, scop_predict, cmap_predict):
        super(ProSEMT, self).__init__()
        self.embedding = embedding
        self.scop_predict = scop_predict
        self.cmap_predict = cmap_predict

    @staticmethod
    def load_pretrained(path='prose_mt'):
        if path is None or path == 'prose_mt':
            root = get_project_root()
            path = os.path.join(root, 'saved_models', 'prose_mt_3x1024.sav')

        from prose.models.lstm import SkipLSTM
        encoder = SkipLSTM(21, 21, 1024, 3)
        encoder.cloze = encoder.proj

        proj_in = encoder.proj.in_features
        proj = nn.Linear(proj_in, 100)
        encoder.proj = proj
        encoder.nout = 100

        scop_predict = OrdinalRegression(100, 5, compare=L1(), allow_insertions=False)
        cmap_predict = BilinearContactMap(proj_in)
        model = ProSEMT(encoder, scop_predict, cmap_predict)

        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def clip(self):
        self.scop_predict.clip()
        self.cmap_predict.clip()

    def forward(self, x):
        return self.embedding(x)

    def transform(self, x):
        return self.embedding.transform(x)

    def score(self, z_x, z_y):
        return self.scop_predict(z_x, z_y)

    def predict(self, z):
        return self.cmap_predict(z)


class ConvContactMap(nn.Module):
    def __init__(self, embed_dim, hidden_dim=50, width=7, act=nn.ReLU()):
        super(ConvContactMap, self).__init__()
        self.hidden = nn.Conv2d(2*embed_dim, hidden_dim, 1)
        self.act = act
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width//2)
        self.clip()

    def clip(self):
        # force the conv layer to be transpose invariant
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5*(w + w.transpose(2,3))

    def forward(self, z):
        return self.predict(z)

    def predict(self, z):
        # z is (b,L,d)
        z = z.transpose(1, 2) # (b,d,L)
        z_dif = torch.abs(z.unsqueeze(2) - z.unsqueeze(3))
        z_mul = z.unsqueeze(2)*z.unsqueeze(3)
        z = torch.cat([z_dif, z_mul], 1)
        # (b,2d,L,L)
        h = self.act(self.hidden(z))
        logits = self.conv(h).squeeze(1)
        return logits


class BilinearContactMap(nn.Module):
    """
    Predicts contact maps as sigmoid(z_i W z_j + b)
    """
    def __init__(self, embed_dim, hidden_dim=1000, width=7, act=nn.LeakyReLU()):
        super(BilinearContactMap, self).__init__()

        self.scale = np.sqrt(hidden_dim)
        self.linear = nn.Linear(embed_dim, embed_dim) #, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def clip(self):
        pass

    def forward(self, z):
        return self.predict(z)

    def predict(self, z):
        z_flat = z.view(-1, z.size(2))
        h = self.linear(z_flat).view(z.size(0), z.size(1), -1)
        s = torch.bmm(h, z.transpose(1,2))/self.scale + self.bias
        return s


def pad_gap_scores(s, gap):
    col = gap.expand(s.size(0), 1)
    s = torch.cat([s, col], 1)
    row = gap.expand(1, s.size(1))
    s = torch.cat([s, row], 0)
    return s


class L1(nn.Module):
    def forward(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1)-y), -1)


class L2(nn.Module):
    def forward(self, x, y):
        return -torch.sum((x.unsqueeze(1)-y)**2, -1)


class OrdinalRegression(nn.Module):
    def __init__(self, embed_dim, n_classes, compare=L1()
                , transform=None, align_method='ssa', beta_init=None
                , allow_insertions=False, gap_init=-10
                ):
        super(OrdinalRegression, self).__init__()
        
        self.n_in = embed_dim
        self.n_out = n_classes

        self.compare = compare
        self.align_method = align_method
        self.allow_insertions = allow_insertions
        self.gap = nn.Parameter(torch.FloatTensor([gap_init]))
        self.transform = transform


        if beta_init is None:
            # set beta to expectation of comparison
            # assuming embeddings are unit normal

            if type(compare) is L1:
                ex = 2*np.sqrt(2/np.pi)*embed_dim # expectation for L1
                var = 4*(1 - 2/np.pi)*embed_dim # variance for L1
            elif type(compare) is L2:
                ex = 4*embed_dim # expectation for L2
                var = 32*embed_dim # variance for L2
            else:
                ex = 0
                var = embed_dim
                
            beta_init = ex/np.sqrt(var)

        self.theta = nn.Parameter(torch.ones(1,n_classes-1)/np.sqrt(var))
        self.beta = nn.Parameter(torch.zeros(n_classes-1)+beta_init)

        self.clip()

    def clip(self):
        # clip the weights of ordinal regression to be non-negative
        self.theta.data.clamp_(min=0)

    def forward(self, z_x, z_y):
        return self.score(z_x, z_y)

    def score(self, z_x, z_y):

        s = self.compare(z_x, z_y)
        if self.allow_insertions:
            s = pad_gap_scores(s, self.gap)

        if self.align_method == 'ssa':
            a = torch.softmax(s, 1)
            b = torch.softmax(s, 0)

            if self.allow_insertions:
                index = s.size(0)-1
                index = s.data.new(1).long().fill_(index)
                a = a.index_fill(0, index, 0)

                index = s.size(1)-1
                index = s.data.new(1).long().fill_(index)
                b = b.index_fill(1, index, 0)

            a = a + b - a*b
            a = a/torch.sum(a)
        else:
            raise Exception('Unknown alignment method: ' + self.align_method)

        a = a.view(-1,1)
        s = s.view(-1,1)

        if hasattr(self, 'transform'):
            if self.transform is not None:
                s = self.transform(s)

        c = torch.sum(a*s)
        logits = c*self.theta + self.beta
        return logits.view(-1)
