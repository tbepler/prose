from __future__ import print_function,division

import numpy as np
from pathlib import Path

import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def pack_sequences(X, order=None):
    
    #X = [x.squeeze(0) for x in X]
    
    n = len(X)
    lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]
        order = np.ascontiguousarray(order)
    m = max(len(x) for x in X)

    if len(X[0].size()) > 1:
        d = X[0].size(1)
        X_block = X[0].new(n,m,d).zero_()
        
        for i in range(n):
            j = order[i]
            x = X[j]
            X_block[i,:len(x),:] = x
    else:  
        X_block = X[0].new(n,m).zero_()
        
        for i in range(n):
            j = order[i]
            x = X[j]
            X_block[i,:len(x)] = x
        
    #X_block = torch.from_numpy(X_block)
        
    lengths = lengths[order]
    X = pack_padded_sequence(X_block, lengths, batch_first=True)
    
    return X, order


def unpack_sequences(X, order):
    X,lengths = pad_packed_sequence(X, batch_first=True)
    X_block = [None]*len(order)
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i,:lengths[i]]
    return X_block


def infinite_iterator(it):
    while True:
        for x in it:
            yield x


class LargeWeightedRandomSampler(torch.utils.data.sampler.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def collate_seq2seq(args):
    x,y = zip(*args)
    x = list(x)
    y = list(y)

    x,order = pack_sequences(x)
    y,_ = pack_sequences(y, order=order)

    return x,y

def collate_lists(args):
    x = [a[0] for a in args]
    y = [a[1] for a in args]
    return x, y


class AllPairsDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment=None):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)**2

    def __getitem__(self, k):
        n = len(self.X)
        i = k//n
        j = k%n

        x0 = self.X[i].long()
        x1 = self.X[j].long()
        if self.augment is not None:
            x0 = self.augment(x0)
            x1 = self.augment(x1)

        y = self.Y[i,j]
        #y = torch.cumprod((self.Y[i] == self.Y[j]).long(), 0).sum()

        return x0, x1, y


def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)


class MultinomialResample:
    def __init__(self, trans, p):
        self.p = (1-p)*torch.eye(trans.size(0)).to(trans.device) + p*trans

    def __call__(self, x):
        #print(x.size(), x.dtype)
        p = self.p[x] # get distribution for each x
        return torch.multinomial(p, 1).view(-1) # sample from distribution
