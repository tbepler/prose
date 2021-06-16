"""
Copyright (C) Tristan Bepler - All Rights Reserved
Author: Tristan Bepler <tbepler@gmail.com>
"""

from __future__ import print_function, division

import sys
import os
import glob
import random
from PIL import Image
import numpy as np
import pandas as pd

import torch

from prose.alphabets import Uniprot21
import prose.scop as scop
import prose.fasta as fasta


class SCOPeDataset:
    def __init__(self, path='data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.fa'
                , alphabet=Uniprot21(), augment=None):
        print('# loading SCOP sequences:', path, file=sys.stderr)

        self.augment = augment

        names, structs, sequences = self.load(path, alphabet)
            
        self.names = names
        self.x = [torch.from_numpy(x) for x in sequences]
        self.y = torch.from_numpy(structs)

        print('# loaded', len(self.x), 'sequences', file=sys.stderr)


    def load(self, path, alphabet):
        with open(path, 'rb') as f:
            names, structs, sequences = scop.parse_astral(f, encoder=alphabet)    
        # make sure no sequences of length 0 are included
        names_filtered = []
        structs_filtered = []
        sequences_filtered = []
        for i in range(len(sequences)):
            s = sequences[i]
            if len(s) > 0:
                names_filtered.append(names[i])
                structs_filtered.append(structs[i])
                sequences_filtered.append(s)
        names = names_filtered
        structs = np.stack(structs_filtered, 0)
        sequences = sequences_filtered

        return names, structs, sequences


    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        x = self.x[i].long()
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[i]


class SCOPePairsDataset:
    def __init__(self, path='data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.sampledpairs.txt'
                , alphabet=Uniprot21()):
        print('# loading SCOP sequence pairs:', path, file=sys.stderr)

        table = pd.read_csv(path, sep='\t')
        x0 = [x.encode('utf-8').upper() for x in table['sequence_A']]
        self.x0 = [torch.from_numpy(alphabet.encode(x)) for x in x0]
        x1 = [x.encode('utf-8').upper() for x in table['sequence_B']]
        self.x1 = [torch.from_numpy(alphabet.encode(x)) for x in x1]

        self.y = torch.from_numpy(table['similarity'].values).long()

        print('# loaded', len(self.x0), 'sequence pairs', file=sys.stderr)


    def __len__(self):
        return len(self.x0)


    def __getitem__(self, i):
        return self.x0[i].long(), self.x1[i].long(), self.y[i]


class ContactMapDataset:
    def __init__(self, path, root='data/SCOPe/pdbstyle-2.06'
                , k=1, min_length=0, max_length=0
                , alphabet=Uniprot21()
                , augment=None
                ):

        names, sequences, contact_maps = self.load(path, root, k=k)

        self.names = names
        self.x = [torch.from_numpy(alphabet.encode(s)) for s in sequences]
        self.y = contact_maps

        self.augment = augment

        self.min_length = min_length
        self.max_length = max_length

        self.fragment = False
        if self.min_length > 0 and self.max_length > 0:
            self.fragment = True

        print('# loaded', len(self.x), 'contact maps', file=sys.stderr)


    def load(self, path, root, k=1):
        print('# loading contact maps:', root, 'for sequences:', path, file=sys.stderr)

        with open(path, 'rb') as f:
            names,sequences = fasta.parse(f)

        # find all of the contact maps and index them by protein identifier
        cmap_paths = glob.glob(root + os.sep + '*' + os.sep + '*.png')
        cmap_index = {os.path.basename(path).split('.cmap-')[0] : path for path in cmap_paths}

        # match the sequences to the contact maps
        names_filtered = []
        sequences_filtered = []
        contact_maps = []
        for (name,seq) in zip(names, sequences):
            name = name.decode('utf-8')
            pid = name.split()[0]
            if pid not in cmap_index:
                # try changing first letter to 'd'
                # required for some SCOPe identifiers
                pid = 'd' + pid[1:]
            path = cmap_index[pid]
            # load the contact map image
            im = np.array(Image.open(path), copy=False)
            contacts = np.zeros(im.shape, dtype=np.float32)
            # set the positive, negative, and masked residue pairs
            contacts[im == 1] = -1
            contacts[im == 255] = 1
            # mask the matrix below the kth diagonal
            mask = np.tril_indices(contacts.shape[0], k=k)
            contacts[mask] = -1

            # filter out empty contact matrices
            if np.any(contacts > -1):
                contact_maps.append(torch.from_numpy(contacts))
                names_filtered.append(name)
                sequences_filtered.append(seq)

        return names_filtered, sequences_filtered, contact_maps


    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]

        mi_length = self.min_length
        ma_length = self.max_length
        if self.fragment and len(x) > mi_length:
            l = np.random.randint(mi_length, ma_length+1)
            if len(x) > l:
                i = np.random.randint(len(x)-l+1)
                xl = x[i:i+l]
                yl = y[i:i+l,i:i+l]
            else:
                xl = x
                yl = y
            # make sure there are unmasked observations
            while torch.sum(yl >= 0) == 0:
                l = np.random.randint(mi_length, ma_length+1)
                if len(x) > l:
                    i = np.random.randint(len(x)-l+1)
                    xl = x[i:i+l]
                    yl = y[i:i+l,i:i+l]
            y = yl.contiguous()
            x = xl

        x = x.long()
        if self.augment is not None:
            x = self.augment(x)

        return x, y


class FastaDataset:
    def __init__(self, path, max_length=0, alphabet=Uniprot21(), debug=False):

        print('# loading fasta sequences:', path, file=sys.stderr)
        with open(path, 'rb') as f:
            if debug:
                count = 0
                names = []
                sequences = []
                for name,sequence in fasta.parse_stream(f):
                    if count > 10000:
                        break
                    names.append(name)
                    sequences.append(sequence)
                    count += 1
            else:
                names,sequences = fasta.parse(f)

        self.names = names
        self.x = [torch.from_numpy(alphabet.encode(s)) for s in sequences]

        self.max_length = max_length

        print('# loaded', len(self.x), 'sequences', file=sys.stderr)


    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        x = self.x[i]
        max_length = self.max_length
        if max_length > 0 and len(x) > max_length:
            # randomly sample a subsequence of length max_length
            j = random.randint(0, len(x) - max_length)
            x = x[j:j+max_length]
        return x.long()


class ClozeDataset:
    def __init__(self, x, p, noise):
        self.x = x
        self.p = p
        self.noise = noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        p = self.p
        n = len(self.noise) # number of tokens

        # create the random mask... i.e. which positions to infer
        mask = torch.rand(len(x), device=x.device)
        mask = (mask < p).long() # we mask with probability p

        y = mask*x + (1-mask)*(n-1) # assign unmasked positions to (n-1)

        # sample the masked positions from the noise distribution
        noise = torch.multinomial(self.noise, len(x), replacement=True)
        x = (1-mask)*x + mask*noise

        return x, y



