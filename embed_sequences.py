from __future__ import print_function,division

import sys
import numpy as np
import h5py

import torch

from prose.alphabets import Uniprot21
import prose.fasta as fasta


def embed_sequence(model, x, pool='none', use_cuda=False):
    if len(x) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.cuda()

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = model.transform(x)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    return z


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('path')
    parser.add_argument('-m', '--model', default='prose_mt', help='pretrained model to load, prose_mt loads the pretrained ProSE MT model, prose_dlm loads the pretrained Prose DLM model, otherwise unpickles torch model directly (default: prose_mt)')
    parser.add_argument('-o', '--output')
    parser.add_argument('--pool', choices=['none', 'sum', 'max', 'avg'], default='none', help='apply some sort of pooling operation over each sequence (default: none)')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')

    args = parser.parse_args()

    
    path = args.path

    # load the model
    if args.model == 'prose_mt':
        from prose.models.multitask import ProSEMT
        print('# loading the pre-trained ProSE MT model', file=sys.stderr)
        model = ProSEMT.load_pretrained()
    elif args.model == 'prose_dlm':
        from prose.models.lstm import SkipLSTM
        print('# loading the pre-trained ProSE DLM model', file=sys.stderr)
        model = SkipLSTM.load_pretrained()
    else:
        print('# loading model:', args.model, file=sys.stderr)
        model = torch.load(args.model)
    model.eval()

    # set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    if use_cuda:
        model = model.cuda()

    # parse the sequences and embed them
    # write them to hdf5 file
    print('# writing:', args.output, file=sys.stderr)
    h5 = h5py.File(args.output, 'w')

    pool = args.pool
    print('# embedding with pool={}'.format(pool), file=sys.stderr)
    count = 0
    with open(path, 'rb') as f:
        for name,sequence in fasta.parse_stream(f):
            pid = name.decode('utf-8')
            z = embed_sequence(model, sequence, pool=pool, use_cuda=use_cuda)
            # write as hdf5 dataset
            h5.create_dataset(pid, data=z)
            count += 1
            print('# {} sequences processed...'.format(count), file=sys.stderr, end='\r')
    print(' '*80, file=sys.stderr, end='\r')


if __name__ == '__main__':
    main()
