from __future__ import print_function,division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.utils.data

from prose.utils import collate_seq2seq
from prose.utils import LargeWeightedRandomSampler
from prose.datasets import FastaDataset, ClozeDataset

from prose.models.lstm import SkipLSTM


def cloze_grad(model, x, y, use_cuda):

    # unpack y
    y = y.data

    if use_cuda:
        x = PackedSequence(x.data.cuda(), x.batch_sizes)
        y = y.cuda()

    mask = (y < 20)
    # check that we have noised positions...
    loss = 0
    correct = 0
    n = mask.float().sum().item()
    if n > 0:
        logits = model(x).data

        # only calculate loss for noised positions
        logits = logits[mask]
        y = y[mask]

        loss = F.cross_entropy(logits, y)

        _,y_hat = torch.max(logits, 1)

        w_loss = loss
        w_loss.backward()

        loss = loss.item()
        correct = torch.sum((y == y_hat).float()).item()

    return loss, correct, n


def infinite_loop(it):
    while True:
        for x in it:
            yield x


def main():
    import argparse
    parser = argparse.ArgumentParser('Script for training multitask embedding model')

    # training dataset
    parser.add_argument('--path-train', default='data/uniprot/uniref90.fasta', help='path to training dataset in fasta format (default: data/uniprot/uniref90.fasta)')

    # embedding model architecture
    parser.add_argument('model', nargs='?', help='pretrained model (optional)')

    parser.add_argument('--resume', action='store_true', help='resume training')

    parser.add_argument('--rnn-dim', type=int, default=512, help='hidden units of RNNs (default: 512)')
    parser.add_argument('--num-layers', type=int, default=3, help='number of RNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability (default: 0)')

    # training parameters
    parser.add_argument('-n', '--num-steps', type=int, default=1000000, help='number ot training steps (default: 1,000,000)')
    parser.add_argument('--save-interval', type=int, default=100000, help='frequency of saving (default:; 100,000)')

    parser.add_argument('--max-length', type=int, default=500, help='sample sequences down to this maximum length during training (default: 500)')

    parser.add_argument('-p', type=float, default=0.1, help='cloze residue masking rate (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=100, help='minibatch size (default: 100)')

    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)')

    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    ## load the dataset
    max_length = args.max_length # language modeling sequences to have this maximum length
                                 # for limiting memory usage during training
    path = args.path_train

    fasta_train = FastaDataset(path, max_length=max_length
                                , debug=args.debug
                                )

    # calculate the distribution over the amino acids
    # to use as the noise distribution
    counts = np.zeros(21)
    for x in fasta_train.x:
        v,c = np.unique(x.numpy(), return_counts=True)
        counts[v] = counts[v] + c
    noise = counts/counts.sum()
    print('# amino acid marginal distribution:', noise, file=sys.stderr)
    noise = torch.from_numpy(noise)
    p = args.p
    cloze_train = ClozeDataset(fasta_train, p, noise)

    ## make the minbatch iterators
    num_steps = args.num_steps
    batch_size = args.batch_size

    # weight each sequence by the number of fragments
    L = np.array([len(x) for x in fasta_train.x])
    weight = np.maximum(L/max_length, 1)
    sampler = LargeWeightedRandomSampler(weight, batch_size*num_steps)

    cloze_iterator = torch.utils.data.DataLoader(cloze_train, batch_size=batch_size
                                                , sampler=sampler
                                                , collate_fn=collate_seq2seq
                                                )

    ## initialize the model
    if args.model is not None:
        # load pretrained model
        print('# using pretrained model:', args.model, file=sys.stderr)
        model = torch.load(args.model)
    else:
        nin = 21
        nout = 21
        hidden_dim = args.rnn_dim
        num_layers = args.num_layers
        dropout = args.dropout

        model = SkipLSTM(nin, nout, hidden_dim, num_layers, dropout=dropout)

    step = 0
    model.train()
    if use_cuda:
        model.cuda()

    ## setup training parameters and optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    clip = args.clip

    print('# training with Adam: lr={}, weight_decay={}'.format(lr, weight_decay), file=sys.stderr)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    ## train the model
    print('# training model', file=sys.stderr)

    save_prefix = args.save_prefix
    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')
    digits = int(np.floor(np.log10(num_steps))) + 1
    tokens = ['iter', 'loss', 'perplexity', 'accuracy']
    line = '\t'.join(tokens)
    print(line, file=output)

    model.train()

    minibatch_iterator = iter(cloze_iterator)
    n = 0
    loss_estimate = 0
    acc_estimate = 0

    save_iter = 100
    save_interval = args.save_interval
    while save_iter <= step:
        save_iter = min(save_iter*10, save_iter+save_interval, num_steps) # next save

    for i in range(step, num_steps):
        x,y = next(minibatch_iterator)
        loss,correct,b = cloze_grad(model, x, y, use_cuda)

        n += b 
        delta = b*(loss - loss_estimate)
        loss_estimate += delta/n
        delta = correct - b*acc_estimate
        acc_estimate += delta/n

        # clip the gradients if needed
        if not np.isinf(clip):
            # only clip the RNN layers
            nn.utils.clip_grad_norm_(model.layers.parameters(), clip)

        # parameter update
        optim.step()
        optim.zero_grad()

        # report progressive results
        if (i+1) % 10 == 0:
            line = '# [{}/{}] training {:.1%} loss={:.5f}, acc={:.5f}'
            line = line.format(i+1, num_steps, i/num_steps
                              , loss_estimate, acc_estimate 
                              )
            print(line, end='\r', file=sys.stderr)


        # save model and report training progress
        if i+1 == save_iter:
            save_iter = min(save_iter*10, save_iter+save_interval, num_steps) # next save

            print(' '*80, end='\r', file=sys.stderr)
            tokens = [loss_estimate, np.exp(loss_estimate), acc_estimate]
            tokens = [x if type(x) is str else '{:.5f}'.format(x) for x in tokens]
            line = '\t'.join([str(i+1).zfill(digits)] + tokens)
            print(line, file=output)
            output.flush()

            # reset the accumlation metrics
            n = 0
            loss_estimate = 0
            acc_estimate = 0

            # save the model
            if save_prefix is not None:
                model.eval()
                save_path = save_prefix + '_iter' + str(i+1).zfill(digits) + '.sav'
                model.cpu()
                torch.save(model, save_path)
                if use_cuda:
                    model.cuda()

            # flip back to train mode
            model.train()


if __name__ == '__main__':
    main()
