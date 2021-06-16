from __future__ import print_function,division

import numpy as np
import sys
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score as average_precision

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.utils.data

from prose.utils import pack_sequences, unpack_sequences
from prose.utils import collate_paired_sequences, collate_lists, collate_seq2seq
from prose.utils import infinite_iterator, AllPairsDataset, MultinomialResample
from prose.utils import LargeWeightedRandomSampler
from prose.datasets import SCOPeDataset, SCOPePairsDataset, ContactMapDataset
from prose.datasets import FastaDataset, ClozeDataset

from prose.models.multitask import ProSEMT, OrdinalRegression, BilinearContactMap, L1, L2
from prose.models.lstm import SkipLSTM


def eval_scop(model, test_iterator, use_cuda):
    y = []
    logits = []
    for x0,x1,y_mb in test_iterator:

        b = len(x0)
        x = x0 + x1

        x,order = pack_sequences(x)
        if use_cuda:
            x = x.cuda()
            #x0 = [x_.cuda() for x_ in x0]
            #x1 = [x_.cuda() for x_ in x1]
            y_mb = y_mb.cuda()
        y.append(y_mb.long())

        z = model(x) # embed the sequences
        z = unpack_sequences(z, order)

        z0 = z[:b]
        z1 = z[b:]

        for i in range(b):
            z_a = z0[i]
            z_b = z1[i]
            logits.append(model.score(z_a, z_b))

    y = torch.cat(y, 0)
    logits = torch.stack(logits, 0)
    #p = torch.stack(logits, 0).data

    log_p = F.logsigmoid(logits).data
    log_m_p = F.logsigmoid(-logits).data
    zeros = log_p.new(log_p.size(0),1).zero_()
    log_p_ge = torch.cat([zeros, log_p], 1)
    log_p_lt = torch.cat([log_m_p, zeros], 1)
    log_p = log_p_ge + log_p_lt

    loss = F.cross_entropy(log_p, y).item()

    p = F.softmax(log_p, 1)
    _,y_hard = torch.max(log_p, 1)
    levels = torch.arange(5).to(p.device)
    y_hat = torch.sum(p*levels, 1)

    accuracy = torch.mean((y == y_hard).float()).item()
    mse = torch.mean((y.float() - y_hat)**2).item()

    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    r,_ = pearsonr(y_hat, y)
    rho,_ = spearmanr(y_hat, y)

    ## calculate average-precision score for each structural level
    aupr = np.zeros(4, dtype=np.float32)
    for i in range(4):
        target = (y > i).astype(np.float32)
        aupr[i] = average_precision(target, y_hat)

    return loss, accuracy, mse, r, rho, aupr


def batch_similarity_grad(model, x0, x1, y, use_cuda, weight=1.0):
    b = len(x0)
    x = x0 + x1

    #if use_cuda:
    #    y = y.cuda()
    #    x = [x_.cuda() for x_ in x]

    x,order = pack_sequences(x)
    if use_cuda:
        y = y.cuda()
        x = x.cuda()

    z = model(x) # embed the sequences

    # for memory efficiency
    # we backprop to the representations from each loss pair
    # then backprop through the embedding model

    z_detach = z.data.detach()
    z_detach.requires_grad = True
    z_detach = PackedSequence(z_detach, z.batch_sizes)

    z_unpack = unpack_sequences(z_detach, order)

    z0 = z_unpack[:b]
    z1 = z_unpack[b:]

    logits = torch.zeros_like(y)
    weight = weight/b

    for i in range(b):
        z_a = z0[i]
        z_b = z1[i]
        
        li = model.score(z_a, z_b)
        loss = weight*F.binary_cross_entropy_with_logits(li, y[i])
        loss.backward(retain_graph=True)

        logits[i] = li.detach()

    # now backprop from z
    grad = z_detach.data.grad
    z.data.backward(grad)

    # calculate minibatch performance metrics
    with torch.no_grad():
        y = torch.sum(y.long(), 1)

        log_p = F.logsigmoid(logits)
        log_m_p = F.logsigmoid(-logits)
        zeros = log_p.new(b,1).zero_()
        log_p_ge = torch.cat([zeros, log_p], 1)
        log_p_lt = torch.cat([log_m_p, zeros], 1)
        log_p = log_p_ge + log_p_lt

        loss = F.cross_entropy(log_p, y).item()

        p = F.softmax(log_p, 1)
        _,y_hard = torch.max(log_p, 1)
        levels = torch.arange(5).to(p.device).float()
        y_hat = torch.sum(p*levels, 1)

        correct = torch.sum((y == y_hard).float()).item()
        mse = torch.mean((y.float() - y_hat)**2).item()

    return loss, correct, mse, b


def cmap_grad(model, x, y, use_cuda, weight=1.0):
    b = len(x)
    #if use_cuda:
    #    x = [x_.cuda() for x_ in x]
    x,order = pack_sequences(x)
    if use_cuda:
        x = x.cuda()

    z = model.transform(x) # embed the sequences

    # backprop each sequence individually for memory efficiency
    z_detach = z.data.detach()
    z_detach.requires_grad = True
    z_detach = PackedSequence(z_detach, z.batch_sizes)

    z_unpack = unpack_sequences(z_detach, order)

    # calculate loss for each sequence and backprop
    weight = weight/b

    loss = 0 # loss over minibatch
    tp = 0 # true positives over minibatch
    gp = 0 # number of ground truth positives in minibatch
    pp = 0 # number of predicted positives in minibatch
    total = 0 # total number of residue pairs

    for i in range(b):
        zi = z_unpack[i]
        logits = model.predict(zi.unsqueeze(0)).view(-1) # flattened predicted contacts
        yi = y[i].contiguous().view(-1) # flattened target contacts

        if use_cuda:
            yi = yi.cuda()

        mask = (yi < 0) # unobserved positions
        logits = logits[~mask]
        yi = yi[~mask]

        li = weight*F.binary_cross_entropy_with_logits(logits, yi) # loss for this sequence
        li.backward(retain_graph = True) # backprop to the embeddings

        loss += li.item()
        total += yi.size(0)

        # also calculate the recall and precision
        with torch.no_grad():
            p_hat = torch.sigmoid(logits)
            tp += torch.sum(p_hat*yi).item()
            gp += yi.sum().item()
            pp += p_hat.sum().item()


    # now, backprop the emebedding gradients through the model
    grad = z_detach.data.grad
    z.data.backward(grad)

    return loss, tp, gp, pp, total


def predict_cmap(model, x, y, use_cuda):
    b = len(x)
    #if use_cuda:
    #    x = [x_.cuda() for x_ in x]
    x,order = pack_sequences(x)
    if use_cuda:
        x = x.cuda()

    z = model.transform(x) # embed the sequences
    z = unpack_sequences(z, order)

    logits = []
    y_list = []
    for i in range(b):
        zi = z[i]
        lp = model.predict(zi.unsqueeze(0)).view(-1)

        yi = y[i].contiguous().view(-1)
        if use_cuda:
            yi = yi.cuda()
        mask = (yi < 0)

        lp = lp[~mask]
        yi = yi[~mask]

        logits.append(lp)
        y_list.append(yi)

    return logits, y_list


def eval_cmap(model, test_iterator, use_cuda):
    logits = []
    y = []

    for x,y_mb in test_iterator:
        logits_this, y_this = predict_cmap(model, x, y_mb, use_cuda)
        logits += logits_this
        y += y_this

    y = torch.cat(y, 0)
    logits = torch.cat(logits, 0)

    loss = F.binary_cross_entropy_with_logits(logits, y).item()

    p_hat = torch.sigmoid(logits)
    tp = torch.sum(y*p_hat).item()
    pr = tp/torch.sum(p_hat).item()
    re = tp/torch.sum(y).item()
    f1 = 2*pr*re/(pr + re)            

    y = y.cpu().numpy()
    logits = logits.data.cpu().numpy()

    aupr = average_precision(y, logits)

    return loss, pr, re, f1, aupr


def cloze_grad(model, x, y, use_cuda, weight=1.0):

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
        z = model.transform(x).data
        logits = model.cloze(z)

        # only calculate loss for noised positions
        logits = logits[mask]
        y = y[mask]

        loss = F.cross_entropy(logits, y)

        _,y_hat = torch.max(logits, 1)

        w_loss = loss*weight
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

    # model hyperparameters/architecture settings

    # embedding model architecture
    parser.add_argument('model', nargs='?', help='pretrained model (optional)')
    parser.add_argument('--resume', action='store_true', help='resume training')

    parser.add_argument('--embedding-dim', type=int, default=100, help='embedding dimension (default: 100)')
    parser.add_argument('--rnn-dim', type=int, default=512, help='hidden units of RNNs (default: 512)')
    parser.add_argument('--num-layers', type=int, default=3, help='number of RNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability (default: 0)')


    # for the structural similarity prediction module
    parser.add_argument('--allow-insert', action='store_true', help='model insertions (default: false)')
    parser.add_argument('--norm', choices=['l1', 'l2'], default='l1', help='comparison norm (default: l1)')

    # training parameters

    parser.add_argument('-n', '--num-steps', type=int, default=1000000, help='number ot training steps (default: 1,000,000)')
    parser.add_argument('--save-interval', type=int, default=100000, help='frequency of saving (default:; 100,000)')

    parser.add_argument('--similarity-weight', default=1.0, type=float, help='weight on the similarity task (default: 1)')
    parser.add_argument('--similarity-batch-size', type=int, default=100, help='minibatch size for SCOP similarity loss (default: 100)')


    parser.add_argument('--cloze', type=float, default=1, help='weight on the cloze task (default: 1)')
    parser.add_argument('-p', type=float, default=0.1, help='cloze residue masking rate (default: 0.1)')
    parser.add_argument('--cloze-batch-size', type=int, default=100, help='minibatch size for the cloze loss (default: 100)')

    parser.add_argument('--contacts', type=float, default=1, help='weight on the contact prediction task (default: 1)')
    parser.add_argument('--contacts-batch-size', type=int, default=50, help='minibatch size for contact maps (default: 50)')

    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)')
    parser.add_argument('--tau', type=float, default=0.5, help='smoothing on the similarity sampling distribution (default: 0.5)')
    parser.add_argument('--augment', type=float, default=0, help='resample amino acids during training with this probability (default: 0)')

    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()


    prefix = args.output

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    ## load the datasets

    # 1. SCOPe structural similarity
    path='data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.fa'
    scop_train = SCOPeDataset(path=path)
    scop_test = SCOPePairsDataset()

    # 2. contact maps
    if args.contacts > 0:
        mi = 20
        ma = 1000
        root = 'data/SCOPe/pdbstyle-2.06'
        path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.fa'
        contacts_train = ContactMapDataset(path, root=root, min_length=mi, max_length=ma)

        path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.fa'
        contacts_test = ContactMapDataset(path) #, max_length=max_length)

    # 3. Pfam sequences for cloze dataset
    if args.cloze > 0:
        max_length = 500 # language modeling sequences to have this maximum length
                         # for limiting memory usage during training

        path = 'data/uniprot/uniref90.fasta'

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

    # iterators for the structural similarity data
    batch_size = args.similarity_batch_size

    # make the training data iterators, samples pairs of datapoints
    x_train = scop_train.x
    y_train = scop_train.y
    y_train_levels = torch.cumprod((y_train.unsqueeze(1) == y_train.unsqueeze(0)).float(), 2)

    # make the pairs dataset
    # data augmentation by resampling amino acids
    augment = None
    if args.augment > 0:
        augment = args.augment
        trans = torch.ones(21, 21)
        trans = trans/trans.sum(1, keepdim=True)
        if use_cuda:
            trans = trans.cuda()
        augment = MultinomialResample(trans, augment)
    print('# resampling amino acids with p:', args.augment, file=sys.stderr)

    scop_train = AllPairsDataset(x_train, y_train_levels, augment=augment)
    contacts_train.augment = augment

    # make sampler with tau smoothing
    similarity = y_train_levels.long().numpy().sum(2)
    levels,counts = np.unique(similarity, return_counts=True)
    order = np.argsort(levels)
    levels = levels[order]
    counts = counts[order]

    tau = args.tau
    print('# using tau:', tau, file=sys.stderr)
    print('#', counts/np.sum(counts), file=sys.stderr)
    print('#', counts**tau/np.sum(counts**tau), file=sys.stderr)
    weights = counts**tau/counts
    weights = weights[similarity].ravel()
    sampler = LargeWeightedRandomSampler(weights, batch_size*num_steps)

    # two training dataset iterators for sampling pairs of sequences for training
    sim_train_iterator = torch.utils.data.DataLoader(scop_train
                                                    , batch_size=batch_size
                                                    , sampler=sampler
                                                    , collate_fn=collate_paired_sequences
                                                    )
    sim_train_iterator = infinite_iterator(sim_train_iterator)

    batch_size = 100 # number of pairs per batch for calculating heldout performance
    sim_test_iterator = torch.utils.data.DataLoader(scop_test, batch_size=batch_size
                                                   , collate_fn=collate_paired_sequences
                                                   )

    # iterators for the contact map data
    if args.contacts > 0:
        batch_size = args.contacts_batch_size

        cmap_train_iterator = torch.utils.data.DataLoader(contacts_train
                                                         , batch_size=batch_size
                                                         #, sampler=sampler
                                                         , shuffle=True
                                                         , collate_fn=collate_lists
                                                         )
        #batch_size = 4 # use smaller batch size for calculating test set results
        cmap_test_iterator = torch.utils.data.DataLoader(contacts_test
                                                        , batch_size=batch_size
                                                        , collate_fn=collate_lists
                                                        )

    # iterators for the cloze data
    cloze_iterator = None
    if args.cloze > 0:
        batch_size = args.cloze_batch_size

        # weight each sequence by the number of fragments
        L = np.array([len(x) for x in fasta_train.x])
        weight = np.maximum(L/max_length, 1)
        sampler = LargeWeightedRandomSampler(weight, batch_size*num_steps)

        cloze_iterator = torch.utils.data.DataLoader(cloze_train, batch_size=batch_size
                                                    , sampler=sampler
                                                    , collate_fn=collate_seq2seq
                                                    )


    # weights of similarity loss, contact loss, and cloze LM loss should sum to 1
    cloze = args.cloze
    contacts = args.contacts

    ## initialize the model
    if args.model is not None:
        # load pretrained model
        print('# using pretrained model:', args.model, file=sys.stderr)
        encoder = torch.load(args.model)
    else:
        nin = 21
        nout = 21
        hidden_dim = args.rnn_dim
        num_layers = args.num_layers
        dropout = args.dropout

        encoder = SkipLSTM(nin, nout, hidden_dim, num_layers, dropout=dropout)
       
    resume = args.resume
    step = 0

    if resume:
        model = encoder # we are resuming from pretrained model
        encoder = model.embedding
        # which step are we on?
        path = args.model
        name,_ = os.path.splitext(path)
        it = name.split('_')[-1]
        step = int(it[4:])
    else:
        # encoder is multilayer LSTM with projection layer
        # replace projection layer for structure-based embeddings
        proj = encoder.proj
        encoder.cloze = proj  # keep the projection layer for the cloze task
        
        # make new projection layer for the structure embeddings
        embedding_size = args.embedding_dim

        n_hidden = proj.in_features
        proj = nn.Linear(n_hidden, embedding_size)
        encoder.proj = proj
        encoder.nout = embedding_size

        
        # create the model wrapper for task specific parameters

        allow_insert = args.allow_insert

        if args.norm == 'l1':
            norm = L1()
            print('# norm: l1', file=sys.stderr)
        elif args.norm == 'l2':
            norm = L2()
            print('# norm: l2', file=sys.stderr)

        scop_predict = OrdinalRegression(embedding_size, 5, compare=norm, allow_insertions=allow_insert)
        cmap_predict = None
        if contacts > 0:
            # contact map prediction parameters
            cmap_predict = BilinearContactMap(n_hidden)
        model = ProSEMT(encoder, scop_predict, cmap_predict)

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
    #optim = torch.optim.Adagrad(params, lr=0.0003, weight_decay=weight_decay)

    ## train the model
    print('# training model', file=sys.stderr)

    save_prefix = args.save_prefix
    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')
    digits = int(np.floor(np.log10(num_steps))) + 1
    tokens = ['iter', 'split', 'loss', 'mse', 'accuracy', 'r', 'rho', 'class', 'fold', 'superfamily', 'family']
    if contacts > 0:
        tokens += ['rrc_loss', 'rrc_pr', 'rrc_re', 'rrc_f1', 'rrc_aupr']
    if cloze > 0:
        tokens += ['cloze_loss', 'cloze_perplexity', 'cloze_accuracy']
    line = '\t'.join(tokens)
    print(line, file=output)

    model.train()

    minibatch_iterator = sim_train_iterator
    n = 0
    loss_estimate = 0
    mse_estimate = 0
    acc_estimate = 0

    if contacts > 0:
        # iterate the contacts infinitely
        rrc = infinite_loop(cmap_train_iterator)
        #rrc = iter(cmap_train_iterator)
        cmap_n = 0
        cmap_loss_accum = 0
        cmap_pp = 0
        cmap_pr_accum = 0
        cmap_gp = 0
        cmap_re_accum = 0

    if cloze_iterator is not None:
        cz = iter(cloze_iterator)
        cz_n = 0
        cz_loss = 0
        cz_acc = 0
        
    # normalize the weights
    similarity_weight = args.similarity_weight
    denom = similarity_weight + cloze + contacts

    similarity_weight /= denom
    cloze /= denom
    contacts /= denom

    print('# training with similarity weight={}, contacts weight={}, cloze weight={}'.format(similarity_weight, contacts, cloze), file=sys.stderr)

    save_iter = 100
    save_interval = args.save_interval
    while save_iter <= step:
        save_iter = min(save_iter*10, save_iter+save_interval, num_steps) # next save

    for i in range(step, num_steps):

        # structure similarity minibatch
        x0,x1,y = next(minibatch_iterator)
        loss, correct, mse, b = batch_similarity_grad(model, x0, x1, y, use_cuda
                                                     , weight=similarity_weight)

        n += b
        delta = b*(loss - loss_estimate)
        loss_estimate += delta/n
        delta = correct - b*acc_estimate
        acc_estimate += delta/n
        delta = b*(mse - mse_estimate)
        mse_estimate += delta/n

        # residue-residue contacts loss
        if contacts > 0:
            c_x, c_y = next(rrc)
            loss, tp, gp_, pp_, b = cmap_grad(model, c_x, c_y, use_cuda, weight=contacts)

            cmap_gp += gp_
            delta = tp - gp_*cmap_re_accum
            cmap_re_accum += delta/cmap_gp

            cmap_pp += pp_
            delta = tp - pp_*cmap_pr_accum
            cmap_pr_accum += delta/cmap_pp

            cmap_n += b
            delta = b*(loss - cmap_loss_accum)
            cmap_loss_accum += delta/cmap_n


        # cloze LM minibatch
        if cloze_iterator is not None:
            x,y = next(cz)
            loss,correct,b = cloze_grad(encoder, x, y, use_cuda, weight=cloze)

            cz_n += b 
            delta = b*(loss - cz_loss)
            cz_loss += delta/cz_n
            delta = correct - b*cz_acc
            cz_acc += delta/cz_n

        # clip the gradients if needed
        if not np.isinf(clip):
            # only clip the RNN layers
            nn.utils.clip_grad_norm_(model.embedding.layers.parameters(), clip)

        # parameter update
        optim.step()
        optim.zero_grad()
        model.clip() # projected gradient for bounding ordinal regression parameters

        # report progressive results
        if (i+1) % 10 == 0:
            line = '# [{}/{}] training {:.1%} loss={:.5f}, mse={:.5f}, acc={:.5f}'
            line = line.format(i+1, num_steps, i/num_steps
                              , loss_estimate, mse_estimate, acc_estimate 
                              )
            print(line, end='\r', file=sys.stderr)


        # evaluate and save model
        if i+1 == save_iter:
            save_iter = min(save_iter*10, save_iter+save_interval, num_steps) # next save

            print(' '*80, end='\r', file=sys.stderr)
            tokens = [loss_estimate, mse_estimate, acc_estimate, '-', '-', '-', '-', '-', '-']
            if contacts > 0:
                f1 = 2*cmap_pr_accum*cmap_re_accum/(cmap_pr_accum + cmap_re_accum)
                tokens += [cmap_loss_accum, cmap_pr_accum, cmap_re_accum, f1, '-']
            if cloze > 0:
                cz_perp = np.exp(cz_loss)
                tokens += [cz_loss, cz_perp, cz_acc]
            tokens = [x if type(x) is str else '{:.5f}'.format(x) for x in tokens]
            line = '\t'.join([str(i+1).zfill(digits), 'train'] + tokens)
            print(line, file=output)
            output.flush()

            # reset the accumlation metrics
            n = 0
            loss_estimate = 0
            mse_estimate = 0
            acc_estimate = 0

            if contacts > 0:
                cmap_n = 0
                cmap_loss_accum = 0
                cmap_pp = 0
                cmap_pr_accum = 0
                cmap_gp = 0
                cmap_re_accum = 0

            if cloze > 0:
                cz_n = 0
                cz_loss = 0
                cz_acc = 0

            # eval and save model
            model.eval()

            with torch.no_grad():
                loss, accuracy, mse, r, rho, aupr = eval_scop(model, sim_test_iterator, use_cuda)
                if contacts > 0:
                    cmap_loss, cmap_pr, cmap_re, cmap_f1, cmap_aupr = \
                            eval_cmap(model, cmap_test_iterator, use_cuda)

            tokens = [loss, mse, accuracy, r, rho, aupr[0], aupr[1], aupr[2], aupr[3]]
            if contacts > 0:
                tokens += [cmap_loss, cmap_pr, cmap_re, cmap_f1, cmap_aupr]
            if cloze > 0:
                tokens += ['-', '-', '-']
            tokens = [x if type(x) is str else '{:.5f}'.format(x) for x in tokens]
            line = '\t'.join([str(i+1).zfill(digits), 'test'] + tokens)
            print(line, file=output)
            output.flush()


            # save the model
            if save_prefix is not None:
                save_path = save_prefix + '_iter' + str(i+1).zfill(digits) + '.sav'
                model.cpu()
                torch.save(model, save_path)
                if use_cuda:
                    model.cuda()

            # flip back to train mode
            model.train()


if __name__ == '__main__':
    main()
