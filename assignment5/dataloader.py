#-*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

import numpy as np
from itertools import chain
from collections import Counter, OrderedDict
import pdb
import utils


class NMTSimpleDataset:
    def __init__(self,
                 max_len=20,
                 src_filepath='YOUR/TEXT/FILE/PATH',
                 tgt_filepath=None,
                 vocab=(None, None),
                 is_src=True, is_tgt=True, is_train=True):

        self.max_len = max_len

        src, tgt = [], []

        orig_src, src, vocab = self.load_data(src_filepath, vocab=vocab[0], is_src=is_src, is_train=is_train)
        self.orig_src = orig_src
        self.src = src
        #self.vocab_src = vocab_src
        # self.vocab = vocab

        orig_tgt, tgt, vocab = self.load_data(tgt_filepath, vocab=vocab, is_src=is_tgt, is_train=is_train)
        self.orig_tgt = orig_tgt
        self.tgt = tgt
        self.vocab = vocab
        #self.vocab_tgt = vocab_tgt

    def __getitem__(self, index):
        data, targets = self.src[index], self.tgt[index]
        return data, targets

    def __len__(self):
        return len(self.src)

    def load_data(self, filepath, vocab=None, is_src=True, is_train=True):
        if filepath is None:
            # lines: empty list, seq: fake labels, vocab: vocab
            return [], torch.zeros((self.src.shape[0], self.src.shape[1]+1), dtype=self.src.dtype), vocab

        lines = []
        with open(filepath, 'r', encoding='UTF-8') as f:
            for line in f:
                lines.append(line.strip().split(' '))

        if is_train:
            vocab = self.init_vocab(lines, vocab)

        seqs = self.convert_sent2seq(lines, vocab=vocab, is_src=is_src)

        return lines, seqs, vocab

    def init_vocab(self, sents, vocab):

        vocab = OrderedDict({
            '[PAD]': 2,
            '[UNK]': 3,
            '[SOS]': 0,
            '[EOS]': 1,
        }) if vocab is None else vocab
        n_special_word = len(vocab)
        counter = Counter(list(chain.from_iterable(sents)))
        ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
        for key, _ in ordered_dict.copy().items():
            if key in vocab.keys():
                del ordered_dict[key]
        vocab.update({k: idx + n_special_word for idx, k in enumerate(ordered_dict.keys())})
        return vocab

    def convert_sent2seq(self, sents, vocab=None, is_src=True):
        sent_seq = []
        for s in sents:
            s_pad = utils.padding(s, max_len=self.max_len, is_src=is_src)
            s_seq = []
            for w in s_pad:
                w_mod = w if w in vocab else '[UNK]'
                s_seq.append(vocab[w_mod])
            sent_seq.append(torch.tensor(s_seq, dtype=torch.int64).unsqueeze(0))
        sent_seq = torch.vstack(sent_seq)

        return sent_seq

