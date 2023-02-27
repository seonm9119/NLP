import torch
from torch.utils.data import DataLoader

import numpy as np
from itertools import chain
from collections import Counter, OrderedDict

import utils


class NMTSimpleDataset:
	def __init__(self,
				 max_len=20,
				 src_filepath='YOUR/TEXT/FILE/PATH',
				 tgt_filepath=None,
				 vocab=(None, None)):
		
		self.max_len = max_len

		src, tgt = [], []

		orig_src, src, vocab_src = self.load_data(src_filepath, vocab=vocab[0])
		self.orig_src = orig_src
		self.src = src
		self.vocab_src = vocab_src

		orig_tgt, tgt, vocab_tgt = self.load_data(tgt_filepath, vocab=vocab[1], is_train=False)
		self.orig_tgt = orig_tgt
		self.tgt = tgt
		self.vocab_tgt = vocab_tgt
	
	def __getitem__(self, index):
		data, targets  = self.src[index], self.tgt[index]
		return data, targets

	def __len__(self):
		return len(self.src)

	def load_data(self, filepath, vocab=None, is_train=True):
		if filepath is None:
			# lines: empty list, seq: fake labels, vocab: vocab
			return [], torch.zeros((self.src.shape), dtype=self.src.dtype), vocab

		lines = []

		with open(filepath, 'rt', encoding='UTF8') as f:
			for line in f:
				lines.append(line.strip().split(' '))

		#lines = lines[0:10]
		if vocab is None:
			vocab = self.init_vocab(lines)

		seqs = self.convert_sent2seq(lines, vocab=vocab, is_train=is_train)
		
		return lines, seqs, vocab
	
	def init_vocab(self, sents):
		vocab = OrderedDict({
			'[PAD]': 0,
			'[UNK]': 1,
			'[EOS]': 2,
		})
		n_special_word = len(vocab)
		counter = Counter(list(chain.from_iterable(sents)))
		ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
		vocab.update({k: idx+n_special_word for idx, k in enumerate(ordered_dict.keys())})
		return vocab
		
	def convert_sent2seq(self, sents, vocab=None, is_train=True):
		sent_seq = []
		for s in sents:
			s_pad = utils.padding(s, max_len=self.max_len, is_train=is_train)
			s_seq = []
			for w in s_pad:
				w_mod = w if w in vocab else '[UNK]'
				s_seq.append(vocab[w_mod])
			sent_seq.append(torch.tensor(s_seq, dtype=torch.int64).unsqueeze(0))
		sent_seq = torch.vstack(sent_seq)

		return sent_seq
		
