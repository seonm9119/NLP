import torch
from torch.nn import init

import math
import random
import collections
import numpy as np
import pickle as pkl


def set_random_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def check_gpu_id(gpu_id):
	if not torch.cuda.is_available():
		return False
	else:
		if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
			return True
		else:
			return False


def read_pkl(filepath):
	with open(filepath, 'rb') as f:
		d = pkl.load(f)
		return d


def padding(tokens, max_len=20, is_src=True):
	w_sos = ['[SOS]']
	w_seq = [tokens[i] if i < len(tokens) else '[PAD]' for i in range(max_len + 1)]

	if is_src:
		w_seq = w_seq[:-1]
	else:
		#w_seq[len(tokens)-1 if len(tokens) <= max_len else max_len-1] = '[EOS]'
		w_seq = w_sos + w_seq[:-1]
		w_seq[len(tokens) + 1 if len(tokens) < max_len  else max_len] = '[EOS]'
	return w_seq


def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight'):
			if init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'uniform':
				init.uniform_(m.weight.data, -0.08, 0.08)
			else:
				# default: normal distribution
				init.normal_(m.weight.data, 0.0, gain)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)

	net.apply(init_func)


def uniform_distribution(shp, lb, rb):
	length = rb - lb if rb > lb else lb - rb
	mean = length / 2.
	return torch.rand(shp) * length - mean


def bleu_score(pred, tgt, k=5):
	"""
		(ref) https://d2l.ai/chapter_recurrent-modern/seq2seq.html
	"""
	len_pred, len_tgt = len(pred), len(tgt)
	score = math.exp(min(0, 1 - len_tgt / len_pred))
	for n in range(1, k+1):
		num_matches, tgt_subs = 0, collections.defaultdict(int)
		for i in range(len_tgt - n + 1):
			tgt_subs[' '.join(tgt[i: i+n])] += 1
		for i in range(len_pred - n + 1):
			if tgt_subs[' '.join(pred[i: i+n])] > 0:
				num_matches += 1
				tgt_subs[' '.join(pred[i: i+n])] -= 1
		score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))

	return score

