import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
import pandas as pd
import utils, dataloader, lstm


parser = argparse.ArgumentParser(description='NMT - Seq2Seq with Attention')
""" recommend to use default settings """
# environmental settings
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--seed-num', type=int, default=0)
parser.add_argument('--save', action='store_true', default=0)
parser.add_argument('--res-dir', default='../result', type=str)
parser.add_argument('--res-tag', default='seq2seq', type=str)
# architecture
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--max-len', type=int, default=20)
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--max-norm', type=float, default=5.0)
# hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
# option
parser.add_argument('--autoregressive', action='store_true', default=True)
parser.add_argument('--attn', action='store_true', default=True)
# etc
parser.add_argument('--k', type=int, default=4, help='hyper-paramter for BLEU score')
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()


utils.set_random_seed(seed_num=args.seed_num)

use_cuda = utils.check_gpu_id(args.gpu_id)
device = torch.device('cuda:{}'.format(args.gpu_id) if use_cuda else 'cpu')

t_start = time.time()

vocab_src = utils.read_pkl('../data/de-en/nmt_simple.src.vocab.pkl')
vocab_tgt = utils.read_pkl('../data/de-en/nmt_simple.tgt.vocab.pkl')

tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
										 src_filepath='../data/de-en/nmt_simple.src.train.txt',
										 tgt_filepath='../data/de-en/nmt_simple.tgt.train.txt',
										 vocab=(vocab_src, vocab_tgt))
ts_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
										 src_filepath='../data/de-en/nmt_simple.src.test.txt',
										 vocab=(tr_dataset.vocab_src, tr_dataset.vocab_tgt))
vocab_src = tr_dataset.vocab_src
vocab_tgt = tr_dataset.vocab_tgt

i2w_src = {v:k for k, v in vocab_src.items()}
i2w_tgt = {v:k for k, v in vocab_tgt.items()}

tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
ts_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)

encoder = lstm.Encoder(len(vocab_src), args.hidden_size, args.hidden_size, args.num_layers)
if not args.attn:
	decoder = lstm.Decoder(len(vocab_tgt), args.hidden_size, args.hidden_size, args.num_layers)
else:
	decoder = lstm.AttnDecoderRNN(args.hidden_size, len(vocab_tgt))

utils.init_weights(encoder, init_type='uniform')
utils.init_weights(decoder, init_type='uniform')

if args.autoregressive:
	model = lstm.Seq2Seq(encoder, decoder, device).to(device)
else:
	model = lstm.Seq2Seq(encoder, decoder, device, auto=False).to(device)


""" TO DO: (masking) convert this line for masking [PAD] token """
criterion = nn.NLLLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

"TO DO : Train Autoregressive Seq2Seq with Attention model on NMT"
def train(dataloader, epoch):

	tr_loss = 0.
	correct = 0

	cnt = 0
	total_score = 0.
	prev_time = time.time()


	for idx, (src, tgt) in enumerate(dataloader):
		src, tgt = src.to(device), tgt.to(device)

		tgt = tgt.permute(1,0)
		optimizer.zero_grad()


		output = model(src, tgt)
		output = output.reshape(args.batch_size * args.max_len, -1)


		tgt = tgt.reshape(-1)
		loss = criterion(output, tgt)
		tr_loss += loss.item()
		loss.backward()

		""" TO DO: (clipping) convert this line for clipping the 'gradient < args.max_norm' """
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
		optimizer.step()

		# accuracy
		pred = output.argmax(dim=1, keepdim=True)
		pred_acc = pred[tgt != 0]

		tgt_acc = tgt[tgt != 0]
		correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

		cnt += tgt_acc.shape[0]

		# BLEU score
		score = 0.
		with torch.no_grad():
			pred = pred.reshape(args.batch_size, args.max_len, -1).detach().cpu().tolist()
			tgt = tgt.reshape(args.batch_size, args.max_len).detach().cpu().tolist()
			for p, t in zip(pred, tgt):
				eos_idx = t.index(vocab_tgt['[PAD]']) if vocab_tgt['[PAD]'] in t else len(t)
				p_seq = [i2w_tgt[i[0]] for i in p][:eos_idx]
				t_seq = [i2w_tgt[i] for i in t][:eos_idx]
				k = args.k if len(t_seq) > args.k else len(t_seq)
				s = utils.bleu_score(p_seq, t_seq, k=k)
				score += s
				total_score += s

		score /= args.batch_size

		# verbose
		batches_done = (epoch - 1) * len(dataloader) + idx
		batches_left = args.n_epochs * len(dataloader) - batches_done
		time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
		prev_time = time.time()
		print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} (eta: {})".format(
			epoch, args.n_epochs, idx+1, len(dataloader), loss, time_left), end='')

	tr_loss /= cnt
	tr_acc = correct / cnt
	tr_score = total_score / len(dataloader.dataset)


	
	return tr_loss, tr_acc, tr_score

"TODO : Evaluate the performance"
def test(dataloader, lengths=None):
	model.eval()

	idx = 0
	total_pred = []

	with torch.no_grad():
		for _, (src, _) in enumerate(dataloader):
			src = src.to(device)

			optimizer.zero_grad()
			outputs = model(src, mode='test')
			outputs = outputs.permute(1,0,2)

			for i in range(outputs.shape[0]):
				pred = outputs[i].argmax(dim=-1)
				total_pred.append(pred[:lengths[idx+i]].detach().cpu().numpy())

			idx += args.batch_size
	
	total_pred = np.concatenate(total_pred)

	return total_pred


def main():

	for epoch in range(1, args.n_epochs + 1):
		tr_loss, tr_acc, tr_score = train(tr_dataloader, epoch)
		# {format: (loss, acc, BLEU)}
		print("\ntr: ({:.4f}, {:5.2f}, {:5.2f}) | ".format(tr_loss, tr_acc * 100, tr_score * 100))

	print("\n[ Elapsed Time: {:.4f} ]".format(time.time() - t_start))

	#for kaggle: by using 'pred_{}.npy' to make your submission file
	with open('../data/de-en/nmt_simple_len.tgt.test.npy', 'rb') as f:
		lengths = np.load(f)
	pred = test(ts_dataloader, lengths=lengths)

	final = pd.read_csv('../result/nmt_simple.pred.csv')
	final = final.drop(['pred'], axis=1)
	final = final.assign(pred=pred)

	if args.attn:
		final.to_csv('../result/nmt_simple.pred_Autoregressive_attn_tf.csv', index=False)
	else:
		if args.autoregressive:
			final.to_csv('../result/nmt_simple.pred_Autoregressive_tf.csv', index=False)
		else:
			final.to_csv('../result/nmt_simple.pred_Non_autoregressive.csv', index=False)


	print("done")

if __name__ == '__main__':
	main()