import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import dataloader
import utils


from model.transformer import Transformer
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


#7. Training Transformer
def train(dataloader, model, criterion, args, vocab, i2w, optimizer):


	model.train()
	model.zero_grad()

	for epoch in range(args.n_epochs):
		correct = 0
		cnt = 0
		total_score = 0.
		global_step = 0
		tr_loss = 0

		for idx, (src, tgt) in enumerate(dataloader):
			optimizer.zero_grad()

			src, tgt = src.to(device), tgt.to(device)
			output = model(src, tgt[:,:-1])

			output_dim = output.shape[-1]
			outputs = output.contiguous().view(-1, output_dim)
			tgt = tgt[:,1:].contiguous().view(-1)

			loss = criterion(outputs, tgt)
			tr_loss += loss.item()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
			optimizer.step()
			model.zero_grad()
			global_step += 1

			pred = outputs.argmax(dim=1, keepdim=True)
			pred_acc = pred[tgt != 2]
			tgt_acc = tgt[tgt != 2]
			correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

			cnt += tgt_acc.shape[0]
			score = 0.
			with torch.no_grad():
				pred = pred.reshape(args.batch_size, args.max_len, -1).detach().cpu().tolist()
				tgt = tgt.reshape(args.batch_size, args.max_len).detach().cpu().tolist()
				for p, t in zip(pred, tgt):
					eos_idx = t.index(vocab['[PAD]']) if vocab['[PAD]'] in t else len(t)
					p_seq = [i2w[i[0]] for i in p][:eos_idx]
					t_seq = [i2w[i] for i in t][:eos_idx]
					k = args.k if len(t_seq) > args.k else len(t_seq)
					s = utils.bleu_score(p_seq, t_seq, k=k)
					score += s
					total_score += s

			score /= args.batch_size

			print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} acc: {:.4f} BLEU: {:.4f})".format(
				epoch, args.n_epochs, idx + 1, len(dataloader), loss, correct / cnt, score), end=' ')

		tr_loss /= cnt
		tr_acc = correct / cnt
		tr_score = total_score / len(dataloader.dataset) / args.n_epochs

		print("tr: ({:.4f}, {:5.2f}, {:5.2f}) | ".format(tr_loss, tr_acc * 100, tr_score * 100))



def eval(dataloader, model, args, lengths=None):

	model.eval()
	total_pred = []
	idx = 0

	for src, tgt in dataloader:
		src, tgt = src.to(device), tgt.to(device)

		with torch.no_grad():
			enc = model.encoder(src)

		# 6. Auto-regressive Decoder
		tgt = tgt[:, 0].unsqueeze(1)
		for i in range(args.max_len):

			with torch.no_grad():
				output = model.decoder(tgt, src, enc)
				output = model.linear(output)
				output = model.softmax(output)

			pred_token = output.argmax(2)[:, -1].unsqueeze(1)
			tgt = torch.concat((tgt, pred_token), axis=1)

		prev = tgt[:, 1:]
		for i in range(tgt.shape[0]):
			pred = prev[i]
			total_pred.append(pred[:lengths[idx + i]].detach().cpu().numpy())
		idx += args.batch_size

	total_pred = np.concatenate(total_pred)

	return total_pred



def main():

	parser = argparse.ArgumentParser(description='NMT - Transformer')
	""" recommend to use default settings """

	# environmental settings
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--save', action='store_true', default=0)

	# architecture
	parser.add_argument('--num_enc_layers', type=int, default=6, help='Number of Encoder layers')
	parser.add_argument('--num_dec_layers', type=int, default=6, help='Number of Decoder layers')
	parser.add_argument('--num_token', type=int, help='Number of Tokens')
	parser.add_argument('--max_len', type=int, default=20)
	parser.add_argument('--model_dim', type=int, default=512, help='Dimension size of model dimension')
	parser.add_argument('--hidden_size', type=int, default=2048, help='Dimension size of hidden states')
	parser.add_argument('--d_k', type=int, default=64, help='Dimension size of Key and Query')
	parser.add_argument('--d_v', type=int, default=64, help='Dimension size of Value')
	parser.add_argument('--n_head', type=int, default=8, help='Number of multi-head Attention')
	parser.add_argument('--d_prob', type=float, default=0.1, help='Dropout probability')
	parser.add_argument('--max_norm', type=float, default=5.0)
	parser.add_argument('--i_pad', type=int, default=2)

	# hyper-parameters
	parser.add_argument('--n_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--lr', type=float, default=5e-2)
	parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 hyper-parameter for Adam optimizer')
	parser.add_argument('--beta2', type=float, default=0.98, help='Beta2 hyper-parameter for Adam optimizer')
	parser.add_argument('--eps', type=float, default=1e-9, help='Epsilon hyper-parameter for Adam optimizer')
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--teacher-forcing', action='store_true', default=False)
	parser.add_argument('--warmup_steps', type=int, default=78, help='Warmup step for scheduler')
	parser.add_argument('--logging_steps', type=int, default=500, help='Logging step for tensorboard')
	parser.add_argument('--layer_norm_epsilon', type=float, default=1e-12)

	# etc
	parser.add_argument('--k', type=int, default=4, help='hyper-paramter for BLEU score')
	parser.add_argument("--config", default="config.json", type=str, required=False,
						help="config file")
	args = parser.parse_args()

	utils.set_random_seed(args)
	tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
											 src_filepath='./data/de-en/nmt_simple.src.train.txt',
											 tgt_filepath='./data/de-en/nmt_simple.tgt.train.txt',
											 vocab=(None, None),
											 is_src=True, is_tgt=False, is_train=True)
	ts_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
											 src_filepath='./data/de-en/nmt_simple.src.test.txt',
											 tgt_filepath=None,
											 vocab=(tr_dataset.vocab,None),
											 is_src=True, is_tgt=False, is_train=False)



	vocab = tr_dataset.vocab
	i2w = {v: k for k, v in vocab.items()}
	args.num_token = len(tr_dataset.vocab)
	args.i_pad = vocab['[PAD]']


	tr_dataloader = DataLoader(tr_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
	ts_dataloader = DataLoader(ts_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=2)


	model = Transformer(args)
	model.to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.NLLLoss(ignore_index=vocab['[PAD]'])


	train(tr_dataloader, model, criterion, args, vocab, i2w, optimizer)

	with open('./data/de-en/length.npy', 'rb') as f:
		lengths = np.load(f)
	pred = eval(ts_dataloader, model, args, lengths=lengths)

	final = pd.read_csv('result/example_answer.csv')
	final = final.drop(['label'], axis=1)
	final = final.assign(label=pred)
	final.to_csv('result/final_example_answer.csv', index=False)


if __name__ == "__main__":
	main()
















