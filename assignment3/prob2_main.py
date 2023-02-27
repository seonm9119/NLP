import pandas as pd
import argparse
import numpy as np
import torch

from utils import load_word_file, load_label_dic, load_pos_data, LabelSmoothingLoss
from Models.RNN import RNNmodel

parser = argparse.ArgumentParser(description='NLP-lab3-prob2')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--output_size', default=18, type=int, metavar='N',
                    help='output size')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--num_layers', default=3, type=int,
                    help='number of RNN layer')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='RNN hidden-size')
parser.add_argument('--max_len', default=20, type=int,
                    help='lenght of sequence')
parser.add_argument('--test_len', default=65, type=int,
                    help='lenght of sequence')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')



def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def CrossEntropyLoss_with_mask(outputs, targets, mask):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs*mask)/num_examples

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, fx, bx, y):

        self.fx = fx
        self.bx = bx
        self.y = y

    def __len__(self):
        return len(self.fx)

    def __getitem__(self, idx):
        fx = torch.LongTensor(self.fx[idx])
        bx = torch.LongTensor(self.bx[idx])
        y = torch.LongTensor(self.y[idx])
        return fx, bx, y

def post_padding_n_truncation(sentence, vocabulary, max_len):

    # 2. Post-padding & pre-sequence truncation
    inputs = []
    for tokens in sentence:
        row = []
        for w in tokens:
            # 2-1. pre-sequence truncation
            if len(row) < max_len:
                try:
                    row.append(vocabulary[w])
                except KeyError:
                    row.append(vocabulary['[UNK]'])
        # 2-2. post-padding
        row += [0] * (max_len - len(tokens))
        inputs.append(row)

    inputs = np.asarray(inputs)

    return inputs

def train(args,train_loader,model, optimizer):


    for epoch in range(args.epochs):
        loss_epoch = 0
        for step, (fx, bx, y) in enumerate(train_loader):

            fx = fx.cuda()
            bx = bx.cuda()
            tags = y.cuda()

            optimizer.zero_grad()
            predictions = model(fx, bx)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            #Masking
            mask = fx.clone()
            mask[mask > 0] = True
            mask = mask.view(-1).cuda()

            loss = CrossEntropyLoss_with_mask(predictions, tags, mask)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}")


def evaluate_model(model, dictionary, max_len):

    tokens, _ = load_pos_data('dataset/pos/pos_datasets/test_set.json')

    tokens = post_padding_n_truncation(tokens, dictionary, max_len)
    test_dataset = torch.LongTensor(tokens).cuda()

    pred = model(test_dataset)
    top_predictions = pred.argmax(-1).to('cpu')
    top_predictions = top_predictions.numpy()

    idx = np.where(tokens > 0)
    idx = top_predictions[idx]

    final = pd.read_csv('result/pos_class.pred.csv')
    final = final.drop(['label'], axis=1)
    final = final.assign(label=idx)
    final.to_csv('result/pos_class.pred.csv', index=False)

def flipping_data(sentence, vocabulary, max_len):

    inputs = []
    for tokens in sentence:
        row = []
        for w in tokens:
            if len(row) < max_len:
                try:
                    row.append(vocabulary[w])
                except KeyError:
                    row.append(vocabulary['[UNK]'])
        # flipping data
        row = list(reversed(row))
        row += [0] * (max_len - len(tokens))
        inputs.append(row)

    inputs = np.asarray(inputs)

    return inputs

def main():

    args = parser.parse_args()

    # Use given Fasttext word embedding
    dictionary, embedding_dic, args.vectors = load_word_file('dataset/pos/pos_datasets/fasttext_word.json')
    tokens, ud_tags = load_pos_data('dataset/pos/pos_datasets/train_set.json')
    ud_dict = load_label_dic()

    # post-padding & pre-sequence truncation
    f_input = post_padding_n_truncation(tokens, dictionary, args.max_len)

    # Bidirectional input flipping
    b_input = flipping_data(tokens, dictionary, args.max_len)
    ud_tags = post_padding_n_truncation(ud_tags, ud_dict, args.max_len)

    # Training model
    train_dataset = CustomDataset(f_input, b_input, ud_tags)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

    model = RNNmodel(args, bidirectional=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.weight_decay)

    train(args,train_loader, model, optimizer)

    evaluate_model(model, dictionary, args.test_len)


if __name__ == "__main__":
    main()