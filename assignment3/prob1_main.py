import pandas as pd
import argparse
import numpy as np
import torch

from utils import load_file, load_word_file, convert_to_tokens
from Models.RNN import RNNmodel

parser = argparse.ArgumentParser(description='NLP-lab3-prob1')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', default=1e-5, type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--num_layers', default=3, type=int,
                    help='number of RNN layer')
parser.add_argument('--output_size', default=6, type=int, metavar='N',
                    help='output size')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='RNN hidden-size')
parser.add_argument('--max_len', default=20, type=int,
                    help='lenght of sequence')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x,y):

        self.x = x
        self.y = y.reshape((y.shape[0],1))


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.x[idx])
        y = torch.LongTensor(self.y[idx])
        return x, y

def pre_padding_n_truncation(sentence, vocabulary, max_len):

    # 2. Pre-padding & pre-sequence truncation
    inputs = []
    for tokens in sentence:
        row = []
        # 2-1. pre-padding
        row += [0] * (max_len - len(tokens))
        for w in tokens:
            # 2-2. pre-sequence truncation
            if len(row) < max_len:
                try:
                    row.append(vocabulary[w])
                except KeyError:
                    row.append(vocabulary['[UNK]'])
        inputs.append(row)

    inputs = np.asarray(inputs)

    return inputs

def train(args,train_loader,model, criterion, optimizer):


    for epoch in range(args.epochs):
        loss_epoch = 0
        for step, (x, y) in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                x = x.cuda()
                y = y.cuda()

                outputs = model(x)
                loss = criterion(outputs, y.flatten())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            loss_epoch += loss.item()

        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}")

def evaluate_model(model, dictionary, max_len):

    sentence, _ = load_file('test_set.csv')
    w_lemmas = convert_to_tokens(sentence)
    inputs = pre_padding_n_truncation(w_lemmas, dictionary, max_len)

    test_dataset = torch.LongTensor(inputs).cuda()
    pred = model(test_dataset)

    # Save result
    idx = pred.max(axis=1)[1].to('cpu')
    idx = idx.numpy().tolist()
    final = pd.read_csv('result/classification_class.pred.csv')

    final = final.drop(['label'], axis=1)
    final = final.assign(label=idx)
    final.to_csv('result/classification_class.pred.csv', index=False)

def main():

    args = parser.parse_args()
    sentence, label = load_file('train_set.csv')

    # Step 1. Word level embedding
    # 1. Use word_tokenize
    w_lemmas = convert_to_tokens(sentence)

    # 3. Use given Glove word embedding dictionary and vectors
    dictionary, embedding_dic, args.vectors = load_word_file('dataset/classification/classification_datasets/glove_word.json')

    # 2. Pre-padding & pre-sequence truncation
    inputs = pre_padding_n_truncation(w_lemmas, dictionary, args.max_len)

    # Step 2. Training Classification model
    train_dataset = CustomDataset(inputs, label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

    model = RNNmodel(args, bidirectional=False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Step 3. Training classification Model
    train(args,train_loader, model, criterion, optimizer)

    # Step 4. Evaluate model
    evaluate_model(model, dictionary, args.max_len)


if __name__ == "__main__":
    main()