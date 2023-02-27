import pandas as pd
import torch
import pickle
import math
import torch.nn.functional as F
from utils import convert_word, One_Hot_Encoding

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputSize = 2548 * 20
        self.hiddenSize1 = 1000
        self.hiddenSize2 = 100
        self.outputSize = 19

        self.w1 = torch.nn.Parameter(torch.empty(self.inputSize, self.hiddenSize1, requires_grad=True))
        self.b1 = torch.nn.Parameter(torch.zeros(self.hiddenSize1))
        self.w2 = torch.nn.Parameter(torch.empty(self.hiddenSize1, self.hiddenSize2, requires_grad=True))
        self.b2 = torch.nn.Parameter(torch.zeros(self.hiddenSize2))
        self.w3 = torch.nn.Parameter(torch.empty(self.hiddenSize2, self.outputSize, requires_grad=True))
        self.b3 = torch.nn.Parameter(torch.zeros(self.outputSize))


        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))



    def forward(self, x):
        x = torch.matmul(x, self.w1) + self.b1
        x = F.relu(x)
        x = torch.matmul(x, self.w2) + self.b2
        x = F.relu(x)
        x = torch.matmul(x, self.w3) + self.b3



        return x


def main():
    train = pd.read_csv('../data/simple_seq.train.csv', names=list(range(21))).fillna('padding')
    test = pd.read_csv('../data/simple_seq.test.csv', names=list(range(21))).fillna('padding')

    train_x = train.replace('D[0-9]*', 'padding', regex=True)
    train_x = train_x.to_numpy()

    train_y = train.replace('W[0-9]*', 'padding', regex=True)
    train_y = train_y.to_numpy()

    test_x = test.replace('D[0-9]*', 'padding', regex=True)
    test_x = test_x.to_numpy()

    vocabulary = convert_word(train_x)
    label = convert_word(train_y, True)

    id_to_word = {i: w for w, i in label.items()}

    train_x = One_Hot_Encoding(train_x, vocabulary)
    onehot_metrix = torch.eye(len(vocabulary))
    train_onehot = onehot_metrix[train_x]
    train_onehot = train_onehot.reshape((train_x.shape[0], len(vocabulary) * 20))

    train_y = One_Hot_Encoding(train_y, label, True)
    test_x = One_Hot_Encoding(test_x, vocabulary)
    onehot_metrix = torch.eye(len(vocabulary))
    test_onehot = onehot_metrix[test_x]
    test_onehot = test_onehot.reshape((test_x.shape[0], len(vocabulary) * 20))

    x = torch.FloatTensor(train_onehot).cuda()
    y = torch.LongTensor(train_y).cuda()
    test_x = torch.FloatTensor(test_onehot).cuda()

    model = CustomModel()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch_counter in range(1000):
        out = model(x)
        optimizer.zero_grad()
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        scheduler.step()
        print(loss)


    pred = model(test_x)
    idx = pred.max(axis=1)[1].to('cpu')
    idx = idx.numpy().tolist()
    final = pd.read_csv('../data/submission_example.csv')

    res = [id_to_word[w] for w in idx]
    final = final.drop(['pred'],axis=1)
    final = final.assign(pred=res)
    final.to_csv('simple_seq.answer.csv', index=False)


    with open('../data/voca.pickle', 'wb') as fw:
        pickle.dump(vocabulary, fw)


if __name__ == "__main__":
    main()