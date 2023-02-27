import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch

from utils import pos_tagging, One_Hot_Encoding, CustomDataset
from models.Model import CustomModel

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)

def convert_word(sentence):

    # STEP 1. Tokenize the input sentence
    tokens = []
    for sample in sentence:
        tokens.append(word_tokenize(sample))


    # STEP 2. Lemmatize the tokenized words
    lemmatizer = WordNetLemmatizer()
    w_lemmas = []
    for sample in tokens:
        tag_dict = pos_tagging(sample)
        w_lemma = [lemmatizer.lemmatize(w,tag_dict[w]) for w in sample]
        w_lemmas.append(w_lemma)

    return w_lemmas


def train_model(train_x, label):

    # STEP 4. Train your sentence classification model
    batch_size = 2250
    epochs = 200

    train_dataset = CustomDataset(train_x, label)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size)

    # STEP 3. Word Representation using Character Embedding
    model = CustomModel()
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.cuda()

    for epoch_counter in range(epochs):
        loss_ = 0
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            optimizer.zero_grad()
            loss = criterion(out, y.flatten())
            loss.backward()
            optimizer.step()

            loss_ +=loss
        print("LOSS",loss_/2)

    return model


def evaluate_model(model, char_vocabulary):

    # STEP 5. Evaluate the performance of your trained model on test set
    test = pd.read_csv('data/sent_class.test.csv')
    sentence = test['sentence'].to_numpy()

    test_words = convert_word(sentence)
    test_x = One_Hot_Encoding(test_words, char_vocabulary, 30).cuda()

    pred = model(test_x)

    # Save result
    idx = pred.max(axis=1)[1].to('cpu')
    idx = idx.numpy().tolist()
    final = pd.read_csv('data/sent_class.pred.csv')

    final = final.drop(['pred'], axis=1)
    final = final.assign(pred=idx)
    final.to_csv('result/final_sent_class.pred.csv', index=False)

    print(idx)


def main():

    train = pd.read_csv('data/sent_class.train.csv')
    label = train['label'].to_numpy()
    sentence = train['sentence'].to_numpy()

    # STEP 1,2. Tokenize the input sentence & Lemmatize the tokenized words
    train_words = convert_word(sentence)
    words = sum(train_words, [])
    words = list(dict.fromkeys(words))

    chars = set([w_i for w in words for w_i in w])
    chars = sorted(list(chars))
    char_vocabulary = {"[PAD]": 0, "[UNK]": 1}
    for c in chars:
        char_vocabulary[c] = len(char_vocabulary)

    train_x = One_Hot_Encoding(train_words, char_vocabulary, 30)

    # STEP 3,4. Word Representation using Character Embedding & Train your sentence classification model
    trained_model = train_model(train_x, label)


    # STEP 5. Evaluate the performance of your trained model on test set
    evaluate_model(trained_model, char_vocabulary)


if __name__ == "__main__":
    main()