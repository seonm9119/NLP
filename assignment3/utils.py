import numpy as np
import pandas as pd
import os
import json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import torch.nn as nn
import torch
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1).cuda()
        model_prob.scatter_(1, target.unsqueeze(1), 0).cuda()
        target.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return nn.CrossEntropyLoss(output, ta)

def load_pos_data(file_name):
    with open(file_name, 'r') as f:
        embedding_dic = json.load(f)

    tokens = []
    ud_tags = []
    for key in embedding_dic.keys():
        data = embedding_dic[key]
        if 'ud_tags' in data:
            ud_tags.append(data['ud_tags'])
        tokens.append(data['tokens'])

    return tokens, ud_tags

def load_label_dic():

    ud_dict = {}
    with open('dataset/pos/pos_datasets/tgt.txt', 'r') as f:
        for key in f:
            ud_dict[key.rstrip()] = len(ud_dict)

    return ud_dict


def load_file(file_name):
    df = pd.read_csv(os.path.join('dataset/classification/classification_datasets', file_name))

    sentence = df['sentence'].to_numpy()
    label = df['label'].to_numpy() if 'label' in df.columns else None

    return sentence, label

def load_word_file(file_name):

    # 3. Use given Glove word embedding dictionary and vectors
    with open(file_name, 'r') as f:
        embedding_dic = json.load(f)

    dictionary = {}
    vectors = []
    for key in embedding_dic.keys():
        item = len(dictionary)
        dictionary[key] = item
        vectors.append(embedding_dic[key])

    vectors = np.asarray(vectors)

    return dictionary, embedding_dic, vectors

def pos_tagging(words):
    """
        Use this method when lemmatizing

        Input: list of words
        Output: {word: tag}
    """
    from nltk import pos_tag
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')

    words_only_alpha = [w for w in words if w.isalpha()]

    def format_conversion(v, pos_tags=['n', 'v', 'a', 'r', 's']):
        w, p = v
        p_lower = p[0].lower()
        p_new = 'n' if p_lower not in pos_tags else p_lower
        return w, p_new

    res_pos = pos_tag(words_only_alpha)
    word2pos = {w: p for w, p in list(map(format_conversion, res_pos))}

    for w in words:
        if w not in word2pos:
            word2pos[w] = 'n'

    return word2pos

def convert_to_tokens(sentence):


    # 1. Use word_tokenize
    tokens = []
    for sample in sentence:
        token = word_tokenize(sample)
        tokens.append(token)


    lemmatizer = WordNetLemmatizer()
    w_lemmas = []
    for sample in tokens:
        tag_dict = pos_tagging(sample)
        w_lemma = [lemmatizer.lemmatize(w,tag_dict[w]) for w in sample]
        w_lemmas.append(w_lemma)

    return w_lemmas
