import numpy as np
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x,y):

        self.x = x
        self.y = y.reshape((y.shape[0],1))


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        #x = torch.LongTensor(self.x[idx])
        y = torch.LongTensor(self.y[idx])
        return x, y

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


def convert_char(token, vocabulary, w_size):

    inputs = []
    for word in token:
        row = []
        for c in word:
            try:
                row.append(vocabulary[c])
            except KeyError:
                row.append(vocabulary['[UNK]'])
        row += [0] * (w_size - len(row))
        inputs.append(row)


    return inputs


def One_Hot_Encoding(sentence, vocabulary, w_size):

    char_train_x = []
    padding = [0] * w_size

    for token in sentence:
        row = convert_char(token, vocabulary, w_size)
        for _ in range(40 - len(row)):
            row.append(padding)
        char_train_x.append(row)

    char_train_x = np.asarray(char_train_x)
    one_hot_mat = np.eye(len(vocabulary))
    one_hot_mat = one_hot_mat[char_train_x]
    one_hot_mat = torch.FloatTensor(one_hot_mat)

    return one_hot_mat



