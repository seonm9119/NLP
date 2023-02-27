import numpy as np


def One_Hot_Encoding(data, vocabulary, label=False):

    inputs = []
    for sample in data:

        if not label:
            row = []
            for w in sample:
                try:
                    row.append(vocabulary[w])
                except KeyError:
                    row.append(vocabulary['unknown'])
            row.pop()

        else:
            for w in sample:
                if w != 'padding':
                    row = vocabulary[w]
        inputs.append(row)
    inputs = np.array(inputs)

    return inputs


def convert_word(data, label=False):
    words = data.reshape(-1)
    words = list(dict.fromkeys(words))
    words.remove('padding')

    if not label:
        dictionary = {'padding': 0, 'unknown': 1}
    else:
        dictionary = {}

    for w in words:
        dictionary[w] = len(dictionary)

    if label:
        dictionary['padding'] = len(dictionary)

    return dictionary