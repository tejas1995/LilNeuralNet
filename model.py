import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from preprocess_lyrics import preprocLyrics
from encoder import Encoder
from decoder import Decoder


START_TOKEN = 'START_TOKEN'
END_TOKEN = 'END_TOKEN'

# Define some constants
EMBEDDING_DIM = 32
HIDDEN_DIM = 100
MAX_LENGTH = 50


def buildVocab(list_verses):

    vocab = []
    
    for verse in list_verses:
        for line in verse:
            for word in line:
                if word not in vocab:
                    vocab.append(word)

    word_to_index = {}
    for word in vocab:
        # print word
        word_to_index[word] = vocab.index(word)

    print 'Vocabulary size:', len(vocab)

    return vocab, word_to_index


def getTrainingData(list_verses, word_to_index):

    training_data = []

    for verse in list_verses:

        num_lines = len(verse)
        for i in range(num_lines-1):

            X = [word_to_index[w] for w in verse[i]]
            Y = [word_to_index[w] for w in verse[i+1]]
            training_data.append((X, Y))

    random.shuffle(training_data)
    print 'Number of training examples:', len(training_data)

    # print 'Example input sequence:', [vocab[i] for i in training_data[0][0]]
    # print 'Example output sequence:', [vocab[i] for i in training_data[0][1]]

    return training_data


def seq2Tensor(seq):
    tensor = torch.LongTensor(seq).view(-1, 1)
    return autograd.Variable(tensor)


def list2Variables(training_data):

    training_pairs = []
    for seq_in, seq_out in training_data:
        var_in = seq2Tensor(seq_in)
        var_out = seq2Tensor(seq_out)
        training_pairs.append((var_in, var_out))

    return training_pairs


if __name__=='__main__':

    verses_data = preprocLyrics('Kanye-lyrics.txt')
    vocab, word_to_index = buildVocab(verses_data)
    training_data = getTrainingData(verses_data, word_to_index)
