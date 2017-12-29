import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import random
import time

from preprocess_lyrics import *
from encoder import Encoder
from decoder import Decoder

# Starting and ending tokens
START_TOKEN = 'START_TOKEN'
END_TOKEN = 'END_TOKEN'
UNK_TOKEN = 'UNK'

# Files for saving the model
enc_pkl_file = 'kanye-encoder.pkl'
dec_pkl_file = 'kanye-decoder.pkl'

# Define some constants
EMBEDDING_DIM = 32
HIDDEN_DIM = 100
MAX_OUTPUT_LENGTH = 15
NUM_EPOCHS = 20


def seq2Tensor(seq):
    tensor = torch.LongTensor(seq).view(-1, 1)
    return autograd.Variable(tensor)


def predictNextLine(input_var, encoder, decoder, word_to_index):

    encoder_hidden = encoder.initHidden()

    input_length = input_var.size()[0]

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_var[ei], encoder_hidden)

    decoder_hidden = encoder_hidden
    decoder_input = autograd.Variable(torch.LongTensor([word_to_index[START_TOKEN]]))


    next_line = []
    for di in range(MAX_OUTPUT_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(2)
        ni = topi[0][0]

        if ni is word_to_index[END_TOKEN]:
            break

        if ni is word_to_index[START_TOKEN] or ni is word_to_index[UNK_TOKEN]:
            ni = topi[0][1]
        decoder_input = autograd.Variable(torch.LongTensor([ni]))

        next_line.append(ni)

    return next_line


if __name__=='__main__':

    verses_data = preprocLyrics('Kanye-lyrics.txt')
    vocab, word_to_index = buildVocab(verses_data)
    VOCAB_SIZE = len(vocab)

    print 'Starting up...'
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM)
    decoder = Decoder(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE)
    encoder.load_state_dict(torch.load(enc_pkl_file))
    decoder.load_state_dict(torch.load(dec_pkl_file))

    print 'Enter a rap line:'
    first_line = raw_input()

    first_line = first_line.strip()
    first_line = first_line.split(' ')
    first_line = [processWord(w) for w in first_line]

    input_seq = []
    for word in first_line:
        if word in vocab:
            input_seq.append(word_to_index[word])
        else:
            input_seq.append(word_to_index[UNK_TOKEN])

    # print input_seq
    # print [vocab[ind] for ind in input_seq]

    verse_length = random.choice([12, 16, 20, 24])

    for line in range(verse_length-1):

        input_var = seq2Tensor(input_seq)

        output_seq = predictNextLine(input_var, encoder, decoder, word_to_index)
        output_line = ' '.join([vocab[ind] for ind in output_seq])
        print output_line

        input_seq = output_seq
