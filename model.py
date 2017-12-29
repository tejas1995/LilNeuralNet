import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
NUM_EPOCHS = 10


def buildVocab(list_verses):

    vocab = []
    vocab.append(START_TOKEN)
    vocab.append(END_TOKEN)

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
            Y.append(END_TOKEN)
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


def train(input_var, target_var, encoder, decoder, enc_optim, dec_optim, criterion, word_to_index):

    encoder_hidden = encoder.initHidden()

    enc_optim.zero_grad()
    dec_optim.zero_grad()

    input_length = input_var.size()[0]
    target_length = target_var.size()[0]

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)

    decoder_hidden = encoder_hidden
    decoder_input = autograd.Variable(torch.LongTensor([word_to_index[START_TOKEN]])

    for di in range(target_length):

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_var[di])
        decoder_input = target_var[di]

    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss.data[0]/target_length


def trainIters(training_data, encoder, decoder, num_epochs, word_to_index, learning_rate=1e-3, print_every=10):

    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = list2Variables(training_data)
    size_data = len(training_pairs)
    criterion = nn.NLLLoss()

    sum_loss = 0
    list_losses = []

    for epoch in range(epochs):

        sum_loss = 0
        for iter in range(1, size_data+1):

            training_pair = training_pairs[iter-1]
            input_var = training_pair[0]
            output_var = training_pair[1]

            loss = train(input_var, output_var, encoder, decoder, encoder_optim, decoder_optim, criterion, word_to_index)
            sum_loss += loss

            if iter % print_every == 0:

                print 'Summed loss over last', print_every, 'iters:', sum_loss
                list_losses.append(sum_loss/print_every)
                sum_loss = 0

    # Save encoder and decoder

    showPlot(list_losses)


def showPlot(list_losses):

    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(list_losses)



if __name__=='__main__':

    verses_data = preprocLyrics('Kanye-lyrics.txt')
    vocab, word_to_index = buildVocab(verses_data)
    training_data = getTrainingData(verses_data, word_to_index)

    VOCAB_SIZE = len(vocab)
    encoder = Encoder(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM)
    decoder = Decoder(VOCAB_SIZE, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE)

    trainIters(training_data, encoder, decoder, NUM_EPOCHS, word_to_index)
