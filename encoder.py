import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim, embed_dim, n_layers=1):

        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(input_dim, vocab_size)
        self.gru = nn.GRU(embed_dim, hidden_dim)


    def initHidden(self):

        result = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        return result


    def forward(self, input, hidden):

        embedded = self.embed(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)

        return output, hidden
