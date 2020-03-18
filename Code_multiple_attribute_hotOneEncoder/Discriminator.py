import torch.nn as nn
import torch.nn.functional as F
import torch
'''
class Discriminator(nn.Module):

    def __init__(self, input_size, h1_size, h2_size, output_size, dropout):
        super().__init__()

        ############### hidden_size #############

        self.h1_size = h1_size
        self.output_size = output_size
        self.input_size = input_size

        #########################################

        self.fc1 = nn.Linear(input_size,h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
'''




class Discriminator(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, output_size=2,dropout=0.3):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.BatchNorm1d(h1_size),
            nn.ReLU(),
            nn.Linear(h1_size, h2_size),
            nn.BatchNorm1d(h2_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h2_size, output_size),
        )

    def forward(self, z_s):
        preds = self.linears(z_s)
        return preds


class LSTM_discr(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


import torch.nn as nn


class RNN_discr(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim):
        super().__init__()

        #self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.GRU(input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        # embedded = [sent len, batch size, emb dim]
       # print("text device: ", text.device)
        print("inout: ", text.size())
        output, hidden = self.rnn(text)
        print("hidden: ",hidden.size())
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

       # assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class NewRNN_discr(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        #self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(input_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                         )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]
        # embedded = [sent len, batch size, emb dim]
       # print("text device: ", text.device)
        output, hidden = self.rnn(text)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

       # assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden)