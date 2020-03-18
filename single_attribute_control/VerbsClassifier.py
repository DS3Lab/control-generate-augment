import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
import time
import json
from Embedder import Embedder
import numpy as np
import torch.nn.functional as F



class VerbsClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                attribute_size, sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, cuda, num_layers=1, bidirectional=False, pretrained=False, embedding_layer=None):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.cuda = cuda
        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if pretrained == False:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embedding = embedding_layer
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.attribute_size = attribute_size
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()
        self.enc_bidirectional = True
        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        print(bidirectional)
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size + attribute_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

        # fully connected params
        h1_size = 32
        self.fc1 = nn.Linear(latent_size, h1_size)
        self.fc2 = nn.Linear(h1_size, 2)



    def encoder(self, input_embedding, sorted_lengths, batch_size):

        # ENCODER
        padded = rnn_utils.pad_sequence(input_embedding, batch_first=True)
        sorted_batch_lengths = [len(x) for x in padded]

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_batch_lengths, batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)



        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()



        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        return z, mean, logv


    def forward(self, input_sequence, length, label, encoder=False):
        # PARAMETERS
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]


        #EMBEDDING LAYER
        input_embedding = self.embedding(input_sequence)
        # ENCODER

        z, mean, logv = self.encoder(input_embedding, sorted_lengths, batch_size)

        # classification

        x = F.relu(self.fc1(z))
        logits = self.fc2(x)


        return logits,z


    def formatting_output(self, outputs, sorted_idx):
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        return padded_outputs

