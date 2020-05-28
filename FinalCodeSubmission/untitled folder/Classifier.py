import torch.nn as nn
import torch
from Embedder import Embedder
import torch.nn.utils.rnn as rnn_utils


'''
Classifier for Data Augmentation

'''
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, embedding_layer, sos_idx, eos_idx, unk_idx,device):
        '''

        :param vocab_size: (int)
        :param embedding_dim: (int)
        :param hidden_dim: (int)
        :param output_dim: Binary label for sentiment classification
        :param n_layers: (int)
        :param bidirectional: (bool)
        :param dropout: (float)
        :param pad_idx: (int)
        :param embedding_layer: [Embedding_Layer]
        :param sos_idx:
        :param eos_idx:
        :param unk_idx:
        :param device:
        '''
        super().__init__()
        self.device = device
        if embedding_layer != None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx).to(device)

        self.rnn = nn.GRU(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                          batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 , 1)
        #self.fc1 = nn.Linear(hidden_dim, 16)
        #self.fc2 = nn.Linear(16,1)

        self.dropout = nn.Dropout(dropout)

    def embedded_dropout(self, embed, words, dropout=0.3):

        '''

        Embedding word dropout for multi-layer bidirectional Sequential Model

        :param embed:
        :param words:
        :param dropout:
        :return: the embeddings matrix with embedding dropout applied on it
        '''

        mask = embed.weight.data.new_empty((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
        return torch.nn.functional.embedding(words, masked_embed_weight)


    def forward(self, input_sequence, length, label,verbose=False):
        # PARAMETERS
        batch_size = input_sequence.size(0)
        # print("Lenght:" ,length)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        if verbose:
            print(input_sequence)
        label = label[sorted_idx]

        # EMBEDDING LAYER
        input_embedding = self.embedding(input_sequence)
        input_embedding = self.embedded_dropout(self.embedding, input_sequence)
        # ENCODER

        padded = rnn_utils.pad_sequence(input_embedding, batch_first=True)
        sorted_batch_lengths = [len(x) for x in padded]

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_batch_lengths, batch_first=True)

        _, hidden = self.rnn(packed_input)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        hidden = hidden.squeeze(0)
        #print(hidden)
        hidden = self.dropout(hidden)

        logits = self.fc(hidden)
        #fc1_out = torch.nn.functional.relu(self.fc1(self.dropout(fc_out)))
        #logits = self.fc2(self.dropout(fc1_out))
        return logits, label

