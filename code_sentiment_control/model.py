import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
import time
import json
from Embedder import Embedder
import numpy as np



class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                attribute_size, sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, cuda, word_dropout_type,back, num_layers=1, bidirectional=False, pretrained=False, embedding_layer=None):

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
        self.word_dropout_type=word_dropout_type

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

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size + attribute_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)
        self.back = back


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

        z = to_var(torch.randn([batch_size, self.latent_size]), self.cuda)
        z = z * std + mean

        return z, mean, logv

    def word_dropout(self, step, maximum=0.7, minimum=0.3, warmup=2000, period=500):

        if step < warmup:
            return maximum
        y = np.abs(np.cos(((2 * np.pi) / period) * step))
        if y > maximum:
            y = maximum
        if y < minimum:
            y = minimum
        return y

    def decoder(self, input_sequence, z , input_embedding, sorted_lengths, batch_size, word_dropout_rate):

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob = prob.to(self.cuda)
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        return outputs


    def forward(self, input_sequence, length, label, step,encoder=False):
        # PARAMETERS
        batch_size = input_sequence.size(0)

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        label = label[sorted_idx]


        #EMBEDDING LAYER
        input_embedding = self.embedding(input_sequence)
        # ENCODER

        z, mean, logv = self.encoder(input_embedding, sorted_lengths, batch_size)

        if encoder:
            return z, label, input_sequence
        # DECODER
        z_a = torch.cat((z, label), dim=1)


        if self.word_dropout_type == 'static':
            word_dropout_rate = self.word_dropout_rate
        if self.word_dropout_type == 'cyclical':
            word_dropout_rate = self.word_dropout(step)


        outputs = self.decoder(input_sequence, z_a, input_embedding, sorted_lengths, batch_size, word_dropout_rate)
        # OUTPUT FORMAT
        padded_outputs = self.formatting_output(outputs, sorted_idx)
        b,s,_ = padded_outputs.size()
        # PROJECTION TO VOCAB
        out = self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2)))
        back_prob = nn.functional.softmax(out, dim=1)
        logp = nn.functional.log_softmax(out, dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)


        if self.back:
            back_prob = back_prob.view(b, s, self.embedding.num_embeddings)
            back_input = torch.argmax(back_prob, dim=2)
            back_input = back_input[sorted_idx]
            back_input_embedding = self.embedding(back_input)
            z_back, mean_back, logv_back = self.encoder(back_input_embedding, sorted_lengths, batch_size)
            loss = torch.abs(z - z_back)
            l1_loss = loss.sum()/batch_size
            return logp, mean, logv, z, z_a,l1_loss

        return logp, mean, logv, z, z_a, None

    def formatting_output(self, outputs, sorted_idx):
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)

        padded_outputs = padded_outputs[reversed_idx]
        return padded_outputs


    def inference(self,sentiment,n = 2, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([1, self.latent_size]), "cpu")
            z_f = z
            for i in range(1):
                z_f = torch.cat((z_f, z), dim=0)

            if sentiment == "Positive":

                b = np.array([[0, 1], [0, 1]]).astype(np.float32)
            if sentiment == "Negative":
                b = np.array([[1, 0], [1, 0]]).astype(np.float32)
            c = torch.from_numpy(b)
            z = torch.cat((z_f, c), dim=1)

        else:
            batch_size = z.size(0)


        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        if self.bidirectional:
            hidden = hidden.squeeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long(), "cpu")
            input_sequence = input_sequence.unsqueeze(1)
            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)
            if t == 0:
                l = self.outputs2vocab(output)

            logits = self.outputs2vocab(output)
            l = torch.cat((l,logits), dim=1)
            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z, l

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':

            #dist = torch.nn.functional.softmax(dist,1)
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
