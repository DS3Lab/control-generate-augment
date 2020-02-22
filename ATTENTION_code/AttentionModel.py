import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_var
from torch.distributions.bernoulli import Bernoulli
class AttentionModule(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.attention2mean = nn.Linear(hidden_size, latent_size)
        self.attention2logv = nn.Linear(hidden_size, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)

    def reparametization(self, context, batch_size):
        # REPARAMETERIZATION
        mean = self.attention2mean(context)
        logv = self.attention2logv(context)
        std = torch.exp(0.5 * logv)
        a = to_var(torch.randn([batch_size, self.latent_size]))
        a = a * std + mean
        a = self.latent2hidden(a)
        return a, mean, logv


    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        #normalized attn_weight
        #decoder_hidden = decoder_hidden.unsqueeze(dim=2)
        #print("encoder_outputs: ",encoder_outputs.size(), "decoder_hidden: ", decoder_hidden.transpose(0,1).size())
        attn_weight = torch.bmm(encoder_outputs, decoder_hidden.transpose(0,1).transpose(1,2))
        attn_weight = F.softmax(attn_weight,dim=1)
        #print("attn_weigh: ", attn_weight.size())
        #context_vector
        context = torch.bmm(encoder_outputs.transpose(1,2), attn_weight).squeeze(2) # questo è un primo modo di ottenere il tutto
        #print("context size: ", context.size())
        #sampling the context_vector
        a, att_mean, att_logv = self.reparametization(context, batch_size)
        # linear layer secondo me è carino

        # KL-loss on attention
        #print("att_mean: ",att_mean.size())
        KL_loss_attention = -0.5 * torch.sum(1 + att_logv - att_mean.pow(2) - att_logv.exp())

        return a, KL_loss_attention





class AttnDecoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, latent_size, word_dropout_rate, unk_idx,device):

        super().__init__()
        self.hidden_size = hidden_size
        self.attn = AttentionModule(hidden_size, latent_size)
        self.decoder = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.bernoulli = Bernoulli(torch.tensor([word_dropout_rate]))
        self.unk_idx = unk_idx
        self.device=device
    def forward(self, word, hidden, context, encoder_outputs, seq_len):
        context = context.unsqueeze(1)
        input = torch.cat((word,context),dim=2)
        if self.bernoulli.sample().item() == 1:
            input = torch.Tensor(input.size(0),input.size(1), input.size(2))
            input = input.fill_(self.unk_idx).to(self.device)
        next_output, next_hidden = self.decoder(input, hidden)
        context, KL_loss_attention = self.attn(next_hidden, encoder_outputs)



        return next_output, next_hidden, context, KL_loss_attention/seq_len

