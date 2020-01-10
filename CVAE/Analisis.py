from torch.utils.data import DataLoader
import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from collections import OrderedDict, defaultdict
from math import exp, log
from utils import decoding_ouput
from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE
from Embedder import Embedder
import pandas as pd
from yelp import YELP
from Discriminator import Discriminator
import torch.functional as F

ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
splits = ['train', 'valid']
cuda2 = torch.device('cuda:0')
data_dir = 'data'
create_data = True
max_sequence_length = 50
min_occ = 0
attr_size = 3
embedding_size = 300
hidden_size = 256
h1_size = 100
rnn_type = 'gru'
word_dropout = 0
embedding_dropout = 0.5
latent_size = 16
num_layers = 1
batch_size = 32
bidirectional = False
save_model_path = 'bin'
lr = 0.001
coeff = 0.0025
x0 = 1000
epochs = 10
pretrained = False
datasets = OrderedDict()

exp = [
    [ts, max_sequence_length, embedding_size, hidden_size, word_dropout, embedding_dropout, latent_size, bidirectional,
     lr, epochs, pretrained]]
exp_description = pd.DataFrame(exp,
                               columns=['Time', 'Seq Len', 'Embed Len', 'Hidden', 'WordDrop', 'embedDrop', 'Latent',
                                        'Bidirectional', 'lr', 'epochs', 'pretrained'])
exp_description.to_csv("EXP_DESCR/" + str(ts) + ".csv", index=False)

for split in splits:
    datasets[split] = YELP(
        data_dir=data_dir,
        split=split,
        create_data=create_data,
        max_sequence_length=max_sequence_length,
        min_occ=min_occ
    )



if pretrained:
    embedding_dir = "glove/glove.6B." + str(embedding_size) + "d.txt"
    glove = Embedder(filepath=embedding_dir)
    embedding_mtx = glove.build_weight_matrix(datasets['train'].w2i.keys(), embedding_size)
    emb_layer, num_embeddings, embedding_dim = glove.create_embedding_layer(weight_matrix=embedding_mtx)

else:
    emb_layer = None
model = SentenceVAE(
    vocab_size=datasets['train'].vocab_size,
    sos_idx=datasets['train'].sos_idx,
    eos_idx=datasets['train'].eos_idx,
    pad_idx=datasets['train'].pad_idx,
    unk_idx=datasets['train'].unk_idx,
    max_sequence_length=max_sequence_length,
    embedding_size=embedding_size,
    rnn_type=rnn_type,
    hidden_size=hidden_size,
    word_dropout=word_dropout,
    embedding_dropout=embedding_dropout,
    latent_size=latent_size,
    attribute_size=attr_size,
    num_layers=num_layers,
    cuda=cuda2,
    bidirectional=bidirectional,
    pretrained=pretrained,
    embedding_layer=emb_layer
)

if torch.cuda.is_available():
    model = model.cuda(cuda2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#disc = Discriminator(input_size=latent_size, h1_size=h1_size, output_size=latent_size).to(device)


print(model)
#print("Current: ", torch.cuda.current_device())
save_model_path = os.path.join(save_model_path, ts)
os.makedirs(save_model_path)


def kl_anneal_function(anneal_function, x, k, steps, eps=1e-5):
    if anneal_function == 'logistic':
        k = -(log(-1 + 1 / (1 - eps))) / (0.5 * steps)
        return float((1 / (1 + np.exp(-k * (x - steps)))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)


def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):
    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight


optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer for the VAE
#disc_optim = torch.optim.Adam(disc.parameters(), lr=1e-2, betas=(0.5,0.999)) #optimizer for disc

tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
step = 0
for epoch in range(epochs):

    for split in splits:
        data_loader = DataLoader(
            dataset=datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )


        tracker = defaultdict(tensor)

        # Enable/Disable Dropout
        if split == 'train':
            model.train()

        else:
            model.eval()
        elbo = []
        recon_loss = []
        kl_loss = []
        for iteration, batch in enumerate(data_loader):

            batch_size = batch['input'].size(0)

            # simply creating variables and no more only simple tensors
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            '''
            # training of the discriminator

                #1: encoding the data

            _, _, _, z = model(batch['input'], batch['length']) # da sistemare in modo da ricevere solo z - solo encoding
            
            
            

            #2: prediction using the discriminator

            attr_probs = disc(z)
            disc_loss = F.cross_entropy(attr_probs, labels, reduction='mean') #add the labels and check the reduction
            #sum_disc_loss += disc_loss.item()
            if split == 'train':
                disc_optim.zero_grad()
                disc_loss.backward()
                disc_optim.step()

            # discriminator accuracy 

            disc_pred = torch.argmax(attr_probs, 1)
            disc_acc = torch.sum(disc_pred == digits)
            #sum_disc_acc += disc_acc.item()
            
            '''



            #3: training step of the variational autoencoder
            logp, mean, logv, z = model(batch['input'], batch['length'], batch['label'])
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                                                   batch['length'], mean, logv, 'logistic', step, coeff,
                                                   x0)

            loss = (NLL_loss + KL_weight * KL_loss) / batch_size
            elbo.append(loss.item())
            recon_loss.append(NLL_loss.item() / batch_size)
            kl_loss.append(KL_loss.item() / batch_size)

            # fader loss to be written which is the one that we should optimize

            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
            # tracker['ELBO'] = torch.cat((tracker['ELBO'], loss)) for now we don't use this but later we must
            # sentences = idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
            #                     pad_idx=datasets['train'].pad_idx)
            # sentences_decode = decoding_ouput(logp, datasets['train'].get_i2w())
            if iteration % 100 == 0 or iteration + 1 == len(data_loader):
                print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                      % (split.upper(), iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
                         KL_loss.item() / batch_size, KL_weight))
            if iteration % 100 == 0 and split == "valid":
                sentences = idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
                                     pad_idx=datasets['train'].pad_idx)
                sentences_decode = decoding_ouput(logp, datasets['train'].get_i2w())
                sel = np.random.randint(0, len(sentences))
                print("PHASE: ", split.upper())
                print("Input: ", sentences[sel])
                print("Decoding: ", sentences_decode[sel])

        print("Epoch: ", epoch, " ELBO: ", np.mean(elbo), " NLL-Loss: ", np.mean(recon_loss), " KL-Loss:",
              np.mean(kl_loss))

        if split == 'train':
            checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % (epoch))
            torch.save(model.state_dict(), checkpoint_path)
            print("Model saved at %s" % checkpoint_path)

# target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
