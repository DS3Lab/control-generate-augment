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
from imdb import Processor
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import tqdm

from torch.nn import Softmax
from torch.nn import LogSoftmax

ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
splits = ['train', 'valid']
cuda2 = torch.device('cuda:5')
data_dir = 'data'
create_data = True
max_sequence_length = 50
min_occ = 5
attr_size = 2
embedding_size = 300
hidden_size = 256
rnn_type = 'gru'
word_dropout = 0.6
embedding_dropout = 0.5
latent_size = 64
num_layers = 1
batch_size = 32
bidirectional = False
save_model_path = 'bin'
lr = 0.001
coeff = 0.0025
x0 = 7500
epochs = 15
pretrained = False
datasets = OrderedDict()
data_name = 'imdb'

for split in splits:
    datasets[split] = Processor(
        data_dir=data_dir,
        split=split,
        create_data=create_data,
        dataset='yelp',
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

from VerbsClassifier import VerbsClassifier
model = VerbsClassifier(
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
    model = model.to(cuda2)

print(model)

verbs_opt = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer for the VAE


train_loader = DataLoader(dataset=datasets['train'], batch_size=batch_size, shuffle=(split == 'train'),
                                 num_workers=cpu_count(), pin_memory=torch.cuda.is_available())
valid_loader = DataLoader(dataset=datasets['valid'], batch_size=batch_size, shuffle=(split == 'train'),
                                 num_workers=cpu_count(), pin_memory=torch.cuda.is_available())

device = cuda2 if torch.cuda.is_available() else "cpu"
softmax = Softmax(1)
nll = torch.nn.NLLLoss()
tb = SummaryWriter()

for epoch in range(epochs):
    z_dataset = np.zeros((1, latent_size))
    labels = np.array([])
    for split in splits:
        epoch_accuracy = []
        if split == 'train':
            data_loader = train_loader
            model.train()
        else:
            data_loader=valid_loader
            model.eval()

        for iteration, batch in enumerate(data_loader):

            batch_size = batch['input'].size(0)
            # simply creating variables and no more only simple tensors
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            logits, z = model(batch['input'].to(device), batch['length'].to(device), batch['label'].to(device), encoder=True)

            _, target_attr = batch['label'].max(dim=1)
            disc_loss = F.cross_entropy(logits, target_attr)

            # Compute discriminator accuracy
            attr_prob = softmax(logits)
            disc_pred = torch.argmax(attr_prob, 1)
            y_true = target_attr.detach().cpu().numpy()
            y_pred = disc_pred.detach().cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred)
            epoch_accuracy.append(accuracy)
            # Train discriminator
            if split == 'train':
                verbs_opt.zero_grad()
                disc_loss.backward()
                verbs_opt.step()

            if split == 'train':
                _, target_attr = batch['label'].max(dim=1)
                target = target_attr.cpu().detach().numpy()
                labels = np.concatenate((labels, target))
                z_saved = z.cpu().detach().numpy()
                z_dataset = np.vstack((z_dataset,z_saved))

            if iteration % 100 == 0:
                print("Epoch: {} |Disc_loss: {:.4f} | Accuracy: {:.6f}".format(str(epoch) + " "+ split.upper(),disc_loss.item(), accuracy))


        print("Epoch: {} | Accuracy: {:.6f}".format(str(epoch) + " "+ split.upper(), np.mean(epoch_accuracy)))
        if split == "train":
            tb.add_histogram('grad_linear_fc1', model.fc1.weight.grad, epoch)
            tb.add_histogram('grad_linear_fc2', model.fc2.weight.grad, epoch)
            tb.add_histogram('grad_hidden', model.encoder_rnn.weight_hh_l0.grad, epoch)
            tb.add_histogram('hidden2mean_grad', model.hidden2mean.weight.grad, epoch)
        if split == 'train':
            z_dataset = pd.DataFrame(z_dataset)
            z_dataset.to_csv("Latent/"+str(epoch)+".csv", index=False)
tb.close()