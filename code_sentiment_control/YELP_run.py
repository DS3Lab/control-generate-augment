import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils import decoding_ouput
from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE
from Embedder import Embedder
import pandas as pd
from yelp import YELP


ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
splits = ['train', 'valid']

data_dir = 'data'
create_data = True
max_sequence_length = 60
min_occ = 1

embedding_size = 300
hidden_size = 256
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
x0 = 2500
epochs = 10
pretrained = False
datasets = OrderedDict()

exp = [[ts, max_sequence_length, embedding_size, hidden_size, word_dropout, embedding_dropout, latent_size, bidirectional, lr, epochs, pretrained]]
exp_description = pd.DataFrame(exp, columns=['Time', 'Seq Len', 'Embed Len', 'Hidden', 'WordDrop', 'embedDrop','Latent', 'Bidirectional', 'lr', 'epochs', 'pretrained' ])
exp_description.to_csv("EXP_DESCR/" + str(ts)+".csv", index= False)

for split in splits:
    datasets[split] = YELP(
        data_dir=data_dir,
        split=split,
        create_data=create_data,
        max_sequence_length=max_sequence_length,
        min_occ=min_occ
    )