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
from Discriminator import Discriminator, RNN_discr, LSTM_discr
from imdb import Processor
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from new_labels import multiple_binary_cross_entropy, multiple_accuracy_attributes
from finetuning import finetuning
datasets = OrderedDict()
splits = ['train', 'valid']
data_dir = "data"
for split in splits:
    datasets[split] = Processor(
        data_dir=data_dir,
        split=split,
        create_data=True,
        dataset='yelp',
        max_sequence_length=20,
        min_occ=5,
        rows=200,
        what="text"
    )
print("original vocab: ",len(datasets['train'].w2i.keys()))

model = finetuning(datasets, data_dir, embedding_size=300)
