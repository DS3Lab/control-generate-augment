import json
import os
from os import listdir
from os.path import isfile, join
from model import SentenceVAE
from utils import to_var, idx2word, interpolate
import torch
import pandas as pd


date = "2019-Dec-09-09:24:59"
exp_descr = pd.read_csv("EXP_DESCR/"+date+".csv")
print("Pretained: ", exp_descr['pretrained'][0])
print("Bidirectional: ", exp_descr['Bidirectional'][0])
epoch = str(9)
data_dir = 'data'
# 2019-Dec-02-09:35:25, 60,300,256,0.3,0.5,16,False,0.001,10,False

embedding_size = exp_descr["Embed Len"][0]
hidden_size = exp_descr["Hidden"][0]
rnn_type = 'gru'
word_dropout = exp_descr["WordDrop"][0]
embedding_dropout = exp_descr["embedDrop"][0]
latent_size = exp_descr["Latent"][0]
num_layers = 1
batch_size = 32
bidirectional = bool(exp_descr["Bidirectional"][0])
max_sequence_length = 10
num_samples = 500
save_model_path = 'bin'
ptb = False
if ptb == True:
        vocab_dir = '/ptb.vocab.json'
else:
        vocab_dir= '/yelp_vocab.json'

with open(data_dir + vocab_dir, 'r') as file:
    vocab = json.load(file)

w2i, i2w = vocab['w2i'], vocab['i2w']


model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=max_sequence_length,
        embedding_size=embedding_size,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        word_dropout=word_dropout,
        embedding_dropout=embedding_dropout,
        latent_size=latent_size,
        num_layers=num_layers,
        bidirectional=bidirectional
        )

print(model)
# Results
# 2019-Nov-28-13:23:06/E4-5".pytorch"

load_checkpoint = "bin/"+ date +"/"+ "E"+str(epoch)+".pytorch"
#load_checkpoint = "bin/2019-Nov-28-12:03:44 /E0.pytorch"

if not os.path.exists(load_checkpoint):
    raise FileNotFoundError(load_checkpoint)

if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
else:
        device = "cpu"

model.load_state_dict(torch.load(load_checkpoint, map_location=torch.device(device)))


model.eval()
samples, z = model.inference(n=num_samples)
print('----------SAMPLES----------')
print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')



z1 = torch.randn([latent_size]).numpy()
z2 = torch.randn([latent_size]).numpy()
z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())

samples, _ = model.inference(z=z)

print('-------INTERPOLATION-------')
print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

#for root, dirs, files in os.walk(save_model_path):
#    for name in files:
#            print(name, root, dirs)
