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
from new_labels import multiple_binary_cross_entropy
import tqdm

#################### SETTINGS ########################
ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
splits = ['train', 'valid']
cuda2 = torch.device('cuda:5')
device = cuda2 if torch.cuda.is_available() else "cpu"
#######################################################
################### DATA ##############################
data_dir = 'data'
create_data = True
max_sequence_length = 20
rows = 1000
#######################################################
##################### VAE ############################
min_occ = 5
attr_size = 4
embedding_size = 300
hidden_size = 256
rnn_type = 'gru'
word_dropout = 0.6
embedding_dropout = 0.5
latent_size = 32
num_layers = 1
batch_size = 3
bidirectional = False
save_model_path = 'bin'
lr = 0.001
coeff = 0.0025
x0 = 2500
epochs = 15
pretrained = False
datasets = OrderedDict()
######################################################
################ discriminator params ################
discr_type = 'fc'
input_size = latent_size + attr_size
h1_size = 50
h2_size = 20
dropout = 0
numb_classes = 2
disc_weight = 0
stop = 10000
maximum = 30
disc_step = 0
######################################################
################### TensorBoard ######################

z_dataset = np.zeros((1, latent_size))

############################################

data_name = 'imdb'

exp = [
    [ts, max_sequence_length, embedding_size, hidden_size, word_dropout, embedding_dropout, latent_size, bidirectional,
     lr, epochs, pretrained]]
exp_description = pd.DataFrame(exp,
                               columns=['Time', 'Seq Len', 'Embed Len', 'Hidden', 'WordDrop', 'embedDrop', 'Latent',
                                        'Bidirectional', 'lr', 'epochs', 'pretrained'])

if not os.path.exists("EXP_DESCR"):
    os.makedirs("EXP_DESCR")

exp_description.to_csv("EXP_DESCR/" + str(ts) + ".csv", index=False)

for split in splits:
    datasets[split] = Processor(
        data_dir=data_dir,
        split=split,
        create_data=data_name,
        dataset='yelp',
        max_sequence_length=max_sequence_length,
        min_occ=min_occ,
        rows=rows
    )

print("END")

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
    model = model.to(cuda2)

print(model)

reduction = "sum"
# print("Current: ", torch.cuda.current_device())
save_model_path = os.path.join(save_model_path, ts)
os.makedirs(save_model_path)


def create_storage():
    fader_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
    vae_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
    disc_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
    disc_accuracy_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
    kl_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}

    return fader_loss_summary, vae_loss_summary, disc_loss_summary, disc_accuracy_summary, kl_loss_summary


def restart(summary, split):
    summary[split] = []
    return summary


def disc_weight_function(type, threshold, x_step, x, epoch, eps=1e-5):
    if type == 'linear':
        y = min(threshold, (threshold / x_step) * x)
        return y
    if type == 'hard':
        return min(threshold, 3 * epoch)


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


def discriminator(discr_type):
    if discr_type == "fc":
        return Discriminator(input_size=latent_size, h1_size=h1_size, h2_size=h2_size, output_size=attr_size).to(device)
    if discr_type == "rnn":
        return RNN_discr(input_dim=latent_size, hidden_dim=latent_size // 2, output_dim=attr_size)


# disc = Discriminator(input_size=latent_size, h1_size=h1_size, h2_size=h2_size, output_size=attr_size).to(device)
# disc_rnn = RNN_discr(input_dim=latent_size, hidden_dim=latent_size//2, output_dim=attr_size)

disc = discriminator(discr_type=discr_type)

print(disc)
softmax = torch.nn.Softmax(1)
fader_opt = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer for the VAE
disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-2, betas=(0.5, 0.999))  # optimizer for disc

step = 0
fader_loss_summary, vae_loss_summary, disc_loss_summary, disc_accuracy_summary, kl_loss_summary = create_storage()
'''
train_loader = DataLoader(dataset=datasets['train'], batch_size=batch_size, shuffle=(split == 'train'),
                                 num_workers=cpu_count(), pin_memory=torch.cuda.is_available())
valid_loader = DataLoader(dataset=datasets['valid'], batch_size=batch_size, shuffle=(split == 'train'),
                                 num_workers=cpu_count(), pin_memory=torch.cuda.is_available())
'''
tb = SummaryWriter()
for epoch in range(epochs):
    z_dataset = np.zeros((1, latent_size + attr_size))
    labels = np.array([])
    for split in splits:

        data_loader = DataLoader(dataset=datasets[split], batch_size=batch_size, shuffle=(split == 'train'),
                                 num_workers=cpu_count(), pin_memory=torch.cuda.is_available())

        if split == 'train':
            # data_loader = train_loader
            model.train()
        else:
            # data_loader=valid_loader
            model.eval()

        for iteration, batch in enumerate(data_loader):

            batch_size = batch['input'].size(0)
            # simply creating variables and no more only simple tensors
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)
            # loading on GPUs

            #########################################################################################################################################################
            #######################################################    DISCRIMINATOR TRAINING   #####################################################################
            #########################################################################################################################################################

            z, label = model(batch['input'].to(device), batch['length'].to(device), batch['label'].to(device),
                             encoder=True)
            if discr_type == 'fc':
                attr_probs = disc(z)
            if discr_type == 'rnn':
                attr_probs = disc(torch.unsqueeze(z, dim=0))
            print(label)
            _, target_attr = label.max(dim=1)
            #            print("attr_probs: {} target_att: {}".format(attr_probs.size(), target_attr.size()))
            #disc_loss = multiple_binary_cross_entropy(attr_probs, label)
            disc_loss = F.cross_entropy(attr_probs, target_attr) #attr_probs are logits

            # Train discriminator
            if split == 'train':
                disc_opt.zero_grad()
                disc_loss.backward()
                disc_opt.step()

            # Compute discriminator accuracy
            attr_probs = softmax(attr_probs)
            disc_pred = torch.argmax(attr_probs, 1)
            y_true = target_attr.detach().cpu().numpy()
            y_pred = disc_pred.detach().cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred)
            #########################################################################################################################################################
            #######################################################    Varitional-AE TRAINING   #####################################################################
            #########################################################################################################################################################
            # 5: forward pass vae and discriminator
            logp, mean, logv, z, z_a = model(batch['input'].to(device), batch['length'].to(device),
                                             batch['label'].to(device))
            if discr_type == 'fc':
                attr_probs = disc(z)
            if discr_type == "rnn":
                attr_probs = disc(torch.unsqueeze(z,dim=0))
            # 6: losses of vae and discriminator
            NLL_loss, KL_loss, KL_weight = loss_fn(logp.to(device), batch['target'].to(device),
                                                   batch['length'].to(device), mean, logv, 'logistic', step, coeff,
                                                   x0)

            if split == 'train':
                _, target_attr = batch['label'].max(dim=1)
                target = target_attr.cpu().detach().numpy()
                labels = np.concatenate((labels, target))
                z_saved = z_a.cpu().detach().numpy()
                z_dataset = np.vstack((z_dataset, z_saved))

            vae_loss = (NLL_loss + KL_weight * KL_loss) / batch_size

            fader_discr_loss = (F.cross_entropy(attr_probs, target_attr)) / batch_size
            if split == 'train': disc_weight = disc_weight_function('linear', 30, 5000, step, epoch)
            fader_loss = vae_loss - (disc_weight * fader_discr_loss)

            if split == 'train':
                fader_opt.zero_grad()
                fader_loss.backward()
                fader_opt.step()
                step += 1
            if iteration % 100 == 0 and split == 'train':
                print(
                    '\nPhase: {}| Iteration {}/{}  | Disc Weight: {:.4f} | Fader Loss: {:.4f} | VAE Loss: {:.4f} |KL Weight: {:.4f} |Disc Loss: {:.4} | KL Loss: , {:.4f}| Disc Accuracy:  {:.4f}'
                    .format(split.upper(), iteration, len(data_loader), disc_weight, fader_loss, vae_loss, KL_weight,
                            disc_loss, KL_loss / batch_size, accuracy), flush=True)
                print(disc_pred.detach().cpu().numpy()[:8])

            if iteration % 100 == 0 and split == 'valid':
                print(
                    '\nPhase: {}| Iteration {}/{}  |Disc Weight: {:.4f} | Fader Loss: {:.4f} | VAE Loss: {:.4f} |KL_Weight; {:.4f} |Disc Loss: {:.4} | KL Loss: {:.4f} | Disc Accuracy:  {:.4f}'
                    .format(split.upper(), iteration, len(data_loader), disc_weight, fader_loss, vae_loss, KL_weight,
                            disc_loss, KL_loss / batch_size, accuracy), flush=True)

            if iteration % 100 == 0 and split == "valid":
                sentences = idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
                                     pad_idx=datasets['train'].pad_idx)
                sentences_decode = decoding_ouput(logp, datasets['train'].get_i2w())
                sel = np.random.randint(0, len(sentences))
                print("PHASE: ", split.upper())
                print("Input: ", sentences[sel])
                print("Decoding: ", sentences_decode[sel])

            fader_loss_summary[split].append(fader_loss.item())
            disc_loss_summary[split].append(disc_loss.item())
            disc_accuracy_summary[split].append(accuracy)

        fader_loss_summary[split + "_epoch"].append(np.mean(fader_loss_summary[split]))
        disc_loss_summary[split + "_epoch"].append(np.mean(disc_loss_summary[split]))
        disc_accuracy_summary[split + "_epoch"].append(np.mean(disc_accuracy_summary[split]))

        fader_loss_summary, disc_loss_summary, disc_accuracy_summary = restart(fader_loss_summary, split), \
                                                                       restart(disc_loss_summary, split), \
                                                                       restart(disc_accuracy_summary, split)

        if split == 'train':
            checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % (epoch))
            torch.save(model.state_dict(), checkpoint_path)
            z_dataset = pd.DataFrame(z_dataset)
            z_dataset.to_csv("Latent/" + str(epoch) + ".csv", index=False)
            print("Model saved at %s" % checkpoint_path)
        '''
        if split == "train":
            tb.add_histogram('grad_linear_fc1', disc.linears[0].weight.grad, epoch)
            tb.add_histogram('grad_linear_fc2', disc.linears[3].weight.grad, epoch)
            tb.add_histogram('grad_linear_fc2', disc.linears[7].weight.grad, epoch)
            tb.add_histogram('grad_hidden', model.encoder_rnn.weight_hh_l0, epoch)
            tb.add_histogram('grad_hidden', model.encoder_rnn.weight_ih_l0, epoch)
            tb.add_histogram('hidden2mean_grad', model.hidden2mean.weight.grad, epoch)
        '''
        print("\nEpoch: {} | Fader Loss: {:.4f} | Disc Loss {:.4f} | Disc Accuracy {:.4f}".format(epoch,
                                                                                                  fader_loss_summary[
                                                                                                      split + "_epoch"][
                                                                                                      epoch],
                                                                                                  disc_loss_summary[
                                                                                                      split + "_epoch"][
                                                                                                      epoch],
                                                                                                  disc_accuracy_summary[
                                                                                                      split + "_epoch"][
                                                                                                      epoch]))
tb.close()
