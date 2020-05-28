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
from Discriminator import Discriminator, RNN_discr, LSTM_discr, NewRNN_discr
from imdb import Processor
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from new_labels import multiple_binary_cross_entropy, multiple_accuracy_attributes
import json

import tqdm


def main(args):
    '''

    This method Train the VAE and the Adversarial Discriminator. The paramethers used for each experiment are
    stored in the file Parameters/params.csv

    :param args: all the training hyperparameters we want to set
    :return:
    '''


    argv = vars(args)

    print("back: ", args.back, args.delta)
    #################### SETTINGS ########################
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    ############## STORING RUNNING PARAMETERS ############
    values = [ts]
    columns = ['time']
    for i in argv.keys():
        values.append(argv[i])
        columns.append(i)
    params = pd.DataFrame([values], columns=columns)

    if not os.listdir('Parameters'):
        print("Directory is empty")
        params.to_csv("Parameters/params.csv", index=False)

    else:
        print("Directory is not empty")
        df = pd.read_csv("Parameters/params.csv")
        df = pd.concat([df, params])
        df = df.set_index('time')
        df.to_csv("Parameters/params.csv")

    ######################################################

    print("STARTING: ", ts)
    splits = ['train', 'valid']
    cuda2 = torch.device('cuda:' + args.gpu)
    device = cuda2 if torch.cuda.is_available() else "cpu"
    save_model = args.save_model
    #######################################################
    ################### DATA ##############################
    data_dir = args.data_dir
    create_data = True
    max_sequence_length = args.max_sequence_length
    rows = args.samples
    #######################################################
    ##################### VAE ############################
    min_occ = args.min_occ
    attr_size = args.attr_size
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    rnn_type = args.rnn_type
    word_dropout = args.word_dropout
    embedding_dropout = args.embedding_dropout
    latent_size = args.latent_size
    num_layers = 1
    batch_size = args.batch_size
    bidirectional = args.bidirectional
    save_model_path = 'bin'
    lr = args.learning_rate
    coeff = args.coeff
    x0 = args.x0
    epochs = args.epochs
    pretrained = args.glove
    datasets = OrderedDict()
    ######################################################
    ################ discriminator params ################

    discr_type = args.discr_type
    h1_size = args.h1_size
    h2_size = args.h2_size
    dropout = args.discr_drop
    maximum = args.max_discr_weight

    ######################################################
    ################### TensorBoard ######################

    z_dataset = np.zeros((1, latent_size))

    ############################################

    data_name = 'imdb'

    for split in splits:
        datasets[split] = Processor(
            data_dir=data_dir,
            split=split,
            create_data=data_name,
            dataset='yelp',
            max_sequence_length=max_sequence_length,
            min_occ=min_occ,
            rows=rows,
            what="text"
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
        embedding_layer=emb_layer,
        word_dropout_type=args.word_drop_type,
        back=args.back
    )

    if torch.cuda.is_available():
        model = model.to(cuda2)

    print(model)

    reduction = "sum"
    # print("Current: ", torch.cuda.current_device())
    save_model_path = os.path.join(save_model_path, ts)
    os.makedirs(save_model_path)

    def create_storage():
        '''

        :return: vocabularies to store all the training and validation scores
        '''
        fader_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
        nll_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
        vae_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
        disc_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
        disc_accuracy_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}
        kl_loss_summary = {'train': [], 'valid': [], 'train_epoch': [], 'valid_epoch': []}

        return fader_loss_summary, vae_loss_summary, disc_loss_summary, disc_accuracy_summary, kl_loss_summary, nll_loss_summary

    def restart(summary, split):
        summary[split] = []
        return summary

    def disc_weight_function(type, threshold, x_step, x, epoch, warmup, eps=1e-5):
        '''

        :param type: (str) type of increase that we want for our discriminator
        :param threshold: (int) maximum weight of the discriminator loss in L_CGA
        :param x_step: (int) iafter how many training steps the threshold is reached
        :param x: (int)current training steps
        :param epoch: (int)
        :param warmup: (float)
        :param eps: (float)
        :return: (float) discriminator weight
        '''
        if x < warmup:
            return 0
        if type == 'linear':
            y = min(threshold, (threshold / (x_step)) * (x - warmup))
            return y
        if type == 'hard':
            return min(threshold, 3 * epoch)

    def kl_anneal_function(anneal_function, x, k, steps, eps=1e-5):

        '''

        :param anneal_function: (str) 'logistic' or 'linear' annealing
        :param x: (int) current training steps
        :param k: deprecated
        :param steps: 'int' numb of training steps necessary to restore the VAE loss
        :param eps: (float)
        :return: (float) KL_Weight
        '''
        if anneal_function == 'logistic':
            k = -(log(-1 + 1 / (1 - eps))) / (0.5 * steps)
            return float((1 / (1 + np.exp(-k * (x - steps)))))

        elif anneal_function == 'linear':
            return min(1, step / x0)

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)

    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0, KL_w='standard'):
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        if KL_w == 'standard':
            KL_weight = kl_anneal_function(anneal_function, step, k, x0)
        if KL_w == 'tanh':
            KL_weight = (np.tanh((step - 4500) / 1000) + 1) / 2

        return NLL_loss, KL_loss, KL_weight

    def discriminator(discr_type):
        '''

        :param discr_type: str 'fc' or 'rnn'
        :return:
        '''
        if discr_type == "fc":
            return Discriminator(input_size=latent_size, h1_size=h1_size, h2_size=h2_size, output_size=attr_size).to(
                device)
        if discr_type == "rnn":
            return NewRNN_discr(input_dim=latent_size, hidden_dim=args.hs_rnn_discr, output_dim=attr_size, n_layers=2,
                                bidirectional=True, dropout=0.7)

            r  # eturn RNN_discr(input_dim=latent_size, hidden_dim=args.hs_rnn_discr, output_dim=attr_size)

    # disc = Discriminator(input_size=latent_size, h1_size=h1_size, h2_size=h2_size, output_size=attr_size).to(device)
    # disc_rnn = RNN_discr(input_dim=latent_size, hidden_dim=latent_size//2, output_dim=attr_size)

    disc = discriminator(discr_type=discr_type).to(device)

    print(disc)
    softmax = torch.nn.Softmax(1)
    fader_opt = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer for the VAE
    disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-2, betas=(0.5, 0.999))  # optimizer for disc

    step = 0
    fader_loss_summary, vae_loss_summary, disc_loss_summary, disc_accuracy_summary, \
    kl_loss_summary, nll_loss_summary = create_storage()
    '''
    train_loader = DataLoader(dataset=datasets['train'], batch_size=batch_size, shuffle=(split == 'train'),
                                     num_workers=cpu_count(), pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(dataset=datasets['valid'], batch_size=batch_size, shuffle=(split == 'train'),
                                     num_workers=cpu_count(), pin_memory=torch.cuda.is_available())
    '''
    tb = SummaryWriter()
    kl_and_weight = []
    for epoch in range(epochs):
        z_dataset = np.zeros((1, latent_size + attr_size))
        labels = np.array([])
        for split in splits:
            print(datasets[split])
            data_loader = DataLoader(dataset=datasets[split], batch_size=args.batch_size, shuffle=(split == 'train'),
                                     num_workers=cpu_count(), pin_memory=torch.cuda.is_available())

            if split == 'train':
                # data_loader = train_loader
                model.train()
            else:
                # data_loader=valid_loader
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)
                # if (batch_size == 1):
                #    continue
                # simply creating variables and no more only simple tensors
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v, cuda2)
                # loading on GPUs

                #########################################################################################################################################################
                #######################################################    DISCRIMINATOR TRAINING   #####################################################################
                #########################################################################################################################################################

                z, label, _ = model(batch['input'].to(device), batch['length'].to(device), batch['label'].to(device),
                                    step,
                                    encoder=True)
                if discr_type == 'fc':
                    attr_probs = disc(z)
                if discr_type == 'rnn':
                    # print("z device: ",z.device)
                    attr_probs = disc(torch.unsqueeze(z, dim=0))
                _, target_attr = label.max(dim=1)
                #            print("attr_probs: {} target_att: {}".format(attr_probs.size(), target_attr.size()))
                # print("attr_probs.size(): ",attr_probs.size())
                disc_loss = multiple_binary_cross_entropy(attr_probs, label)
                # disc_loss = F.cross_entropy(attr_probs, target_attr) #attr_probs are logits

                # Train discriminator
                if split == 'train':
                    disc_opt.zero_grad()
                    disc_loss.backward()
                    disc_opt.step()

                # Compute discriminator accuracy
                accuracy = multiple_accuracy_attributes(attr_probs, label)
                #########################################################################################################################################################
                #######################################################    Varitional-AE TRAINING   #####################################################################
                #########################################################################################################################################################
                # 5: forward pass vae and discriminator
                logp, mean, logv, z, z_a, l1_loss = model(batch['input'].to(device), batch['length'].to(device),
                                                          batch['label'].to(device), step)
                if discr_type == 'fc':
                    attr_probs = disc(z)
                if discr_type == "rnn":
                    attr_probs = disc(torch.unsqueeze(z, dim=0))
                # 6: losses of vae and discriminator
                NLL_loss, KL_loss, KL_weight = loss_fn(logp.to(device), batch['target'].to(device),
                                                       batch['length'].to(device), mean, logv, 'logistic', step, coeff,
                                                       x0, KL_w=args.kl_weight)

                if split == 'train':
                    _, target_attr = batch['label'].max(dim=1)
                    target = target_attr.cpu().detach().numpy()
                    labels = np.concatenate((labels, target))
                    z_saved = z_a.cpu().detach().numpy()
                    z_dataset = np.vstack((z_dataset, z_saved))
                    kl_and_weight.append(KL_loss.item() * KL_weight)

                vae_loss = (NLL_loss + KL_weight * KL_loss) / batch_size

                # fader_discr_loss = (F.cross_entropy(attr_probs, target_attr)) / batch_size
                fader_discr_loss = multiple_binary_cross_entropy(attr_probs, label)
                # if split == 'train': disc_weight = disc_weight_function('linear', 30, 5000, step, epoch)

                if split == 'train': disc_weight = disc_weight_function('linear', threshold=args.max_discr_weight,
                                                                        x_step=50000, x=step, epoch=0,
                                                                        warmup=args.discr_warmup)
                if args.back == "True":
                    fader_loss = (vae_loss + args.delta * l1_loss) - (disc_weight * fader_discr_loss)
                else:
                    fader_loss = vae_loss - (disc_weight * fader_discr_loss)

                if split == 'train':
                    fader_opt.zero_grad()
                    fader_loss.backward()
                    fader_opt.step()
                    step += 1
                if iteration % 100 == 0 and split == 'train':
                    print(
                        '\nPhase: {}| Iteration {}/{}  | NLL_LOSS: {:.4f} | Disc Weight: {:.4f} | Fader Loss: {:.4f} | VAE Loss: {:.4f} | L1_loss: {:.4f} | KL Weight: {:.4f} |Disc Loss: {:.4} | KL Loss: , {:.4f}| Disc Accuracy:  {:.4f}'
                            .format(split.upper(), iteration, len(data_loader), NLL_loss.item(), disc_weight, fader_loss, vae_loss,
                                    l1_loss, KL_weight,
                                    disc_loss, KL_loss / batch_size, accuracy), flush=True)

                if iteration % 100 == 0 and split == 'valid':
                    print(
                        '\nPhase: {}| Iteration {}/{}  |NLL_LOSS: {:.4f}| Disc Weight: {:.4f} | Fader Loss: {:.4f} | VAE Loss: {:.4f} |L1 loss: {:.4f}| KL_Weight; {:.4f} |Disc Loss: {:.4} | KL Loss: {:.4f} | Disc Accuracy:  {:.4f}'
                            .format(split.upper(), iteration, len(data_loader), NLL_loss.item(), disc_weight, fader_loss, vae_loss,
                                    l1_loss, KL_weight,
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
                nll_loss_summary[split].append(NLL_loss.item())


            fader_loss_summary[split + "_epoch"].append(np.mean(fader_loss_summary[split]))
            disc_loss_summary[split + "_epoch"].append(np.mean(disc_loss_summary[split]))
            disc_accuracy_summary[split + "_epoch"].append(np.mean(disc_accuracy_summary[split]))

            nll_loss_summary[split + "_epoch"].append(np.mean(nll_loss_summary[split]))
            with open('data/nll.json', 'w') as fp:
                json.dump(nll_loss_summary[split + "_epoch"], fp)

            fader_loss_summary, disc_loss_summary, disc_accuracy_summary, nll_loss_summary = restart(fader_loss_summary, split), \
                                                                           restart(disc_loss_summary, split), \
                                                                           restart(disc_accuracy_summary, split), restart(nll_loss_summary,split)

            if split == 'train' and args.save_model:
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % (epoch))
                torch.save(model.state_dict(), checkpoint_path)
                z_dataset = pd.DataFrame(z_dataset)
                # z_dataset.to_csv("Latent/" + str(epoch) + ".csv", index=False)
                # pd.DataFrame({'kl*W': kl_and_weight}).to_csv("kl_and_weight_" + args.kl_weight + ".csv", index=False)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('-msl', '--max_sequence_length', type=int, default=20)
    parser.add_argument('--min_occ', type=int, default=2)
    parser.add_argument('-emb_size', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default="gru")
    parser.add_argument('-ep', '--epochs', type=int, default=30)
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.8)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=128)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-bid', '--bidirectional', type=bool, default=False)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--coeff', type=float, default=0.0025)
    parser.add_argument('--x0', type=int, default=3000)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--glove', type=bool, default=False)
    parser.add_argument('--att_latent', type=int, default=100)
    parser.add_argument('--stop_annealing', type=int, default=3300)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--attr_size', type=int, default=7)
    parser.add_argument('--discr_type', type=str, default='fc')
    parser.add_argument('--h1_size', type=int, default=50)
    parser.add_argument('--h2_size', type=int, default=20)
    parser.add_argument('--discr_drop', type=float, default=0)
    parser.add_argument('-mdw', '--max_discr_weight', type=int, default=30)
    parser.add_argument('-kl_w', '--kl_weight', type=str, default='standard')
    parser.add_argument('-wd_type', '--word_drop_type', type=str, default='static')
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--hs_rnn_discr', type=int, default=50)
    parser.add_argument('--back', type=str, default="False")
    parser.add_argument('--discr_warmup', type=int, default=20000)
    args = parser.parse_args()

    main(args)
