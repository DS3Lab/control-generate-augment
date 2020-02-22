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
from math import exp, log
from utils import decoding_ouput
from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE
from Embedder import Embedder
import pandas as pd
from yelp import YELP

def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    splits = ['train', 'valid']

    cuda2 = torch.device("cuda:"+args.gpu)
    device = cuda2 if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir
    create_data = True
    max_sequence_length = args.max_sequence_length
    min_occ = args.min_occ

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
    gamma= args.gamma
    pretrained = args.glove
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
        num_layers=num_layers,
        bidirectional=bidirectional,
        pretrained=pretrained,
        embedding_layer=emb_layer,
        device=device
    )

    if torch.cuda.is_available():
        model = model.to(cuda2)

    print(model)

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


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
            kl_loss_attention = []
            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                # simply creating variables and no more only simple tensors
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                logp, mean, logv, z, KL_attention = model(batch['input'].to(device), batch['length'].to(device))
                NLL_loss, KL_loss, KL_weight = loss_fn(logp.to(device), batch['target'].to(device),
                                                       batch['length'].to(device), mean, logv, 'logistic', step, coeff,
                                                       x0)

                loss = (NLL_loss + KL_weight *(KL_loss + gamma * KL_attention)) / batch_size
                elbo.append(loss.item())
                recon_loss.append(NLL_loss.item() / batch_size)
                kl_loss.append(KL_loss.item() / batch_size)
                kl_loss_attention.append(KL_attention.item()/batch_size)
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
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Attention %9.4f, KL-Weight %6.3f"
                          % (split.upper(), iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
                             KL_loss.item() / batch_size, KL_attention.item() / batch_size, KL_weight))
                if iteration % 100 == 0 and split == "valid":
                    sentences = idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
                                         pad_idx=datasets['train'].pad_idx)
                    sentences_decode = decoding_ouput(logp, datasets['train'].get_i2w())
                    sel = np.random.randint(0, len(sentences))
                    print("PHASE: ", split.upper())
                    print("Input: ", sentences[sel])
                    print("Decoding: ", sentences_decode[sel])

            print("Epoch: ", epoch, " ELBO: ", np.mean(elbo), " NLL-Loss: ", np.mean(recon_loss), " KL-Loss:",
                  np.mean(kl_loss), "KL-Loss-Attention: ", np.mean(kl_loss_attention))

            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % (epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)

# target = target[:, :torch.max(length).data[0]].contiguous().view(-1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()



    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('-msl', '--max_sequence_length', type=int, default=20)
    parser.add_argument('--min_occ', type=int, default=5)
    parser.add_argument('-emb_size', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn','--rnn_type', type=str, default="gru")
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-hs', '--hidden_size', type=int,default=256)
    parser.add_argument('-wd', '--word_dropout', type=float,default=0.7)
    parser.add_argument('-ed', '--embedding_dropout', type=float,default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int,default=16)
    parser.add_argument('-bs', '--batch_size', type=int,default=32)
    parser.add_argument('-bid','--bidirectional', type=bool, default=False)
    parser.add_argument('-lr','--learning_rate', type=float, default=0.001)
    parser.add_argument('--coeff', type=float,default=0.0025)
    parser.add_argument('--x0', type=int,default=2500)
    parser.add_argument('--gamma',type=float,default=0.1)
    parser.add_argument('--glove', type=bool, default=False)

    args = parser.parse_args()

    main(args)