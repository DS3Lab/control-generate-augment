from imdb import Processor
from collections import OrderedDict
from Embedder import Embedder
import json
import io
import os
from model import SentenceVAE
import pandas as pd
import torch

def ft_vocabulary(w2i, i2w, ft_vocab):
    idx = len(w2i)
    print(len(ft_vocab))
    counter = 0
    for key in ft_vocab.keys():
        if key in w2i.keys():
            continue
        else:
            print(key)
            w2i[key] = idx
            i2w[str(idx)] = key
            idx+=1
            counter += 1
    print("counter: ", counter)
    return w2i, i2w

def save_vocab(w2i, i2w, save_path):

    # save path must be the data dir of the main dataset
    vocab = dict(w2i=w2i, i2w=i2w)
    vocab_file_ft = "imdb" + '_' + 'vocab.json'

    with io.open(os.path.join(save_path, vocab_file_ft), 'wb') as vocab_file:
        data = json.dumps(vocab, ensure_ascii=False)
        vocab_file.write(data.encode('utf8', 'replace'))

def embedding_layer(w2i, embedding_size=300):
    embedding_dir = "glove/glove.6B." + str(embedding_size) + "d.txt"
    glove = Embedder(filepath=embedding_dir)
    embedding_mtx = glove.build_weight_matrix(w2i.keys(), embedding_size)
    emb_layer, num_embeddings, embedding_dim = glove.create_embedding_layer(weight_matrix=embedding_mtx)
    return emb_layer, num_embeddings, embedding_dim


def load_model(date, embed_layer, epoch, device):
    vocab_dir = '/yelp_vocab.json'
    with open("bin/" + date+"/"+ vocab_dir, 'r') as file:
        vocab = json.load(file)


    w2i, i2w = vocab['w2i'], vocab['i2w']


    ############## parameters ##############

    params = pd.read_csv("Parameters/params.csv")
    params = params.set_index('time')
    exp_descr = params.loc[date]
    # 2019-Dec-02-09:35:25, 60,300,256,0.3,0.5,16,False,0.001,10,False

    embedding_size = exp_descr["embedding_size"]
    hidden_size = exp_descr["hidden_size"]
    rnn_type = exp_descr['rnn_type']
    word_dropout = exp_descr["word_dropout"]
    embedding_dropout = exp_descr["embedding_dropout"]
    latent_size = exp_descr["latent_size"]
    num_layers = 1
    batch_size = exp_descr["batch_size"]
    bidirectional = bool(exp_descr["bidirectional"])
    max_sequence_length = exp_descr["max_sequence_length"]
    back = exp_descr["back"]
    attribute_size = exp_descr["attr_size"]
    wd_type = exp_descr["word_drop_type"]




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
        cuda = device,
        bidirectional=bidirectional,
        attribute_size=attribute_size,
        word_dropout_type='static',
        back=back
    )

    print(model)
    # Results
    # 2019-Nov-28-13:23:06/E4-5".pytorch"

    load_checkpoint = "bin/" + date + "/" + "E" + str(epoch) + ".pytorch"
    # load_checkpoint = "bin/2019-Nov-28-12:03:44 /E0.pytorch"

    if not os.path.exists(load_checkpoint):
        raise FileNotFoundError(load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    model.load_state_dict(torch.load(load_checkpoint, map_location=torch.device(device)))
    model.embedding = embed_layer
    return model


def finetuning(datasets, data_dir, device, embedding_size=300):

    ft_datasets = OrderedDict()
    splits = ['train', 'valid']
    min_occ = 5
    max_sequence_length = 20
    rows = -1
    ft_data_dir = "data/Finetuning/IMDB"
    for split in splits:
        ft_datasets[split] = Processor(
            data_dir=ft_data_dir,
            split=split,
            create_data=True,
            dataset='imdb',
            max_sequence_length=max_sequence_length,
            min_occ=min_occ,
            rows=rows,
            what="text"
        )

    w2i, i2w = ft_vocabulary(datasets['train'].w2i, datasets['train'].i2w, ft_datasets['train'].w2i)

    save_vocab(w2i, i2w, data_dir)
    print("new embedding vocab: ", len(w2i))
    emb_layer, num_embeddings, embedding_dim = embedding_layer(w2i)
    date = "2020-Mar-03-10:13:09"
    epoch = 6
    model = load_model(date, emb_layer,epoch, device)

    return model





