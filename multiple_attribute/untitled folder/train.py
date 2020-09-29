#### generation ####
from Evaluation.Classifier import RNN
from collections import OrderedDict
from Evaluation.imdb import Processor
from Embedder import Embedder
import torch
from multiprocessing import cpu_count
import numpy as np
from sklearn.metrics import accuracy_score
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import to_var
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from Evaluation.EarlyStopping import Monitor

def training(date, params, folder, epochs, save_model_path,dataset, save=True):


    '''
    Training procedure for the LSTM Sentiment classifier for the data augmentation task

    :param date:
    :param params:
    :param folder:
    :param epochs:
    :param save_model_path:
    :param dataset:
    :param save:
    :return:
    '''
    # considering the experiment
    params = params.set_index('time')
    exp_descr = params.loc[date]
    splits = ['train', 'valid']
    datasets = OrderedDict()
    vocab_dir = save_model_path
    ###########################
    gpu = "1"
    cuda2 = torch.device('cuda:' + gpu)
    device = cuda2 if torch.cuda.is_available() else "cpu"
    early_stopping = Monitor(patience=7, delta=0)
    # così le directories dovrebbero essere sistemate
    for split in splits:

        if split == 'train':
            dataset = "yelp"
            data_dir = folder
            rows=-1

        else:
            dataset = "yelp"
            data_dir = "Evaluation/TestData"
            rows=5000
        print(data_dir)
        datasets[split] = Processor(
            data_dir=data_dir,
            split=split,
            create_data=True,
            dataset=dataset,
            max_sequence_length=exp_descr["max_sequence_length"],
            min_occ=2,
            rows=rows,
            what="text",
            vocab_directory=vocab_dir
        )

    # initializing the model
    embedding_size = 300
    pretrained = False
    INPUT_DIM = datasets['train'].vocab_size
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL =True
    DROPOUT = 0.8
    if pretrained:
        embedding_dir = "../glove/glove.6B." + str(embedding_size) + "d.txt"
        glove = Embedder(filepath=embedding_dir)
        embedding_mtx = glove.build_weight_matrix(datasets['train'].w2i.keys(), embedding_size)
        emb_layer, num_embeddings, embedding_dim = glove.create_embedding_layer(weight_matrix=embedding_mtx)
    else:
        emb_layer = None

    model = RNN(INPUT_DIM,
                embedding_size,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                datasets['train'].pad_idx,
                embedding_layer=emb_layer,
                sos_idx=datasets['train'].sos_idx,
                eos_idx=datasets['train'].eos_idx,
                unk_idx=datasets['train'].unk_idx,
                device=device)
    print(model)

    clf_opt = optim.Adam(model.parameters(), lr=0.001)
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    model = model.to(device)
    batch_size = 32

    acc = lambda x: 1 if x >= 0.5 else 0
    tb = SummaryWriter()
    train_loss = []
    valid_loss = []
    batch_size = 16
    train_loader = DataLoader(dataset=datasets['train'], batch_size=batch_size, shuffle=(split == 'train'),
                              num_workers=cpu_count(), pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(dataset=datasets['valid'], batch_size=batch_size, shuffle=(split == 'valid'),
                              num_workers=cpu_count(), pin_memory=torch.cuda.is_available())
    valid_accuracy = []
    valid_loss_epoch = []
    train_loss_epoch = []
    for epoch in range(epochs):

        for split in splits:
            if split == 'train':
                # data_loader = train_loader

                model.train()
                data_loader = train_loader
            else:
                # data_loader=valid_loader

                data_loader = valid_loader
                model.eval()
            accuracy_res = []
            valid_loss_tracker = []
            train_loss_tracker = []
            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                # if (batch_size == 1):
                #    continue
                #
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v, cuda2)
                logits, label = model(batch['input'].to(device), batch['length'].to(device), batch['label'].to(device))
                logits = logits.squeeze(1)
                _, target = label.max(dim=1)
                target = target.type(torch.FloatTensor)
                clf_loss = loss(m(logits), target)
                if split == 'valid':
                    valid_loss_tracker.append(clf_loss.item())
                if split == "train":
                    train_loss_tracker.append(clf_loss.item())
                # optimization
                if split == 'train':
                    clf_opt.zero_grad()
                    clf_loss.backward()
                    clf_opt.step()
                # accuracy

                class_prob = m(logits)
                class_prob = class_prob.detach().cpu().numpy()
                y_true = target.detach().cpu().numpy()
                y_pred = np.asarray([acc(i) for i in class_prob])
                accuracy = accuracy_score(y_true, y_pred)
                accuracy_res.append(accuracy)
                if split == 'train':
                    tb.add_histogram('train_acc: ', np.mean(accuracy), iteration)

            if split == 'valid':
                valid_accuracy.append(np.mean(accuracy_res))
                valid_loss_epoch.append(np.mean(valid_loss_tracker))
                print("{} Epoch: {} Accuracy: {} loss: {}".format(split, epoch, np.mean(accuracy_res), np.mean(valid_loss_tracker)))
                early_stopping(epoch, valid_loss_epoch)
            if split == 'train':
                train_loss.append(np.mean(accuracy_res))
                train_loss_epoch.append(np.mean(train_loss_tracker))

            if split == 'train' and save:
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % (epoch))
                torch.save(model.state_dict(), checkpoint_path)


        if early_stopping.stop:
            print("stopping the training at epoch:  ", epoch-7)
            true_epoch = epoch-patience -1
            break
    plt.plot(train_loss_epoch, label="train_loss")
    plt.plot(valid_loss_epoch, label="valid_loss")
    plt.legend()
    plt.ylim(0,1)
    plt.show()

    return true_epoch, valid_loss_epoch[early_stopping.epoch], valid_accuracy[early_stopping.epoch]