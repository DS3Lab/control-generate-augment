from Evaluation.Classifier import RNN
import torch
import json
from multiprocessing import cpu_count
import numpy as np
from sklearn.metrics import accuracy_score
from collections import OrderedDict, defaultdict
from Evaluation.imdb import Processor
import pandas as pd
from torch.utils.data import DataLoader
from Evaluation.utils import to_var
import torch.nn as nn
import os
model_name = "Baseline/"
n_rows = 20*(1000)

results = []
for epoch in range(4,30):
    epoch = str(epoch)


    data_name = 'imdb'
    gpu = "1"
    cuda2 = torch.device('cuda:'+ gpu)
    device = cuda2 if torch.cuda.is_available() else "cpu"
    data_dir = "Evaluation/TestData"
    date = "2020-Mar-02-18:50:01"
    params = pd.read_csv("Parameters/params.csv")
    params = params.set_index('time')
    exp_descr = params.loc[date]
    datasets = OrderedDict()
    splits = ['test']

    vocab_dir = "Evaluation/EvaluationData/" + str(n_rows) +"/Classifier/" + model_name

    for split in splits:
        datasets[split] = Processor(
            data_dir=data_dir,
            split=split,
            create_data=data_name,
            dataset='yelp',
            max_sequence_length=exp_descr["max_sequence_length"],
            min_occ=exp_descr["min_occ"],
            rows=5000,
            what="text",
            vocab_directory=vocab_dir
        )

    embedding_size = 300
    pretrained = False
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    emb_layer=None
    acc = lambda x: 1 if x >=0.5 else 0
    m = nn.Sigmoid()
    batch_size=32
    aug = 10

    with open(vocab_dir +"/yelp_vocab.json", 'r') as file:
        vocab = json.load(file)


    w2i, i2w = vocab['w2i'], vocab['i2w']
    INPUT_DIM = len(w2i)

    model = RNN(INPUT_DIM,
                    embedding_size,
                    HIDDEN_DIM,
                    OUTPUT_DIM,
                    N_LAYERS,
                    BIDIRECTIONAL,
                    DROPOUT,
                    w2i['<pad>'],
                    embedding_layer=emb_layer,
                    sos_idx=w2i['<sos>'],
                    eos_idx= w2i['<eos>'],
                    unk_idx=w2i['<unk>'],
                    device="cpu")

    load_checkpoint = vocab_dir + "E" + str(epoch) + ".pytorch"


    if not os.path.exists(load_checkpoint):
        raise FileNotFoundError(load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    model.load_state_dict(torch.load(load_checkpoint, map_location=torch.device(device)))

    loss = nn.BCELoss()

    model.eval()
    test_loader = DataLoader(dataset=datasets['test'], batch_size=batch_size,
                                         num_workers=cpu_count(),pin_memory=torch.cuda.is_available())
    accuracy_res = []
    for iteration, batch in enumerate(test_loader):

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

        class_prob = m(logits)
        class_prob = class_prob.detach().cpu().numpy()
        y_true = target.detach().cpu().numpy()
        y_pred = np.asarray([acc(i) for i in class_prob])
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_res.append(accuracy)

    print("Model: {} -- Test Accuracy: {}".format(model_name,np.mean(accuracy_res)))
    results.append(np.mean(accuracy_res))

