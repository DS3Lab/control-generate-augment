import json
import os
from os import listdir
from os.path import isfile, join
from model import SentenceVAE
from utils import to_var, idx2word, interpolate
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def generate(date, epoch, sentiment, n_samples):
    date = date
    cuda2 = torch.device('cuda:0')
    epoch = epoch
    #date = "2020-Feb-26-17:47:47"
    #exp_descr = pd.read_csv("EXP_DESCR/" + date + ".csv")
    #print("Pretained: ", exp_descr['pretrained'][0])
    #print("Bidirectional: ", exp_descr['Bidirectional'][0])
    #epoch = str(10)
    #data_dir = 'data'
    #


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
    num_samples = 2
    save_model_path = 'bin'
    ptb = False
    if ptb == True:
        vocab_dir = '/ptb.vocab.json'
    else:
        vocab_dir = '/yelp_vocab.json'

    with open("bin/" + date+"/"+ vocab_dir, 'r') as file:
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
        word_dropout=0,
        embedding_dropout=0,
        latent_size=latent_size,
        num_layers=num_layers,
        cuda = cuda2,
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
    def attr_generation(n):
        labels = np.random.randint(2, size=n)
        enc = OneHotEncoder(handle_unknown='ignore')
        labels = np.reshape(labels, (len(labels), 1))
        enc.fit(labels)
        one_hot = enc.transform(labels).toarray()
        one_hot = one_hot.astype(np.float32)
        one_hot = torch.from_numpy(one_hot)
        return one_hot

    model.eval()
    labels = attr_generation(n=num_samples)


    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sklearn.metrics import accuracy_score
    analyser = SentimentIntensityAnalyzer()

    def sentiment_analyzer_scores(sentence):
        score = analyser.polarity_scores(sentence)
        if score['compound']> 0.05:
            return 1, 'Positive'
        else:
            return 0, 'Negative'

    print('----------SAMPLES----------')
    labels = []
    generated = []
    for i in range(n_samples):
        samples, z, l = model.inference(sentiment)
        s = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
        #print(sentiment_analyzer_scores(s[0]))
        #if sentiment_analyzer_scores(s[0])[1] == sentiment:
        generated.append(s[0])
        print(s[0])

        #labels.append(sentiment_analyzer_scores(s[0])[0])
        #print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    print(sum(labels))
    translation = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
    return generated
    '''
    z1 = torch.randn([latent_size]).numpy()
    z2 = torch.randn([latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())

    samples, _ = model.inference(z=z, n=20)

    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    # for root, dirs, files in os.walk(save_model_path):
    #    for name in files:
    #            print(name, root, dirs)

    ## saving ##
    import numpy as np

    filename = "check.csv"

    translation = np.unique(translation)

    def clean(translation):
        sentences = []

        for s in translation:
            splitted = s.split()
            last = splitted[len(splitted) - 1]
            if '<eos>' == last:
                s = s.replace('<eos>', '')
                sentences.append(s)

            else:
                 print(s)

        return sentences

    t = clean(translation)
    labels = np.ones((len(t))).astype(int)

    X = pd.DataFrame({'sentence': t, 'label': labels})
    X.to_csv("data/"+filename)
    '''
date = "2020-Mar-15-16:15:32"
epoch = 13

generate(date, epoch,"PresNeg", 10)
#generate(date, epoch, "PastPos", 2)

#generate(date, epoch, "PresPos", 2)
#generate(date, epoch, "PresNeg", 2)


