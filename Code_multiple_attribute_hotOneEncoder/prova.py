import torch
import torch.nn.utils.rnn as rnn_utils
import time
a = torch.Tensor([[1], [2], [3]])
b = torch.Tensor([[4], [5]])
c = torch.Tensor([[6]])
d = torch.Tensor([[7],[8],[9],[10]])
batch = [a,b,c,d]
padded= rnn_utils.pad_sequence(batch, batch_first=True)
for x in padded:
    print(len(x))
sorted_batch_lengths = [len(x) for x in padded]
print(sorted_batch_lengths)
packed = rnn_utils.pack_padded_sequence(padded, sorted_batch_lengths, batch_first=True, enforce_sorted=False).to('cuda:1')
lstm = torch.nn.LSTM(input_size=1, hidden_size=3, batch_first=True).cuda('cuda:1')
lstm(packed)
print("end")
time.sleep(120)

##############################
import pandas as pd

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


file = pd.read_csv("data/yelp_valid.csv")
sid = SentimentIntensityAnalyzer()
file = file.dropna(axis=0)
doubt = []
positive = []
y_positive = []
negative = []
y_negative = []
for idx, i in enumerate(file.review):

    score = sid.polarity_scores(i)['compound']
    if score >= 0.05:
        y_positive.append(1)
        positive.append(i)
    if score <= -0.05:
        y_negative.append(0)
        negative.append(i)
    if score> -0.05 and score< 0.05:
        doubt.append(i)

data = [positive, negative]
print(len(data))
flatten = lambda l: [item for sublist in l for item in sublist]
data = flatten(data)

# converting in one hot encoding

x_pos = positive[:int(1.5*len(negative))]
x_neg = negative
X = [x_pos, x_neg]
y_pos = y_positive[:int(1.5*len(negative))]
y_neg = y_negative

X = flatten([x_pos, x_neg])
y = flatten([y_pos, y_neg])

X = pd.DataFrame({'review': X})
y = pd.DataFrame({'labels': y})

#
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import numpy as np
enc = OneHotEncoder(handle_unknown='ignore')
y = enc.fit_transform(y).toarray()
y = pd.DataFrame({'Negative': y[:,0], 'Positive': y[:,1]})

X, y = shuffle(X, y)

X.to_csv("data/yelp_valid.csv", index=False)
y.to_csv("data/y_yelp_valid.csv", index=False)


####################
import pandas as pd
import numpy as np
file = pd.read_csv("data/yelp_train.csv")
labels = pd.read_csv("data/y_yelp_train.csv")
labels = labels.drop(labels.columns[0], axis=1)


l = labels.values
l = np.argmax(labels,axis=1)

for i in range(4):


import pandas as pd
pd.read_csv("data/sentiment.csv")