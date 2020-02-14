import pandas as pd
import operator
import numpy as np
import spacy
'''

def creation():
    d = {i: 0 for i in range(5)}
    return d


data = pd.read_csv("data/yelp_train.csv", nrows=500).text
nlp = spacy.load("en_core_web_sm")
labels = []
for idx, s in enumerate(data):
    doc = nlp(s)
    d = creation()
    for token in doc:
        if token.tag_ == "PRP":
            print(idx, token)
            token = str(token)
            if token == "i" or token == "me" or token == "myself":
                d[0] += 1
            if token == "you" or token == "u":
                d[1] += 1
            if token == "he" or token == "she" or token == "it":
                d[2] += 1
            if token == "we":
                d[3] += 1
            if token == "they":
                d[4] += 1
    # we generate the labels

    label = max(d.items(), key=operator.itemgetter(1))[0]
    labels.append(label)
'''
import torch
import torch.nn.functional as F

def multiple_binary_cross_entropy(output, target, attr_numb = 4):

    bce_loss = 0
    print("inside: ", target)
    for i in range(attr_numb):
        pred_attribute = torch.sigmoid(output[:, i])
        target_attribute = target[:,i]
        bce_loss += F.binary_cross_entropy(pred_attribute, target_attribute)
    bce_loss = bce_loss/attr_numb
    return bce_loss



'''
d = {0: np.array([0, 1, 0, 1]),
     1: np.array([0, 1, 1, 0]),
     2: np.array([1, 0, 0, 1]),
     3: np.array([1, 0, 1, 0])
     }

y_train = pd.read_csv("data/y_yelp_valid.csv")
y = np.zeros((1, 4))
for i in range(y_train.shape[0]):
    l = y_train.iloc[i].values
    key = np.argmax(l)
    y = np.vstack((y, d[key]))
y = np.delete(y, 0, 0)
y = pd.DataFrame(y, dtype=int)
y.to_csv("data/y_yelp_valid.csv", index=False)
'''