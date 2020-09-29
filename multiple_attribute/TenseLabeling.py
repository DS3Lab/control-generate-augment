import pandas as pd
import numpy as np
import json
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
import spacy
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from langdetect import detect
from sklearn.utils import shuffle

nlp = spacy.load("en_core_web_sm")
'''

This scripts generates the verbe tense labels for the dataset and it creates train, dev, test with sentiment and 
verbe tense
'''

def find_indexes(labels):
    labels = np.asarray(labels)
    return np.where(labels==-1)[0].tolist()



def drop_indexes(df, labels):
    indexes = find_indexes(labels)
    df = df.drop(df.index[indexes])
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    return df


def hot_one(labels):
    l = []

    for i in labels:
        if i == 0:
            l.append([1, 0])
        if i == 1:
            l.append([0, 1])
    l = np.asarray(l)
    labels = pd.DataFrame({'Present': l[:, 0], 'Past': l[:, 1]})
    return labels

def pos(sentence):
    sentence = sent_tokenize(sentence)[0]
    doc = nlp(sentence)
    present = 0
    past = 0
    for idx, token in enumerate(doc):
        if token.tag_ == "VBP" or token.tag_ == "VBZ":
            present += 1
        if token.tag_ == "VBD":
            past += 1
    if present > past:
        return "pres", sentence
    if past > present:
        return "pas", sentence
    else:
        return "no", sentence

def labeling(dataset):

    labels = []
    unlabeled = []
    for i in range(dataset.shape[0]):
        sentence = dataset.text.iloc[i]
        tag, sentence = pos(sentence)
        if tag == "pres":
            labels.append(0)
        if tag == "pas":
            labels.append(1)
        if tag == 'no':
            labels.append(-1)

    return labels

split = ['train','valid']

#TRAINING
data_train = pd.read_csv("data/yelp_train.csv")
labels_train = labeling(data_train)
X_train = drop_indexes(data_train, labels_train)
y_train = hot_one(labels_train)
X_train.to_csv("data/TenseData300/proxy_yelp_train.csv", index=False)
y_train.to_csv("data/TenseData300/proxy_y_yelp_train.csv", index=False)


#DEV
data_dev = pd.read_csv("data/yelp_dev.csv")
labels_dev = labeling(data_dev)
X_dev = drop_indexes(data_dev, labels_dev)
y_dev = hot_one(labels_dev)
assert X_dev.shape[0] == y_dev.shape[0]
X_dev, y_dev = shuffle(X_dev,y_dev)
X_dev.to_csv("data/TenseData300/yelp_valid.csv", index=False)
y_dev.to_csv("data/TenseData300/y_yelp_valid.csv", index=False)

#TEST
data_test = pd.read_csv("data/yelp_test.csv")
labels_test = labeling(data_test)
X_test = drop_indexes(data_test, labels_test)
y_test = hot_one(labels_test)
assert X_test.shape[0] == y_test.shape[0]
X_test.to_csv("data/TenseData300/yelp_test.csv", index=False)
y_test.to_csv("data/TenseData300/y_yelp_test.csv", index=False)


#training with labels
new_x_train = pd.concat([X_train, X_test])
new_y_train = pd.concat([y_train,y_test])
#new_x_train, new_y_train = shuffle(new_x_train, new_y_train)
new_x_train.to_csv("data/TenseSentiment300/yelp_train.csv", index=False)
new_y_train.to_csv("data/TenseSentiment300/y_yelp_train.csv", index=False)


####################################################################################################

# sentiment labels for the data processing
# this is for the training ##
path = "/Users/giusepperusso/Desktop/github/Thesis_Data/data_sentiment_300/"
y_sent_train = pd.read_csv(path + "y_yelp_train.csv")
y_sent_train = drop_indexes(y_sent_train, labels_train)
y_sent_test = pd.read_csv(path+"y_yelp_test.csv")
y_sent_test = drop_indexes(y_sent_test, labels_test)
new_y_sent_train = pd.concat([y_sent_train, y_sent_test])
new_y_sent_tense_train = pd.concat([new_y_sent_train, new_y_train], axis=1)
new_x_train.to_csv("data/TenseSentiment300/yelp_train.csv", index=False)
new_y_sent_tense_train.to_csv("data/TenseSentiment300/y_yelp_train.csv", index=False)

# validation


y_sent_dev = pd.read_csv(path + "y_yelp_dev.csv")
y_sent_dev = drop_indexes(y_sent_dev, labels_dev)
X_dev.to_csv("data/TenseSentiment300/yelp_valid.csv", index=False)
new_y_sent_tense_dev = pd.concat([y_dev, y_sent_dev], axis=1).to_csv("data/TenseSentiment300/y_yelp_dev.csv", index=False)






