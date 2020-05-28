import pandas as pd
import spacy
import numpy as np
from nltk.tokenize import sent_tokenize
yelp_train = pd.read_csv("data/yelp_valid.csv")

nlp = spacy.load("en_core_web_sm")

'''
METHODS TO LABELING THE SENTENCES AS PLURAL, SINGULAR OR NEUTRAL

'''

def pos(sentence):

    '''
    Pos labels one single sentences by using Part-Of-Speech Tagging
    :param sentence: (STR)
    :return:
    '''
    sentence = sent_tokenize(sentence)[0]
    doc = nlp(sentence)
    singular = 0
    plural = 0
    neutral = 0
    for idx, token in enumerate(doc):
        if token.tag_ == "NN" or str(token).lower() in ["i","he","she", "it", "myself"]:
            singular+=1
        if token.tag_ == "NNS" or str(token).lower() in ["we", "they", "themselves", "themself", "ourselves", "ourself"]:
            plural+=1
    if singular > plural:
        #print("SINGULAR: ",sentence)
        return [1,0,0]
    if singular < plural:
        #print("PLURAL: ",sentence)
        return [0,0,1]
    if singular == plural:
        #print("NEUTRAL: ",sentence)
        return [0,1,0]
labels = []
for idx, s in enumerate(yelp_train.text):
    print(idx)
    label = pos(s)
    labels.append(label)

labels = np.asarray(labels)

labels = np.asarray(labels)
labels = pd.DataFrame({'Singular': labels[:,0],
                       'Neutral': labels[:,1],
                       'Plural': labels[:,2]})
split = "valid"
labels.to_csv("data/SingularPlural/y_yelp_"+split+".csv", index=False)
