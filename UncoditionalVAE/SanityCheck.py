import pandas as pd
from nltk.tokenize import word_tokenize
from textdistance import levenshtein
import numpy as np
train = pd.read_csv("data/YELP/yelp_train.csv")['review']
train = train.dropna(axis=0)
check = pd.read_csv("data/check.csv")['sentence'].values
print(train.shape)
s = check[2]
scores = []
s = s.replace('<eos>',"")
s_token = word_tokenize(s)
max_len = len(s_token)
print(s, max_len)

score_list = []
for i, sanity in enumerate(train):
    #sanity_token = word_tokenize(sanity[0][:max_len])
    sanity_tokens = word_tokenize(sanity)
    sanity_tokens = sanity_tokens[:max_len]
    sanity = " ".join(sanity_tokens)
    score = levenshtein(s, sanity)
    score_list.append(score)

    print(i, score)

idx = score_list.index(min(score_list))