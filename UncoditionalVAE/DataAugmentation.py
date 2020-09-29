from Generation import generate
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
X_train = pd.read_csv("data/yelp_train.csv", nrows=500).to_csv("Evaluation/EvaluationData/500/yelp_train.csv", index=False)
y_train = pd.read_csv("data/y_yelp_train.csv", nrows=500).to_csv("Evaluation/EvaluationData/500/y_yelp_train.csv", index=False)
n_rows = 1000
X_train = pd.read_csv("data/yelp_train.csv", nrows=n_rows)
y_train = pd.read_csv("data/y_yelp_train.csv", nrows=n_rows)

date = "2020-Mar-03-10:13:09"
epoch = 7
sentiment = "Positive"
n_samples_pos = 110
n_samples_neg = 110
pos_sent = generate(date, epoch,"Positive", n_samples_pos)
neg_sent = generate(date, epoch,"Negative", n_samples_neg)
aug_pos = pd.DataFrame({'text': pos_sent})
aug_neg = pd.DataFrame({'text': neg_sent})
y_pos_aug = []
y_neg_aug = []
for i in range(len(pos_sent)):
    y_pos_aug.append([0,1])
for i in range(len(neg_sent)):
    y_neg_aug.append([1,0])

y_pos_aug = np.asarray(y_pos_aug)
y_neg_aug = np.asarray(y_neg_aug)
y_pos_aug = pd.DataFrame({'Negative': y_pos_aug[:,0], 'Positive': y_pos_aug[:,1]})
y_neg_aug = pd.DataFrame({'Negative': y_neg_aug[:,0], 'Positive': y_neg_aug[:,1]})

# concatenation

#X_train_aug = pd.concat([X_train, aug_pos, aug_neg])
#y_train_aug = pd.concat([y_train, y_pos_aug, y_neg_aug])
X_train_aug = pd.concat([X_train,aug_pos, aug_neg])
y_train_aug = pd.concat([y_train,y_pos_aug, y_neg_aug])
print(X_train_aug.shape, y_train_aug.shape)
aug = True
val = "1000"
if aug:
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug,random_state=42)
    X_train_aug.to_csv("Evaluation/EvaluationData/"+val+"/yelp_train.csv", index=False)
    y_train_aug.to_csv("Evaluation/EvaluationData/"+val+"/y_yelp_train.csv", index=False)
else:
    X_train.to_csv("Evaluation/EvaluationData/"+val+"/yelp_train.csv", index=False)
    y_train.to_csv("Evaluation/EvaluationData/"+val+"/y_yelp_train.csv", index=False)
