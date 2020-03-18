import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import os
def resetting(dataset):
    return dataset.reset_index()

yelp = pd.read_csv("data/verbs.csv")
sentences = []
for i in yelp['text']:
    sentences.append(i.lower())
yelp = pd.DataFrame({'review': sentences})


labels = pd.read_csv("data/verbs_tag.csv")['labels'].values
enc = OneHotEncoder(handle_unknown='ignore')
labels = np.reshape(labels,(len(labels),1))
enc.fit(labels)
labels = enc.transform(labels).toarray().astype(int)
labels = pd.DataFrame({"Negative":labels[:,0], "Positive":labels[:,1]})
yelp_train, yelp_val, y_train, y_val = train_test_split(yelp,  labels, test_size=0.1, random_state=42)
yelp_train, yelp_val, y_train, y_val = resetting(yelp_train), resetting(yelp_val), resetting(y_train), resetting(y_val)


if not os.path.exists("Restaurant"):
    os.makedirs("Restaurant")
data_path = "Restaurant/"
yelp_train.to_csv(data_path + "yelp_train.csv", index=False)
yelp_val.to_csv(data_path + "yelp_valid.csv", index=False)
y_train.to_csv(data_path + "y_yelp_train.csv", index=False)
y_val.to_csv(data_path + "y_yelp_valid.csv", index=False)
