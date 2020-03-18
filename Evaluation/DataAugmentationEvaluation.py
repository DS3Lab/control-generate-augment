#### generation ####
from Evaluation.Generation import generate
from Evaluation.train import training
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
############################################## GENERATION ###########################################

date = "2020-Mar-03-10:13:09"
params = pd.read_csv("Parameters/params.csv")
epoch = 6
training_epochs = 30
compare =False
for n_rows in [500,1000,10000]:
    for aug_perc in [120,150,200]:
        name = n_rows
        n_samples_pos =(int((aug_perc/100)*name) + int(0.1*((aug_perc/100)*name)))//2
        n_samples_neg = (int((aug_perc/100)*name) + int(0.1*((aug_perc/100)*name)))//2
        basepath = 'Evaluation/EvaluationData/'
        for aug in [True]:
            print("n_rows: {} aug_perc: {} aug:{}".format(name, aug_perc,aug))
            if not aug:


                if compare and aug_perc>0:
                    n_rows = name + int(name*(aug_perc/100))
                    print("number of samples: ",n_rows)
                    save_model_path = basepath + str(n_rows) + "/Classifier/" + "Baseline/"
                else:
                    n_rows=name
                    print("number of samples: ", n_rows)
                    save_model_path = basepath + str(n_rows) + "/Classifier/" + "Baseline/"

                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)

                X_train = pd.read_csv("data/yelp_train.csv", nrows=n_rows)
                y_train = pd.read_csv("data/y_yelp_train.csv", nrows=n_rows)
                X_train.to_csv("Evaluation/EvaluationData/" + str(n_rows) + "/yelp_train.csv", index=False)
                y_train.to_csv("Evaluation/EvaluationData/" + str(n_rows) + "/y_yelp_train.csv", index=False)

                # training classifier
                training(date, params, str(n_rows), training_epochs, save_model_path)

                # testing


            else:

                save_model_path = basepath + str(n_rows) + "/Classifier/" +str(aug_perc) +"/"
                print(save_model_path)
                print(os.path.exists(save_model_path))
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)

                X_train = pd.read_csv("data/yelp_train.csv", nrows=n_rows)
                y_train = pd.read_csv("data/y_yelp_train.csv", nrows=n_rows)
                pos_sent = generate(date, epoch, "Positive", n_samples_pos)
                neg_sent = generate(date, epoch, "Negative", n_samples_neg)
                print("qui")
                aug_pos = pd.DataFrame({'text': pos_sent})
                aug_neg = pd.DataFrame({'text': neg_sent})
                y_pos_aug = []
                y_neg_aug = []
                for i in range(len(pos_sent)):
                    y_pos_aug.append([0, 1])
                for i in range(len(neg_sent)):
                    y_neg_aug.append([1, 0])

                y_pos_aug = np.asarray(y_pos_aug)
                y_neg_aug = np.asarray(y_neg_aug)
                y_pos_aug = pd.DataFrame({'Negative': y_pos_aug[:, 0], 'Positive': y_pos_aug[:, 1]})
                y_neg_aug = pd.DataFrame({'Negative': y_neg_aug[:, 0], 'Positive': y_neg_aug[:, 1]})
                X_train_aug = pd.concat([X_train, aug_pos, aug_neg])
                y_train_aug = pd.concat([y_train, y_pos_aug, y_neg_aug])
                print("training: ", X_train_aug.shape, y_train_aug.shape)
                X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=42)
                X_train_aug.to_csv("Evaluation/EvaluationData/" + str(n_rows) + "/yelp_train.csv", index=False)
                y_train_aug.to_csv("Evaluation/EvaluationData/" + str(n_rows) + "/y_yelp_train.csv", index=False)

                training(date, params, str(n_rows), training_epochs, save_model_path)



