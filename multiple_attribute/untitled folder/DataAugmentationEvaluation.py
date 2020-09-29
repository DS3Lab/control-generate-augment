#### generation ####
from Generation import generate ## same scriot as the one in the main folcer
from train import training
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
import torch
############################################## GENERATION ###########################################
### TO CREATE THE DATA FOR THE GPU_DAE.py file###


'''This Script creates all the training dataset with all the different sizes augmented with Synthetic and 
True Data'''

date = "2020-Mar-24-21:43:37"
epoch_sel = 10
params = pd.read_csv("Parameters/params.csv")
training_epochs = 100
compare = False
dataset = "imdb"
for seed in [4]:
    torch.manual_seed(seed)
    for n_rows in [1000]:
        for aug_perc in [100]:
            name = n_rows
            n_samples_pos =(int((aug_perc/100)*name) + int(0.1*((aug_perc/100)*name)))//2
            n_samples_neg = (int((aug_perc/100)*name) + int(0.1*((aug_perc/100)*name)))//2
            basepath = 'Evaluation/EvaluationData/'
            print("Seed: {} n_rows: {} aug_perc: {} ".format(seed, name, aug_perc))
            if aug_perc == 0:


                if compare and aug_perc>0:
                    n_rows = name + int(name*(aug_perc/100))
                    print("number of samples: ",n_rows)
                    basepath = 'Evaluation/EvaluationData/RealData/'

                    save_model_path = basepath + str(n_rows) + "/Classifier/" + "Baseline/Seed_" + str(seed) +"/"
                else:
                    n_rows=name
                    print("number of samples: ", n_rows)
                    save_model_path = basepath + str(n_rows) + "/Classifier/" + "Baseline/Seed_" + str(seed) +"/"

                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)

                X_train = pd.read_csv("data/"+dataset + "_train.csv", nrows=n_rows)
                y_train = pd.read_csv("data/y_"+ dataset +"_train.csv", nrows=n_rows)
                X_train.to_csv(save_model_path + dataset + "_train.csv", index=False)
                y_train.to_csv(save_model_path + "y_" + dataset + "_train.csv", index=False)

                # training classifier
                #epoch, valid_loss, valid_acc = training(date, params, str(n_rows), training_epochs, save_model_path)
                #print(valid_loss, valid_acc)
                # saving_results
                '''
                save_score_path = basepath + str(n_rows) + "/Classifier/" + "Baseline/Scores"

                score = pd.DataFrame({'seed': [seed], 'epoch': [epoch], 'valid_acc': [valid_acc], 'valid_loss': [valid_loss]})
                if not os.path.isdir(save_score_path):
                    os.mkdir(save_score_path)
                if not os.listdir(save_score_path):
                    print("Directory is empty")
                    score.to_csv(save_score_path + "/scores.csv", index=False)

                else:
                    print("Directory is not empty")
                    df = pd.read_csv(save_score_path + "/scores.csv")
                    df = pd.concat([df, score])
                    df.to_csv(save_score_path + "/scores.csv", index=False)
                '''
            else:

                save_model_path = basepath + str(n_rows) + "/Classifier/" +str(aug_perc) +"/Seed_" + str(seed) +"/"
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)

                X_train = pd.read_csv("data/" + dataset + "_train.csv", nrows=n_rows)
                y_train = pd.read_csv("data/y_" + dataset + "_train.csv", nrows=n_rows)
                pos_sent = generate(date, epoch_sel, "Positive", n_samples_pos)
                neg_sent = generate(date, epoch_sel, "Negative", n_samples_neg)
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
                X_train_aug.to_csv(save_model_path + dataset + "_train.csv", index=False)
                y_train_aug.to_csv(save_model_path + "/y_" + dataset+"_train.csv", index=False)

                #epoch, valid_loss, valid_acc = training(date, params, str(n_rows), training_epochs, save_model_path)
                #print(valid_loss, valid_acc)


                # results
                '''
                # saving_results
                save_score_path = basepath + str(n_rows) + "/Classifier/" +str(aug_perc)+ "/Scores"

                score = pd.DataFrame({'seed': [seed], 'epoch': [epoch], 'valid_acc': [valid_acc], 'valid_loss': [valid_loss]})
                if not os.path.isdir(save_score_path):
                    print("creasting")
                    os.mkdir(save_score_path)
                if not os.listdir(save_score_path):
                    print("Directory is empty")
                    score.to_csv(save_score_path + "/scores.csv", index=False)

                else:
                    print("Directory is not empty")
                    df = pd.read_csv(save_score_path + "/scores.csv")
                    df = pd.concat([df, score])
                    df.to_csv(save_score_path + "/scores.csv", index=False)
                    
                '''


