#### generation ####
from Evaluation.Generation import generate
from Evaluation.train import training
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
import torch

############################################## GENERATION ###########################################

'''
Script to compute the Validation and test scores on all the Data Augmentated datasets:
1) CGA AUGMNENTED DATA
2) TRUE-DATA AUGMENTED
3) EDA (Easy Data Augmentation) AUGMENTED
'''


date = "2020-Mar-03-10:13:09"
params = pd.read_csv("Parameters/params.csv")
epoch_sel = 6
training_epochs = 100
compare = False

for seed in [0, 12, 22, 32, 52]:
    torch.manual_seed(seed)
    for n_rows in [500]:
        for aug_perc in [120]:
            name = n_rows

            basepath = 'Evaluation/EvaluationData/'
            print("Seed: {} n_rows: {} aug_perc: {} ".format(seed, name, aug_perc))
            if aug_perc == 0:

                if compare and aug_perc > 0:
                    n_rows = name + int(name * (aug_perc / 100))
                    print("number of samples: ", n_rows)
                    basepath = 'Evaluation/EvaluationData/RealData/'

                    save_model_path = basepath + str(n_rows) + "/Classifier/" + "Baseline/Seed_" + str(seed) + "/"
                else:
                    n_rows = name
                    print("number of samples: ", n_rows)
                    save_model_path = basepath + str(n_rows) + "/Classifier/" + "Baseline/Seed_" + str(seed) + "/"

                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)

                # training classifier
                #
                epoch, valid_loss, valid_acc = training(date, params, str(n_rows), training_epochs, save_model_path)
                #
                print(valid_loss, valid_acc)
                # saving_results
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
            else:

                save_model_path = basepath + str(n_rows) + "/Classifier/" + str(aug_perc) + "/Seed_" + str(seed) + "/"
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)



                epoch, valid_loss, valid_acc = training(date, params, str(n_rows), training_epochs, save_model_path)
                print(valid_loss, valid_acc)

                # results
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



