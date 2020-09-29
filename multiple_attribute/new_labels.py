import pandas as pd
import operator
import numpy as np
import spacy
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F

def multiple_binary_cross_entropy(output, target, attr_numb = 4):

    bce_loss = 0
    for i in range(attr_numb):
        pred_attribute = torch.sigmoid(output[:, i])
        target_attribute = target[:,i]
        bce_loss += F.binary_cross_entropy(pred_attribute, target_attribute)
    bce_loss = bce_loss/attr_numb
    return bce_loss

def multiple_accuracy_attributes(y_pred, y_true, attr_numb = 4):
    '''

    Method to compute binary cross entropy on each of the attributes for the multiclassification task of the decoder

    :param y_pred: predictions logits [batch_size * attr_size]
    :param y_true: labels [batch_size * attr_size]
    :return:
    '''
    y_true = y_true.detach().cpu().numpy()
    accuracy = 0
    # Compute discriminator accuracy
    for i in range(attr_numb):
        pred_prob = torch.sigmoid(y_pred[:, i])
        pred_prob = pred_prob.detach().cpu().numpy()
        pred_prob = pred_prob > 0.5
        predictions = [1 if x else 0 for x in pred_prob]
        truth = y_true[:,i]
        attr_accuracy = accuracy_score(truth, predictions)
        accuracy += attr_accuracy
    accuracy = accuracy/attr_numb
    return accuracy

