import numpy as np

'''EARLY STOPPING PROCEDU§RE'''
class Monitor():

    def __init__(self, patience, delta):

        self.patience = patience
        self.restore_value = patience
        self.delta = delta
        self.minimum = np.Inf
        self.stop = False
        self.save = True
        self.epoch = 0


    def __call__(self, epoch, loss):

        if self.patience == 0:
            self.stop = True

        if self.minimum + self.delta <= np.min(loss):
            self.patience -= 1
            self.save = False

        if self.minimum > np.min(loss):
            self.minimum = np.min(loss)
            self.patience = self.restore_value
            self.epoch = epoch



