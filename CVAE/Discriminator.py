import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_size, h1_size, output_size, dropout):
        super().__init__()

        ############### hidden_size #############

        self.h1_size = h1_size
        self.h2_size = output_size
        self.input_size = input_size

        #########################################

        self.fc1 = nn.Linear(input_size, h1_size)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h1_size, output_size)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(output_size, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
################################## script #######################################

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

z = torch.randn(32,10)
y = torch.zeros(32,1)

'''



import matplotlib.pyplot as plt

X1 = torch.randn(1000, 50)
X2 = torch.randn(1000, 50) + 1.5
X = torch.cat([X1, X2], dim=0)
Y1 = torch.zeros(1000, 1)
Y2 = torch.ones(1000, 1)
Y = torch.cat([Y1, Y2], dim=0)
print(X.size())
print(Y.size())

plt.scatter(X1[:, 0], X1[:, 1], color='b')
plt.scatter(X2[:, 0], X2[:, 1], color='r')
plt.show()

net = Discriminator(input_size=50, h1_size=50, output_size=100, dropout=0.2)
opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.BCELoss()

def train_epoch(model, opt, criterion, batch_size=50):
    model.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = net(x_batch)
        print("prediction: ", y_hat)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
    return losses


e_losses = []
num_epochs = 1
for e in range(num_epochs):
    e_losses += train_epoch(net, opt, criterion)
plt.plot(e_losses)
plt.show()
'''