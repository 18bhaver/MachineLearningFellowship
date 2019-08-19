#!/usr/bin/env python
# coding: utf-8

#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from CutCounter_StitchMPs import * 

#variables 
d_in = 500
d_hid = 250
d_out = 5
learning_rate = 0.01
epochs = 1000
num_StitchMPs = 500
dict_primitives = {1 : [2, 4], 2 : [1, 3], 3 : [1, 3], 4 : [2, 4]}
total_time = 500
min_time_modifier = 50
num_rbfs = 10
bandwidth = 5e-3
num_traj = 100

#prepare data
data, target = make_StitchMP(num_StitchMPs, dict_primitives, total_time, min_time_modifier, num_rbfs, bandwidth, num_traj)
data, target = torch.from_numpy(data).float(), torch.from_numpy(target).long()

#CutNet module
class Net(nn.Module):
    #initialization function
    def __init__(self, d_in, d_hid, d_out):
        super(Net, self).__init__()
        self.mod1 = nn.Linear(d_in, d_hid)
        self.mod2 = nn.Linear(d_hid,d_hid)
        self.mod3 = nn.Linear(d_hid, d_out)
    #forward propagation function
    def forward(self, x):
        x = F.relu(self.mod1(x))
        x = F.relu(self.mod2(x))
        x = self.mod3(x)
        return F.log_softmax(x, dim= -1)

#create an instance of CutNet
cut_net = Net(d_in, d_hid, d_out)

#create a stochastic gradient descent optimizer
optimizer = torch.optim.SGD(cut_net.parameters(), lr = learning_rate)
#create a negative log likelihood loss function
criterion = nn.NLLLoss()

#training loop
for epoch in range(epochs):
    #run CutNet
    cut_net_out = cut_net(data)
    #calculate the loss
    loss = criterion(cut_net_out, target)
    #clear gradients to prepare for back propagation
    optimizer.zero_grad()
    #back propagation of the loss
    loss.backward()
    #update parameters 
    optimizer.step()

#test CutNet
data, target = make_StitchMP(num_StitchMPs, dict_primitives, total_time, min_time_modifier, num_rbfs, bandwidth, num_traj)
data, target = torch.from_numpy(data).float(), torch.from_numpy(target).long()

cut_net_out = cut_net(data)

correct = 0.0
for t in range(len(target)):
    max_val = cut_net_out[t].max(0)[1]
    if target[t] == max_val:
        correct += 1
print("accuracy: ")
print(correct/len(target) * 100)
