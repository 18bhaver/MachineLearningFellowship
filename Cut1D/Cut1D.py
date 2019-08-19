#!/usr/bin/env python
# coding: utf-8

#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from Cut1D_StitchMPs import * 
import time
import sys

np.random.seed(3244232)
torch.manual_seed(3244232)


#CutNet module
class Net(nn.Module):
    #initialization function
    def __init__(self, d_in, d_hid, d_out):
        super(Net, self).__init__()
        self.mod_in = nn.Linear(d_in, d_hid)
        self.mod_out = nn.Linear(d_hid, d_out)
    #forward propagation function
    def forward(self, x):
        x = F.relu(self.mod_in(x))
        x = torch.sigmoid(self.mod_out(x))
        return x



def train(d_hid, learning_rate, epochs):
	#variables 
	d_in = 500
	d_out = 500
	num_StitchMPs = 500
	dict_primitives = {1 : [2, 4], 2 : [1, 3], 3 : [1, 3], 4 : [2, 4]}
	total_time = 500
	min_time_modifier = 50
	num_rbfs = 10
	bandwidth = 5e-3
	num_traj = 100

	#prepare data
	# data, target = make_StitchMP(num_StitchMPs, dict_primitives, total_time, min_time_modifier, num_rbfs, bandwidth, num_traj)
	# np.save('training_data', data)
	# np.save('training_target', target)
	data = np.load('training_data.npy')
	target = np.load('training_target.npy')
	data, target = torch.from_numpy(data).float(), torch.from_numpy(target).float()

	#create an instance of CutNet
	cut_net = Net(d_in, d_hid, d_out)

	#create a stochastic gradient descent optimizer
	optimizer = torch.optim.Adam(cut_net.parameters(), lr = learning_rate)
	#create a mean squared error loss function

	# criterion = nn.MSELoss(reduction = 'mean')
	criterion = nn.BCELoss()


	#training loop

	N_minibatches = 1

	mini_batches,mini_batches_gt = create_minibatches(data, target, N_minibatches)

	bar = Bar("Training CutNet", max = epochs)

	for epoch in range(epochs):
	    bar.next()
	    #run CutNet

	    mini_batches,mini_batches_gt = create_minibatches(data, target, N_minibatches)

	    mini_batch_ids = np.random.choice(N_minibatches,N_minibatches,replace=False)

	    for it, mini_batch_id in zip(range(N_minibatches),mini_batch_ids):

	        mini_batch = mini_batches[mini_batch_id]
	        mini_batch_gt = mini_batches_gt[mini_batch_id]

	        cut_net_out = cut_net(mini_batch)
	        #calculate the loss
	        loss = criterion(cut_net_out, mini_batch_gt)
	        #clear gradients to prepare for back propagation
	        optimizer.zero_grad()
	        #back propagation of the loss
	        loss.backward()
	        #update parameters 
	        optimizer.step()
	        #print loss updates
	        #print("{}/{}({}): {}".format(epoch, it, mini_batch_id, loss))

	bar.finish()
	return cut_net

def create_minibatches(batch, batch_gt, N_minibatches):
	mini_batches = []
	mini_batches_gt = []
	mb_size = int(batch.shape[0]/N_minibatches)
	indices = np.random.choice(batch.shape[0],batch.shape[0],replace=False)
	for it in range(N_minibatches-1):
		mini_batches.append(batch[indices[it*mb_size:(it+1)*mb_size],:])
		mini_batches_gt.append(batch_gt[indices[it*mb_size:(it+1)*mb_size],:])
	mini_batches.append(batch[indices[(N_minibatches-1)*mb_size:],:])
	mini_batches_gt.append(batch_gt[indices[(N_minibatches-1)*mb_size:],:])
	return mini_batches,mini_batches_gt


def test(cut_net):
	num_StitchMPs = 500
	#test CutNet
	# data, target = make_StitchMP(num_StitchMPs, dict_primitives, total_time, min_time_modifier, num_rbfs, bandwidth, num_traj)
	# np.save('test_data', data)
	# np.save('test_target', target)
	data = np.load('test_data.npy')
	target = np.load('test_target.npy')
	data, target = torch.from_numpy(data).float(), torch.from_numpy(target).float()

	cut_net_out = cut_net(data)

	cut_net_out = cut_net_out.data.numpy()
	target = target.data.numpy()
	correct = 0.0
	all_diff = 0.0
	total_f_score = 0.0
	num_StitchMPs_wcuts = 0.0
	for a_target, a_out in zip(target,cut_net_out):
	    gt = np.where(a_target)[0] * 1.0
	    res2 = np.where(a_out > 0.5)[0]
	    res = np.sort(np.argsort(-a_out)[:gt.shape[0]])
	    correct += np.array_equal(gt, res)
	    if(gt.shape[0] is not 0):
	        all_diff += np.mean((np.abs(res - gt))/gt)
	        num_StitchMPs_wcuts += 1
	    total_f_score += f_score(gt, res, 10)

		# print("gt  :{}".format(gt))
		# print("res :{}".format(res))
		# print("    :{}\n".format(a_out[res]))
		# print("res2:{}".format(res2))
		# print("    :{}\n".format(a_out[res2]))
	# print("hit-or-miss accuracy:")
	# accuracy = correct / num_StitchMPs 
	# print(accuracy * 100)
	# print("some other accuracy we are unsure about:")
	other_acc = (1 - all_diff / num_StitchMPs_wcuts) * 100
	# print(other_acc)
	return total_f_score / num_StitchMPs

def f_score(gt, res, bandwidth):
    if gt.shape[0] == 0:
        return 1
    true_pos = 0.0
    false_pos = 0.0
    false_neg = 0.0
    for g in gt:
        dif = np.abs(res - g) - bandwidth
        num_in_range = dif[dif <= 0].shape[0]
        #true positive when at least one in range
        if num_in_range > 0:
            true_pos += 1
            #false positive when more than one in range
            if num_in_range > 1:
                false_pos += num_in_range - 1
        #false negative when none in range
        else:
            false_neg += 1    

    precision = 1 
    recall = 1
    if (true_pos + false_pos) != 0.0:
        precision = true_pos / (true_pos + false_pos)

    if (true_pos + false_neg) != 0.0:
  	    recall = true_pos / (true_pos + false_neg)

    return 2 * (precision * recall) / (precision + recall)
