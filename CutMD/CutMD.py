#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

np.random.seed(3244232)
torch.manual_seed(3244232)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = (7,5), stride=1, padding = (2, 2, 0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,5), stride=5))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1 * 100 * 32, d_hid)
        self.fc2 = nn.Sigmoid(d_hid, 500)
    def forward(self, x):
	    out = self.layer1(x)
	    out = out.reshape(out.size(0), -1)
	    out = self.drop_out(out)
	    out = self.fc1(out)
	    out = self.fc2(out)
	    return out

def train(d_hid, learning_rate, epochs):

	#create an instance of CutNet
	cut_net = ConvNet()

	#create a stochastic gradient descent optimizer
	optimizer = torch.optim.Adam(cut_net.parameters(), lr = learning_rate)
	#create a binary cross entropy error loss function
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
	cut_net_out = cut_net(data)

	cut_net_out = cut_net_out.data.numpy()
	target = target.data.numpy()

	total_f_score = 0.0
	for a_target, a_out in zip(target,cut_net_out):
	    gt = np.where(a_target)[0] * 1.0
	    res = np.sort(np.argsort(-a_out)[:gt.shape[0]])
	    total_f_score += f_score(gt, res, 10)
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
