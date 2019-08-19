#!/usr/bin/env python
# coding: utf-8
from Cut1D import *
start = time.time()
epochs = [1000, 2000, 3000, 4000, 5000]
lrs = [0.0001]
d_hids = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# epoch_accs = []
# for epoch in epochs:
# 	cut_net = train(d_hids[2], lrs[0], epoch)
# 	acc = test(cut_net)
# 	epoch_accs.append(acc)

# lr_accs = []
# for lr in lrs:
# 	cut_net = train(d_hids[2], lr, epochs[0])
# 	acc = test(cut_net)
# 	lr_accs.append(acc)

# d_hid_accs = []
# for d_hid in d_hids:
# 	cut_net = train(d_hid, lrs[0], epochs[0])
# 	acc = test(cut_net)
# 	d_hid_accs.append(acc)

#plots the accuracy given certain parameter values
def plot_acc(par, acc, label):
    plt.figure()
    plt.plot(par, acc)
    plt.xlabel(label)
    plt.ylabel('accuracy')

# np.save('epoch_accs', np.array(epoch_accs))
# np.save('d_hid_accs', np.array(d_hid_accs))

epoch_accs = np.load('epoch_accs.npy')
d_hid_accs = np.load('d_hid_accs.npy')

plot_acc(epochs, epoch_accs, "epochs")
# plot_acc(lrs, lr_accs, "learning rates")
plot_acc(d_hids, d_hid_accs, "hidden layer size")

end = time.time()
print(end - start)

plt.show()
