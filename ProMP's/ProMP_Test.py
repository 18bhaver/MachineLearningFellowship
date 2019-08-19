#!/usr/bin/env python
# coding: utf-8

#import ProMP_Module
from ProMP_Module import *

#variables
num_rbfs = 10
bandwidth = 5e-3
num_time = 1000
num_traj = 100

#draw & store random trajectories
traj1 = make_traj(num_traj, 515615, 0)
traj2 = make_traj(num_traj, 151012, 1)
traj3 = make_traj(num_traj, 765431, 2)
traj4 = make_traj(num_traj, 111111, 5)
trajs = [traj1, traj2, traj3, traj4]
plot_traj(trajs)

#calculate weights for the rbfs, & their properties
basis_function = make_rbf(num_time, num_rbfs, bandwidth)
(W, stacked_W) = make_weights(trajs, num_rbfs, bandwidth, num_time)
weight_properties = calc_weight_props(W, stacked_W, len(trajs), num_rbfs)

#calculate the trajectories' properties
traj_properties = calc_traj_props(basis_function, weight_properties, len(trajs))

#plot the ProMPs
plot_promp(traj_properties)

#create & plot the conditioned ProMP
all_props = calc_cond_props(np.array([0.001]), basis_function, 250, [1.5, np.nan, np.nan, np.nan], weight_properties)
all_props = calc_traj_props(basis_function, all_props[0], all_props[1])
plot_promp(all_props)
