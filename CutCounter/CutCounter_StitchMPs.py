#!/usr/bin/env python
# coding: utf-8

#import ProMP_Module
from SingleProMP import * 

#function declarations

#create randomized sigmoids
def sigmoid(x, switch, start = None, end = None):
	#define random values for start, end, & steepness
	if start is None:
		start = np.random.uniform(-0.2, 0.2, x.shape[0])
	if end is None:
		end = np.random.uniform(0.8, 1.2, x.shape[0])    
	scale = end - start
	steep = np.random.uniform(0.5, 0.6, x.shape[0])
	#return modified sigmoid, scaled to 0 & 1 for smooth concatenations
	sig_basis = 1 / (1 + np.exp(-steep * x.T))
	sig_basis -= np.min(sig_basis, axis = 0)
	sig_basis /= np.max(sig_basis, axis = 0)
	sig_basis = sig_basis * scale + start
	return sig_basis if switch == 1 else -sig_basis
	
#create randomized guassian-like double sigmoids
def double_sigmoid(x, switch):
	# define random values for start, middle, & end
	start = np.random.uniform(-0.2, 0.2,x.shape[0])
	middle = np.random.uniform(0.5, 1.5, x.shape[0])
	end = np.random.uniform(-0.2, 0.2, x.shape[0])    
	#create the each half of the double sigmoid & return them combined
	first = sigmoid(x, 1, start, middle)
	second = sigmoid(x, 1, middle, end)
	if(switch == 3):
		return np.concatenate((first, second), axis = 0)
	else:
		return np.concatenate((-first, -second), axis = 0)

#create multiple instances of sigmoid data
def generate_data(num_trajectories, num_time_stamps, switch):
	np.random.seed()
	if(switch == 1 or switch == 2):
		x = np.linspace(-10, 10, num_time_stamps)
	else:
		x = np.linspace(-10, 10, num_time_stamps / 2)
	x_list = []
	for i in range(num_trajectories):
		x_list.append(x)
	x_list = np.array(x_list)
	#use switch to decide on single or double sigmoid data
	if(switch == 1 or switch == 2):
		data = sigmoid(x_list, switch)
	else:
		data = double_sigmoid(x_list, switch)
	return data

#plot sigmoid data
def plot_data(data):
	for dat in data: 
		plt.figure()
		plt.plot(dat)
	plt.show()

#makes stitchMPs
def make_StitchMP(num_StitchMPs, dict_primitives, total_time, min_time_modifier, num_rbfs, bandwidth, num_traj):
	all_stitchMP = []
	all_num_cuts = []
	for i in range(num_StitchMPs):
		#decide on amount & type of primitives 
		np.random.seed()
		num_primitives = np.random.randint(1, 6)
		list_primitives = []
		for i in range(num_primitives):
			if(i == 0):
				list_primitives.append(np.random.randint(1, 5))
			else:
				choose = np.random.randint(0, 2)
				if(choose == 0):
					list_primitives.append(dict_primitives[list_primitives[i-1]][0])
				else:
					list_primitives.append(dict_primitives[list_primitives[i-1]][1])	
		#decide on the lengths of the primitives
		min_time = total_time / num_primitives - min_time_modifier
		max_time = total_time / num_primitives
		stitch_times = []
		sum_time = 0
		for i in range(num_primitives):
			cur_time = np.random.randint(min_time, max_time + 1)
			sum_time += cur_time
			stitch_times.append(cur_time)
		#distribute leftover time to make the sum of stitch_times equal to total_time
		left_time = total_time - sum_time
		sum_time = 0
		for i in range(num_primitives):
			stitch_times[i] += left_time / num_primitives
			sum_time += stitch_times[i]
		stitch_times[0] += total_time - sum_time
		#for each primitive, sample a trajectory from it's ProMP & reproject it to the new amount of time stamps
		stitchMP = []
		stitchMP_cuts = []
		cur_time = 0
		for i, primitive in enumerate(list_primitives):
			primitive_data = list(generate_data(num_traj, total_time, primitive).T)
			rbf = make_rbf(stitch_times[i], num_rbfs, bandwidth)
			w = make_weights(primitive_data, num_rbfs, bandwidth)
			w_props = calc_weight_props(w)
			if(primitive == 1):
				conditioned_properties_start = condition(0.00001, rbf, 0, 0, w_props)
				conditioned_properties_end = condition(0.00001, rbf, stitch_times[i] - 1, 1, conditioned_properties_start)
				traj_props = calc_traj_props(rbf, conditioned_properties_end)
			if(primitive == 2):
				conditioned_properties_start = condition(0.00001, rbf, 0, 1, w_props)
				conditioned_properties_end = condition(0.00001, rbf, stitch_times[i] - 1, 0, conditioned_properties_start)
				traj_props = calc_traj_props(rbf, conditioned_properties_end)
			if(primitive == 3):
				conditioned_properties_start = condition(0.00001, rbf, 0, 0, w_props)
				conditioned_properties_end = condition(0.00001, rbf, stitch_times[i] - 1, 0, conditioned_properties_start)
				traj_props = calc_traj_props(rbf, conditioned_properties_end)
			if(primitive == 4):
				conditioned_properties_start = condition(0.00001, rbf, 0, 1, w_props)
				conditioned_properties_end = condition(0.00001, rbf, stitch_times[i] - 1, 1, conditioned_properties_start)
				traj_props = calc_traj_props(rbf, conditioned_properties_end)
			sample = np.random.multivariate_normal(traj_props[0], traj_props[1], 1).T
			#stitching together primitives
			for data in sample:
				stitchMP.append(data)
			stitchMP_cuts.append(stitch_times[i] + cur_time)
			cur_time += stitch_times[i]
		#removes the cut at the last time step
		stitchMP_cuts.remove(stitchMP_cuts[len(stitchMP_cuts) - 1])
		stitchMP_cutNum = len(stitchMP_cuts)
		all_stitchMP.append(stitchMP)
		all_num_cuts.append(stitchMP_cutNum)
	all_stitchMP = np.array(all_stitchMP)
	all_stitchMP = all_stitchMP.reshape((all_stitchMP.shape[0], all_stitchMP.shape[1]))
	return(np.array(all_stitchMP), np.array(all_num_cuts))
