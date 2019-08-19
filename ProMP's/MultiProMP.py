#!/usr/bin/env python
# coding: utf-8

#imports
import matplotlib.pylab as plt
import numpy as np
import scipy.linalg as sp

#function definitions 

#create random trajectories
def make_traj(num_trajectories, seed, switch):
    #use random to create variability for the time stamps, amplitude, & offset
    np.random.seed(seed)
    T_trajs = np.random.randint(800, 1200, num_trajectories)
    A_trajs = np.random.normal(1, 0.1, num_trajectories)
    O_trajs = np.random.normal(0, 0.1, num_trajectories)
    #create data points for the trajectories, choosing the distribution based on the switch
    if switch is 0:
        traj = [np.expand_dims(np.sin(np.linspace(0, 2 * np.pi, T)), axis = 0) * A + O for T, A, O in zip(T_trajs, A_trajs, O_trajs)]
    elif switch is 1:
        traj = [np.expand_dims(np.cos(np.linspace(0, 2 * np.pi, T)), axis = 0) * A + O for T, A, O in zip(T_trajs, A_trajs, O_trajs)]
    elif switch is 2:
        traj = [np.expand_dims(-np.cos(np.linspace(0, 2 * np.pi, T)), axis = 0) * A + O for T, A, O in zip(T_trajs, A_trajs, O_trajs)]
    elif switch is 3:
        traj = [np.expand_dims(np.arctan(np.linspace(-np.pi, np.pi, T)), axis = 0) * A + O for T, A, O in zip(T_trajs, A_trajs, O_trajs)]
    elif switch is 4:
        traj = [np.expand_dims(-np.arctan(np.linspace(-np.pi, np.pi, T)), axis = 0) * A + O for T, A, O in zip(T_trajs, A_trajs, O_trajs)]
    else:
        traj = [np.expand_dims(-np.sin(np.linspace(0, 2 * np.pi, T)), axis = 0) * A + O for T, A, O in zip(T_trajs, A_trajs, O_trajs)]
    #return trajectories
    return traj

#plot trajectories
def plot_traj(trajectories):
    for traj in trajectories:
        plt.figure()
        for t in traj:
            plt.plot(t[0, :])
    plt.show()

#create radial basis functions
def make_rbf(num_time_stamps, num_basis_functions, bandwidth):
    #define phase
    phase = np.linspace(0, 1, num_time_stamps)
    #implement rbf equation 
    centers = np.linspace(-bandwidth * 2, 1 + bandwidth * 2, num_basis_functions)
    bases = np.exp([-(phase - c) ** 2 / (2 * bandwidth) for c in centers])
    #normalize rbfs
    bases /= bases.sum(axis = 0)
    #return rbfs
    return bases

#create weights for each trajectory
def make_weights(trajectories, num_basis_functions, bandwidth, num_time_stamps):
    W = []
    stacked_W = np.zeros((len(trajectories[0]), len(trajectories) * num_basis_functions))
    i = 0
    #for each trajectory in each dimension
    for i,dim in enumerate(trajectories):
        weights = []
        for t in dim:
            #define psi
            psi = make_rbf(t.shape[1], num_basis_functions, bandwidth).T
            #implement weights equation 
            w = np.linalg.solve(np.dot(psi.T, psi) + 1e-10 * np.eye(psi.shape[1]), np.dot(psi.T, t.T))
            weights.append(w)
        weights = np.array(weights)
        #remove extra dimension
        weights = np.reshape(weights, (weights.shape[0], weights.shape[1]))
        W.append(weights)
        #stack weights on top of each other
        stacked_W[:,i*(weights.shape[1]):(i+1) * weights.shape[1]] = weights
        i += 1
    W = np.array(W)
    return (W, stacked_W)

#plot weights
def plot_weights(weights):
    for w in weights:
        plt.figure()
        plt.plot(w.T)
        plt.show()

#calculate properties for weights
def calc_weight_props(weights, stacked_weights, num_dim, num_basis_functions):
    mean_w = []
    std_w = []
    #calculate means
    for i in range(num_dim):
        mean_w.append(np.mean(weights[i], axis = 0))
    mean_w = np.array(mean_w).T
    #calculate covariance
    cov_w = np.cov(stacked_weights.T)
    #calculate standard deviations
    for i in range(num_dim):
        cur = cov_w[i * num_basis_functions:(i+1)* num_basis_functions,i * num_basis_functions:(i+1) * num_basis_functions]
        std_w.append(np.sqrt(np.diag(cur)))
    std_w = np.array(std_w).T
    #good job Fortran!
    mean_w = np.ndarray.flatten(mean_w, 'F')
    #return weight properties
    return np.array((mean_w, cov_w, std_w))

#calculate properties for trajectories
def calc_traj_props(basis_functions, weight_properties, num_dim):
    all_props = []
    num_rbfs = basis_functions.shape[0]
    num_time = basis_functions.shape[1]
    block_basis = make_block(basis_functions, num_dim)
    for dim in range(num_dim):
        #calculate mean, covariance, & standard deviation of trajectories 
        mean_traj = np.dot(basis_functions.T, weight_properties[0][dim * num_rbfs:(dim+1) * num_rbfs])
        cov_traj = np.dot(np.dot(block_basis.T, weight_properties[1]), block_basis) + 1e-6 * np.eye(block_basis.shape[1])        
        cur_cov = cov_traj[dim * num_time:(dim+1)* num_time,dim * num_time:(dim+1) * num_time]
        std_traj = np.sqrt(np.diag(cur_cov))
        all_props.append([mean_traj, cov_traj, std_traj])
    return all_props

#calculate block diagonal
def make_block(basis_functions, num_dim):
    return (sp.block_diag(*[basis_functions] * num_dim))

#plot promp
def plot_promp(all_properties):
    for dim in range(len(all_properties)):
        properties = all_properties[dim]
        plt.figure()
        top = np.ndarray.flatten(properties[0] - 2 * properties[2])
        bottom = np.ndarray.flatten(properties[0] + 2 * properties[2])
        plt.fill_between(np.arange(0, properties[0].shape[0]), top, bottom, facecolor = 'blue', alpha = 0.3, edgecolor = 'red')
        plt.plot(properties[0])
    plt.show()

#condition promp at given time & position
def calc_cond_props(observation_strength, basis_functions, des_t, des_pos, properties):
    #isolate the dimensions to be conditioned
    dims = np.where(~np.isnan(des_pos))[0]
    des_y = []
    for d in dims:
        des_y.append(des_pos[d])
    des_y = np.array(des_y)
    mean_w = properties[0]
    cov_w = properties[1]
    #make block diagonal then extract desired time stamp
    block_basis = make_block(basis_functions, len(des_pos)) 
    psi = []
    for d in dims:
        psi.append(block_basis[:, d * basis_functions.shape[1] + des_t])
    psi = np.array(psi).T
    mean_w = np.ndarray.flatten(mean_w)
    #implement conditioning equation 
    L = np.linalg.multi_dot([cov_w, psi, np.linalg.inv(observation_strength + np.linalg.multi_dot([psi.T, cov_w, psi]))]) 
    new_mean_w = mean_w + np.dot(L, (des_y - np.dot(psi.T, mean_w)))
    new_cov_w = cov_w - np.linalg.multi_dot([L, psi.T, cov_w])
    new_std_w = np.sqrt(np.diag(new_cov_w))
    #return conditioning properties
    return ((new_mean_w, new_cov_w, new_std_w), len(des_pos))
