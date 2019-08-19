#!/usr/bin/env python
# coding: utf-8

#imports
import matplotlib.pylab as plt
import numpy as np

#constants
num_rbfs = 10
bandwidth = 5e-3
num_time = 1000
num_traj = 100

#function definitions 

#create random trajectories
def make_traj(num_trajectories):
    #use random to create variability
    np.random.seed(11)
    T_trajs = np.random.randint(800, 1200, num_trajectories)
    A_trajs = np.random.normal(1, 0.1, num_trajectories)
    O_trajs = np.random.normal(0, 0.1, num_trajectories)
    
    #create data points for the trajectories using random number of time stamps, amplitude, & origin
    traj = [np.expand_dims(np.sin(np.linspace(0, 2 * np.pi, T)), axis = 0) * A + O for T, A, O in zip(T_trajs, A_trajs, O_trajs)]

    #return trajectories
    return traj

#plot trajectories
def plot_traj(trajectories):
    plt.figure()
    for t in trajectories:
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

#plot radial basis functions
def plot_rbf(basis_functions):
    plt.figure()
    plt.plot(basis_functions.T)
    plt.show()
    
#create weights to approximate trajectories
def make_weights(trajectories, num_basis_functions, bandwidth):
    weights = []
    for t in trajectories:
        #define phi: used to be t.shape[1], for old data
        phi = make_rbf(t.shape[0], num_basis_functions, bandwidth).T
        #implement weights equation
        w = np.linalg.solve(np.dot(phi.T, phi) + 1e-10 * np.eye(phi.shape[1]), np.dot(phi.T, t.T))
        #used to be w[:,0], for old data
        weights.append(w)
    weights = np.array(weights)

    #return weights
    return weights

#plot weights
def plot_weights(weights):
    plt.figure()
    plt.plot(weights.T)
    plt.show()

#calculate properties for weights
def calc_weight_props(weights):
    #calculate mean, covariance, & standard deviation of weights
    mean_w = np.mean(weights, axis = 0)
    cov_w = np.cov(weights.T)
    std_w = np.sqrt(np.diag(cov_w))

    #return weight properties
    return (mean_w, cov_w, std_w)

#calculate properties for trajectories
def calc_traj_props(basis_functions, weight_properties):
    #calculate mean, covariance, & standard deviation of trajectories
    mean_traj = np.dot(basis_functions.T, weight_properties[0])
    cov_traj = np.dot(np.dot(basis_functions.T, weight_properties[1]), basis_functions) + 1e-10 * np.eye(basis_functions.shape[1])
    std_traj = np.sqrt(np.diag(cov_traj))

    #return trajectory properties
    return (mean_traj, cov_traj, std_traj)

#plot promp
def plot_promp(properties):
    plt.figure()
    plt.fill_between(np.arange(0, properties[0].shape[0]), properties[0] - 2 * properties[2], properties[0] + 2 * properties[2], facecolor = 'blue', alpha = 0.1, edgecolor = 'red')
    plt.plot(properties[0])
    plt.show()

#plot sample mps on a promp
def plot_sample_mp(properties, num):
    plt.figure()
    sampled_trajectories = np.random.multivariate_normal(properties[0], properties[1], num)
    plt.plot(sampled_trajectories.T)
    plt.show()

#conditioning
def condition(observation_strength, basis_functions, des_t, des_pos, properties):
    mean_w = properties[0]
    cov_w = properties[1]
    psi = basis_functions[:,des_t]
    psi = np.array([psi]).T
    con1 = np.dot(cov_w, psi)
    con2 = observation_strength + np.linalg.multi_dot([psi.T, cov_w, psi])
    con3 = des_pos - np.dot(psi.T, mean_w)
    new_mean_w = mean_w + np.linalg.multi_dot([con1, np.linalg.inv(con2), con3])

    cov1 = np.dot(cov_w, psi)
    cov2 = con2
    cov3 = np.dot(psi.T, cov_w)
    new_cov = cov_w - np.linalg.multi_dot([cov1, np.linalg.inv(cov2), cov3])
    new_std = np.sqrt(np.diag(new_cov))
    return (new_mean_w, new_cov, new_std)

# #sample data

# #draw & store random trajectories
# traj = make_traj(num_traj)
# plot_traj(traj)

# #radial basis functions

# #draw & store rbfs
# rbfs = make_rbf(num_time, num_rbfs, bandwidth)
# plot_rbf(rbfs)

# #weights

# #draw & store weights
# w = make_weights(traj, num_rbfs, bandwidth)
# plot_weights(w)

# #weight promp

# #calculate weight properties & draw weight promp
# w_props = calc_weight_props(w)
# plot_promp(w_props)

# #promp

# #create & draw promp
# traj_props = calc_traj_props(rbfs, w_props)
# plot_promp(traj_props)

# #promp with samples

# #create & draw samples of promp
# plot_sample_mp(traj_props, 100)

# conditioned_properties = condition(0.001, rbfs, -1, .1, w_props)
# plot_promp(conditioned_properties)
