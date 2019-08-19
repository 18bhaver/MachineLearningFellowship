from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pylab as plt
import glob
import rosbag
from scipy import interpolate

def extract_trajs(directory_name):
	trajectories = []
	file_names = glob.glob(directory_name + '/*')
	for file_name in file_names:
		bag = rosbag.Bag(file_name)
		traj = []
		for topic, msg, t in bag.read_messages(topics = ['/alexei/joint/states']):
			pos = list(msg.position)
			if len(pos) > 1:
				traj.append(pos[1:8])
		traj = np.array(traj).T
		trajectories.append(traj)
	return trajectories

def extract_end_pose(directory_name):
    search = directory_name + "/*"
    file_names = glob.glob(search)
    end_poses = []
    for file_name in file_names:
        bag = rosbag.Bag(file_name)
        end_pose = []
        for topic, msg, t in bag.read_messages(topics = ['/alexei/end_effector/states']):
            end_pose.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        end_poses.append(np.array(end_pose).T)
    return end_poses

def plot_bags(trajectories):
	for traj in trajectories:
		plt.figure()
		for t in traj:
			plt.plot(t)
		plt.show()

def plot_end_pos(end_poses):
	for end_pos in end_poses:
		plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot(end_pos[0], end_pos[1], end_pos[2], '-b')
		plt.show()

def plot_one_dim(end_poses, dim):
	for end_pos in end_poses:
		plt.figure()
		plt.plot(end_pos[dim])
		plt.show()

def plot_all_dim(end_poses):
	for i in range(3):
		plot_one_dim(end_poses, i)


def find_time_stamps(directory_name):
	times = []
	file_names = glob.glob(directory_name + '/*')
	for file_name in file_names:
		bag = rosbag.Bag(file_name)
		traj = []
		time = 0
		for topic, msg, t in bag.read_messages(topics = ['/alexei/joint/states']):
			time += 1
		times.append(time)
	return times

def normalize(data, des_time):
	all_data = []
	for traj in data:
		datas = []
		for dim in traj:
			time = np.linspace(0, des_time, dim.shape[0])
			func = interpolate.interp1d(time, dim)
			new_time = np.linspace(0, des_time, des_time)
			new_data = func(new_time)
			datas.append(new_data)
		datas = np.array(datas)
		all_data.append(datas)
	return np.array(all_data)

def save_data(file_in, des_time, file_out):
	trajs = extract_trajs(file_in)
	end_poses = extract_end_pose(file_in)
	norm_trajs = normalize(trajs, des_time)
	norm_end_poses = normalize(end_poses, des_time)
	np.save("norm_trajs_" + file_out, norm_trajs)
	np.save("norm_trajs_" + file_out, norm_end_poses)

def pad(data, des_time):
	all_data = []
	times = []
	for traj in data:
		times.append(traj.shape[1])
		num_pad = des_time - traj.shape[1]
		datas = np.pad(traj, [(0,0), (0, num_pad)], mode = 'constant')
		all_data.append(datas)
	print(np.max(np.array(times)))
	return np.array(all_data)
