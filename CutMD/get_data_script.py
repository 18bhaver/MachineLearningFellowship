from get_data import *
# save_data("/projects/fri_rl/mov_seg/three", 2000, "three")
# norm_trajs = np.load("norm_trajs_three.npy")
# norm_end_poses = np.load("norm_end_poses_three.npy")

# plot_one_dim(norm_end_poses, 2)
# plot_end_pos(norm_end_poses)
# plot_bags(norm_trajs)

end_poses = extract_end_pose("/projects/fri_rl/mov_seg/three")
times = find_time_stamps("/projects/fri_rl/mov_seg/three")
pad_end_poses = pad(end_poses, np.max(times))
print(np.max(times))
plot_one_dim(pad_end_poses, 2)
