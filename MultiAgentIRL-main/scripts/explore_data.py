import pickle
from termcolor import cprint
import numpy as np

with open('data/BC_data.pkl', 'rb') as f:
    data = pickle.load(f)
    cprint('Loaded data from BC_data.pkl', 'green')
data_list = []
for key in data.keys():
    for samples in data[key]:
        for sample in samples:
            data_list.append(sample)
data = np.asarray(data_list)

velocities = data[:, 2]
# remove velocity axes from data
features = data[:, [0, 1, 3, 4, 5, 6, 7, 8]]

print('Velocities min: {}'.format(np.min(velocities)))
print('Velocities max: {}'.format(np.max(velocities)))

vmin, vmax = 0.0, 2.3
vel_bins = np.linspace(0.0, 2.3, 100)
vel_bin_count_dict = {vel_bin: 0 for vel_bin in vel_bins}

for velocity in velocities:
    # find closest bin
    vel_bin = vel_bins[np.argmin(np.abs(vel_bins-velocity))]
    vel_bin_count_dict[vel_bin] += 1

weights = []
for velocity in velocities:
    # find closest bin
    vel_bin = vel_bins[np.argmin(np.abs(vel_bins-velocity))]
    weights.append(1.0 / vel_bin_count_dict[vel_bin])

pickle.dump(weights, open('data/sp_weights.pkl', 'wb'))

vel_mean, vel_std = np.mean(velocities), np.std(velocities)
print('Velocities mean: {}'.format(vel_mean))
print('Velocities std: {}'.format(vel_std))
feat_mean, feat_std = np.mean(features, axis=0), np.std(features, axis=0)
print('Features mean: {}'.format(feat_mean))
print('Features std: {}'.format(feat_std))

print('feat_mean: {}'.format(feat_mean.shape))
print('feat_std: {}'.format(feat_std.shape))
print('vel_mean: {}'.format(vel_mean))
print('vel_std: {}'.format(vel_std))

# save mean and std
mean_std_dict = {'vel_mean': vel_mean, 'vel_std': vel_std, 'feat_mean': feat_mean, 'feat_std': feat_std}
pickle.dump(mean_std_dict, open('data/sp_mean_std.pkl', 'wb'))


# # plot histogram
# import matplotlib.pyplot as plt
# plt.hist(velocities, bins=100)
# plt.savefig('velocities.png')