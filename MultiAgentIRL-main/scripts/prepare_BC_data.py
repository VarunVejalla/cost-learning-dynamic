import os
import h5py
import numpy as np
import pickle

# JLD2 file path
dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, "../data/data_bc.jld2")

# Open the JLD2 file in read mode
file = h5py.File(file_path, "r")

# Access the variables in the JLD2 file
# For example, if you have a variable named 'data'


trajectories_agent1 = []
trajectories_agent2 = []
trajectories_agent3 = []

for i in range(file['x_trajectories'].shape[0]):
    traj = file['x_trajectories'][i]
    ref = file['x_refs_bc']
    traj_np = np.array(file[traj])
    traj_np = traj_np.T
    ref_np = np.array(ref)
    ref_np = ref_np.T 

    # Split the array into three Tx4 arrays
    array1 = traj_np[:, :4].reshape(-1, 4)
    array2 = traj_np[:, 4:8].reshape(-1, 4)
    array3 = traj_np[:, 8:].reshape(-1, 4)

    # Append a new column with one-hot vector to each array
    num_arrays = 3
    array1 = np.concatenate((array1, np.array(np.eye(num_arrays)[0] * np.ones((array1.shape[0], 1))),ref_np[:traj_np.shape[0], :2]), axis=1)
    # array1 = np.concatenate((array1, ref_np[:2]), axis=1)
    array2 = np.concatenate((array2, np.array(np.eye(num_arrays)[1] * np.ones((array2.shape[0], 1))), ref_np[:traj_np.shape[0], 4:6]), axis=1)
    array3 = np.concatenate((array3, np.array(np.eye(num_arrays)[2] * np.ones((array3.shape[0], 1))),ref_np[:traj_np.shape[0], 8:10]), axis=1)

    trajectories_agent1.append(array1)
    trajectories_agent2.append(array2)
    trajectories_agent3.append(array3)

file.close()

import pickle

# File path to save the pickle file
file_path = "data/BC_data.pkl"

# Create a dictionary to store the variables
data = {
    'agent1': trajectories_agent1,
    'agent2': trajectories_agent2,
    'agent3': trajectories_agent3
}

# Save variables to pickle file
with open(file_path, "wb") as file:
    pickle.dump(data, file)
