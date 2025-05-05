import h5py
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def plot(traj0):
    # Assuming traj0 is your (61, 20) numpy array
    # Extract agent 1 and agent 2 positions over time
    agent1_positions = traj0[:, 0:2]  # shape (61, 2)
    agent2_positions = traj0[:, 4:6]  # shape (61, 2)
    xref1_positions = traj0[:, 8:10]  # shape (61, 2)
    xref2_positions = traj0[:, 12:14]  # shape (61, 2)

    # Plot the trajectories
    plt.figure(figsize=(8, 6))
    plt.plot(agent1_positions[:, 0], agent1_positions[:, 1], label='Agent 1', marker='o')
    plt.plot(agent2_positions[:, 0], agent2_positions[:, 1], label='Agent 2', marker='x')
    plt.plot(xref1_positions[:, 0], xref1_positions[:, 1], label='X ref', marker='.')
    plt.plot(xref2_positions[:, 0], xref2_positions[:, 1], label='X ref', marker='.')

    # Mark start and end points
    plt.scatter(agent1_positions[0, 0], agent1_positions[0, 1], color='blue', label='Agent 1 Start', s=100, marker='o')
    plt.scatter(agent1_positions[-1, 0], agent1_positions[-1, 1], color='blue', label='Agent 1 End', s=100, marker='*')
    plt.scatter(agent2_positions[0, 0], agent2_positions[0, 1], color='orange', label='Agent 2 Start', s=100, marker='x')
    plt.scatter(agent2_positions[-1, 0], agent2_positions[-1, 1], color='orange', label='Agent 2 End', s=100, marker='*')
    plt.scatter(xref1_positions[0, 0], xref1_positions[0, 1], color='green', label='xref Start', s=100, marker='.')
    plt.scatter(xref1_positions[-1, 0], xref1_positions[-1, 1], color='green', label='xref End', s=100, marker='*')
    plt.scatter(xref2_positions[0, 0], xref2_positions[0, 1], color='red', label='xref Start', s=100, marker='.')
    plt.scatter(xref2_positions[-1, 0], xref2_positions[-1, 1], color='red', label='xref End', s=100, marker='*')

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Trajectories of Agent 1 and Agent 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Equal scaling for x and y
    plt.show()

# filename = "MultiAgentIRL-main\\cioc_data\\twoplayer.h5"
filename = "MultiAgentIRL-main/cioc_data/twoplayer.h5"

# 0-7 is the trajectory data

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    relevant = np.array(f.get("demo_data"))
    print(relevant.shape)
    
    traj0 = relevant[:, :, 2]
    traj0 = traj0.T
    plot(traj0)