import os
import numpy as np
import torch

from utils import *

from nuscenes.nuscenes import NuScenes

DATA = os.environ.get('DATA', '/data')
# data_path = os.join(DATA, 'nuscenes')  # e.g. /data/nuscenes
# version = 'v1.0-trainval'

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from tqdm import tqdm

state_dim = 4 # 2 for position, 2 for velocity
action_dim = 2 # both acceleration

def get_agent_trajectory(nusc, scene, agent_name="ego"):
    sample_token = scene['first_sample_token']
    x_states = []
    times = []
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        
        # Ego car position in world frame
        pos = np.array(ego_pose['translation'][:2])
        rot = Quaternion(ego_pose['rotation'])
        
        # Velocity approximation
        if len(times) >= 2:
            dt = (ego_pose['timestamp'] - times[-1]) / 1e6  # convert from microsec to sec
            vel = (pos - x_states[-1][:2]) / dt
        else:
            vel = np.array([0., 0.])
        
        x_state = np.hstack((pos, vel))
        x_states.append(x_state)
        times.append(ego_pose['timestamp'])
        
        sample_token = sample['next']
    
    x_states = np.array(x_states)  # [T, 4]
    times = np.array(times)
    
    # Approximate control inputs via finite difference on velocity
    dt = np.diff(times) / 1e6
    acc = np.diff(x_states[:, 2:4], axis=0) / dt[:, None]
    
    return x_states[:-1], acc, times[:-1]  # [T-1, 4], [T-1, 2], [T-1]

def get_feature_counts(x_trajectories, u_trajectories, cost_functions):
    # TODO: should this accept as input reference trajectories?
    # cost functions could maybe take in that reference as well
    
    # x_trajectories shape = (num_trajectories, steps_in_trajectory+1, whole state_dim)
    # u_trajectories shape = (num_trajectories, steps_in_trajectory, whole action_dim)
    
    # cost_function is list of functions
    # each function takes in as input state + acceleration
    # i.e. input should be two inputs, one of shape (num_agents, state_dim), other (num_agents, action_dim)
    
    # output: for each num_trajectories, for each cost function, average value of cost_function
    
    print("in feature counts")
    
    # print(x_trajectories.shape)
    # print(u_trajectories.shape)
    
    
    num_trajectories = len(x_trajectories)
    num_features = len(cost_functions)
    
    feature_counts = np.zeros((num_features, num_trajectories))
    for i in range(num_trajectories):
        xtraj = x_trajectories[i]
        utraj = u_trajectories[i]
        steps = utraj.shape[0]
        for t in range(steps+1):
            state = xtraj[t]
            
            if t < steps:
                control = utraj[t]
            else:
                control = None
            for k in range(num_features):
                feature_counts[k, i] += cost_functions[k](state, control)
            
    avg_feature_counts = np.sum(feature_counts, axis=1) / num_trajectories
    return avg_feature_counts, feature_counts
            

# e.g. 
# state space for a single agent is 5
# 6 agents
# action space for a single agent is 7

# dynamics: 5 x 7 --> 5
# cost function: 30 x 7 --> 1
# agent_to_functions: array of arrays of agent indices (disjoint)
# num_agents

x_dims = [4,4]
x_dim = 8

u_dims = [2,2]
u_dim = 4

def ma_irl(dynamics, cost_functions, x_trajectories, u_trajectories, num_max_iter, agent_to_functions, num_agents):
    # x_trajectories shape = (num_trajectories, steps_in_trajectory+1, whole state_dim)
    # u_trajectories shape = (num_trajectories, steps_in_trajectory, whole action_dim)
    
    # cost_function is list of functions
    # each function takes in as input state + acceleration
    # i.e. input should be two inputs, one of shape (num_agents, state_dim), other (num_agents, action_dim)
    # agent_to_functions is e.g. [[0,1,2,3],[4,5,6,7]]. First agent has functions 0,1,2,3.
    
    # output: for each num_trajectories, for each cost function, average value of cost_function
    
    avg_dem_feature_counts, feature_counts = get_feature_counts(x_trajectories, u_trajectories, cost_functions)
    
    w = torch.rand(6)
    x_init = torch.Tensor([1,1.1,0.1,0.1, 0,0,0.5,0.5])
    gamma = 0.01
    num_iter = 0
    
    current_cost_funcs = []
    for i in range(num_agents):
        current_cost_funcs.append(lambda state, action, i=i: sum(w[j] * cost_functions[j](state, action) for j in agent_to_functions[i]))
    
    while num_iter < num_max_iter:
        # TODO: update gamma 
        
        sim_param = SimulationParams(10,-1,10)
        nl_game = NonlinearGame(dynamics, current_cost_funcs, x_dims, x_dim, u_dims, u_dim, 2)
        
        results, x_data, u_data = generate_simulations(sim_param, nl_game, x_init, 10, 2)
        x_trajectories_sim = results.x_trajs
        u_trajectories_sim = results.u_trajs
        
        avg_sim_feature_counts, _ = get_feature_counts(x_trajectories_sim, u_trajectories_sim, cost_functions)
        # relevant_indices = agent_to_functions[agent]
        # print(relevant_indices)
        w -= (gamma * (avg_dem_feature_counts - avg_sim_feature_counts))
        current_cost_funcs = []
        for i in range(num_agents):
            current_cost_funcs.append(lambda state, action, i=i: sum(w[j] * cost_functions[j](state, action) for j in agent_to_functions[i]))
    
        num_iter += 1
        
    return w

def cost_func1(state, action):
    pos_p0 = state[0:2]  # x0, y0
    goal_p0 = torch.tensor([1.0, 1.0])
    dist_to_goal = torch.linalg.norm(pos_p0 - goal_p0)

    return dist_to_goal

def cost_func2(state, action):
    pos_p0 = state[0:2]  # x0, y0
    pos_p1 = state[4:6]  # x1, y1
    dist_to_other = torch.linalg.norm(pos_p0 - pos_p1)

    return 1.0 / (dist_to_other + 1e-6)

def cost_func3(state, action):
    if action is None:
        return 0
    act_p0 = action[0:2]  # ax0, ay0
    eps = 1e-6
    action_cost = torch.sqrt(torch.sum(act_p0 ** 2) + eps)

    return action_cost

def cost_func4(state, action):
    pos_p1 = state[4:6]  # x1, y1
    goal_p1 = torch.tensor([2.0, 2.0])
    dist_to_goal = torch.linalg.norm(pos_p1 - goal_p1)

    return dist_to_goal

def cost_func5(state, action):
    pos_p0 = state[0:2]  # x0, y0
    pos_p1 = state[4:6]  # x1, y1
    dist_to_other = torch.linalg.norm(pos_p1 - pos_p0)

    return 1.0 / (dist_to_other + 1e-6)

def cost_func6(state, action):
    if action is None:
        return 0
    act_p1 = action[2:4]  # ax1, ay1
    eps = 1e-6
    action_cost = torch.sqrt(torch.sum(act_p1 ** 2) + eps)

    return action_cost


def true_cost_p0(state, action):
    return 0.5 * cost_func1(state, action) + 2.0 * cost_func2(state, action) + 1.0 * cost_func3(state, action)
def true_cost_p1(state, action):
    return 1 * cost_func4(state, action) + 1.0 * cost_func5(state, action) + 1.0 * cost_func6(state, action)

# nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
# 
# interactive_keywords = ['merge', 'crosswalk', 'intersection', 'vehicle cut', 'yield', 'overtake']
# 
# interactive_scenes = [
#     scene for scene in nusc.scene
#     if any(kw in scene['name'].lower() or kw in scene['description'].lower()
#            for kw in interactive_keywords)
# ]
# 
# scene = interactive_scenes[0]
# x_traj, u_traj, t_traj = get_agent_trajectory(nusc, scene)

x_init = torch.Tensor([1,1.1,0.1,0.1, 0,0,0.5,0.5])

cost_funcs = [cost_func_p0, cost_func_p1]
dynamics_func = dyn

sim_param = SimulationParams(10,-1,10)
nl_game = NonlinearGame(dyn, cost_funcs, x_dims, x_dim, u_dims, u_dim, 2)

dem_results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param,
                                       nl_game,
                                       x_init,
                                       20, 2)

x_trajectories = dem_results.x_trajs
u_trajectories = dem_results.u_trajs

# print(x.shape, u.shape)

w = ma_irl(dynamics_func, [cost_func1, cost_func2, cost_func3, cost_func4, cost_func5, cost_func6], x_trajectories, u_trajectories, 10, [[0,1,2],[3,4,5]], 2)

print(w)