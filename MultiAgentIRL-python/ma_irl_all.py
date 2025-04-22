import numpy as np
from utils import SimulationParams, SimulationResults, NonlinearGame, generate_simulations, get_simulated_trajectories, lqgame_QRE

state_dim = 4 # 2 for position, 2 for velocity
action_dim = 2 # both acceleration

def get_feature_counts(x_trajectories, u_trajectories, cost_functions):
    # TODO: should this accept as input reference trajectories?
    # cost functions could maybe take in that reference as well
    
    # x_trajectories shape = (num_trajectories, steps_in_trajectory+1, whole state_dim)
    # u_trajectories shape = (num_trajectories, steps_in_trajectory, whole action_dim)
    
    # cost_function is list of functions
    # each function takes in as input state + acceleration
    # i.e. input should be two inputs, one of shape (num_agents, state_dim), other (num_agents, action_dim)
    
    # output: for each num_trajectories, for each cost function, average value of cost_function
    
    
    
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
                feature_counts[k, i] += cost_functions(state, control)
            
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


def ma_irl(dynamics, cost_functions, x_trajectories, u_trajectories, num_max_iter, agent_to_functions, num_agents):
    # x_trajectories shape = (num_trajectories, steps_in_trajectory+1, whole state_dim)
    # u_trajectories shape = (num_trajectories, steps_in_trajectory, whole action_dim)
    
    # cost_function is list of functions
    # each function takes in as input state + acceleration
    # i.e. input should be two inputs, one of shape (num_agents, state_dim), other (num_agents, action_dim)
    # agent_to_functions is e.g. [[0,1,2,3],[4,5,6,7]]. First agent has functions 0,1,2,3.
    
    # output: for each num_trajectories, for each cost function, average value of cost_function
    
    
    
    avg_dem_feature_counts = get_feature_counts(x_trajectories, u_trajectories, cost_functions)
    
    w = # initialize
    
    gamma = 0.001

    while w not converged and num iterations < num_max_iter:
        # update gamma
        for agent in range(num_agents):
            sim_x, sim_u = get_simulated_trajectories(current w)
            avg_sim_feature_counts = get_feature_counts(sim_x, sim_u, cost_functions)
            relevant_indices = agent_to_functions[agent]
            w[relevant_indices] -= (gamma * (avg_dem_feature_counts - avg_sim_feature_counts))[relevant_indices]
        
    return w