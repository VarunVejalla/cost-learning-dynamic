import numpy as np
import pickle
import plotly.graph_objs as go
from datetime import datetime as time
from plotly.subplots import make_subplots

from utils import *

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

DT = 0.1

A = torch.zeros(x_dim, x_dim)

# Define the submatrix
A_block = torch.tensor([
    [1, 0, DT, 0],
    [0, 1, 0, DT],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

# Assign to the top-left and bottom-right blocks of A
A[0:x_dims[0], 0:x_dims[0]] = A_block
A[x_dims[0]:, x_dims[0]:] = A_block

# Control input matrix B1
B1 = torch.zeros(x_dim, u_dims[0])
B1_block = torch.tensor([
    [0, 0],
    [0, 0],
    [DT, 0],
    [0, DT]
], dtype=torch.float32)
B1[0:x_dims[0], :] = B1_block

# Control input matrix B2
B2 = torch.zeros(x_dim, u_dims[0])
B2[x_dims[0]:, :] = B1_block  # same block reused

def dynamics_forward(s):
    state = s[0:x_dim]
    ctrl1 = s[x_dim:x_dim + u_dims[0]]
    ctrl2 = s[x_dim + u_dims[0]:]

    return A @ state + B1 @ ctrl1 + B2 @ ctrl2

def set_up_system(theta):
    # Weight matrices for state costs (player 1)
    w_state1 = torch.zeros(x_dim, x_dim)
    w_state1[0:x_dims[0], 0:x_dims[0]] = (theta[0] + theta[1]) * torch.eye(x_dims[0])
    w_state1[x_dims[0]:, 0:x_dims[0]] = theta[1] * torch.eye(x_dims[1], x_dims[0])
    w_state1[0:x_dims[0], x_dims[0]:] = theta[1] * torch.eye(x_dims[0], x_dims[1])
    w_state1[x_dims[0]:, x_dims[0]:] = (theta[0] + theta[1]) * torch.eye(x_dims[1])

    # Control costs (player 1)
    w_ctrl11 = theta[2] * torch.eye(u_dims[0])
    w_ctrl12 = theta[3] * torch.eye(u_dims[1])

    # Weight matrices for state costs (player 2)
    w_state2 = torch.zeros(x_dim, x_dim)
    w_state2[0:x_dims[0], 0:x_dims[0]] = (theta[4] + theta[5]) * torch.eye(x_dims[0])
    w_state2[x_dims[0]:, 0:x_dims[0]] = theta[5] * torch.eye(x_dims[1], x_dims[0])
    w_state2[0:x_dims[0], x_dims[0]:] = theta[5] * torch.eye(x_dims[0], x_dims[1])
    w_state2[x_dims[0]:, x_dims[0]:] = (theta[4] + theta[5]) * torch.eye(x_dims[1])

    # Control costs (player 2)
    w_ctrl21 = theta[6] * torch.eye(u_dims[0])
    w_ctrl22 = theta[7] * torch.eye(u_dims[1])

    # Dynamics dictionary
    Dynamics = {
        "A": [A for _ in range(plan_steps)],
        "B": [[B1 for _ in range(plan_steps)],
                [B2 for _ in range(plan_steps)]]
    }

    # Costs dictionary
    Costs = {
        "Q": [[w_state1 for _ in range(plan_steps + 1)],
             [w_state2 for _ in range(plan_steps + 1)]],
        "l": [[torch.zeros(x_dim) for _ in range(plan_steps + 1)],
            [torch.zeros(x_dim) for _ in range(plan_steps + 1)]],
        "R":   [[[w_ctrl11 for _ in range(plan_steps)],
[w_ctrl12 for _ in range(plan_steps)]],
                [[w_ctrl21 for _ in range(plan_steps)],
                 [w_ctrl22 for _ in range(plan_steps)]]]
    }

    return Dynamics, Costs


def generate_sim(x_init, theta, plan_steps, state_dim, ctrl_dim, num_agents, num = 200):

    x_trajectories, u_trajectories = [], []

    # compute the Quantal response equilibrium
    Dynamics, Costs = set_up_system(theta)
    Ns, alphas, covs = lqgame_QRE(Dynamics, Costs)

    # run simulations to get optimal equilibrium trajectories
    for i in range(num):
        x_history = torch.zeros((plan_steps+1, state_dim))
        x_history[0,:] = x_init
        u_history = torch.zeros((plan_steps, ctrl_dim))
        for t in range(plan_steps):
            
            u_means = []
            u_dists = []
            for j in range(num_agents):
                u_means.append(-Ns[j][-t-1] @ x_history[t, :] - alphas[j][-t-1])
                u_dists.append(MultivariateNormal(u_means[j], covariance_matrix=covs[j][-t-1]))
            
            u_sample = torch.cat([dist.sample() for dist in u_dists])
            u_history[t, :] = u_sample
            x_input = torch.cat([x_history[t, :], u_sample])
            x_history[t+1, :] = dynamics_forward(x_input)
        x_trajectories.append(x_history)
        u_trajectories.append(u_history)
    return x_trajectories, u_trajectories

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
    # how far first agent is from goal
    pos_p0 = state[0:2]  # x0, y0
    global goal_p0
    
    diff = pos_p0 - goal_p0
    dist_to_goal = torch.sqrt(torch.linalg.norm(diff)**2 + 1e-3)
    # dist_to_goal = torch.abs(torch.linalg.norm(diff))

    return dist_to_goal

def cost_func2(state, action):
    # how far first agent is from second agent
    pos_p0 = state[0:2]  # x0, y0
    pos_p1 = state[4:6]  # x1, y1
    dist_to_other = torch.linalg.norm(pos_p0 - pos_p1)
    # dist_to_other = torch.abs(torch.linalg.norm(pos_p0 - pos_p1))

    return 1.0 / (dist_to_other + 1e-6)

def cost_func3(state, action):
    # total acceleration for this action
    if action is None:
        return 0
    act_p0 = action[0:2]  # ax0, ay0
    eps = 1e-6
    action_cost = torch.sqrt(torch.sum(act_p0 ** 2) + eps)
    # action_cost = (torch.sum(act_p0**2) + eps)

    return action_cost

def cost_func4(state, action):
    pos_p1 = state[4:6]  # x1, y1
    global goal_p1

    diff = pos_p1 - goal_p1
    dist_to_goal = torch.sqrt(torch.linalg.norm(diff)**2 + 1e-3)
    # dist_to_goal = torch.abs(torch.linalg.norm(diff))

    return dist_to_goal

def cost_func5(state, action):
    pos_p0 = state[0:2]  # x0, y0
    pos_p1 = state[4:6]  # x1, y1
    dist_to_other = torch.linalg.norm(pos_p1 - pos_p0)
    # dist_to_other = torch.abs(torch.linalg.norm(pos_p1 - pos_p0))

    return 1.0 / (dist_to_other + 1e-6)

def cost_func6(state, action):
    if action is None:
        return 0
    act_p1 = action[2:4]  # ax1, ay1
    eps = 1e-6
    action_cost = torch.sqrt(torch.sum(act_p1 ** 2) + eps)
    # action_cost = torch.sqrt(torch.sum(act_p1**2) + eps)

    return action_cost


def true_cost_p0(state, action):
    # return 1.0 * cost_func1(state, action) + 1.1 * cost_func2(state, action) + 0.8 * cost_func3(state, action)
    return 1.0 * cost_func1(state, action) + 0.8 * cost_func3(state, action)
def true_cost_p1(state, action):
    # return 1.0 * cost_func4(state, action) + 0.7 * cost_func5(state, action) + 0.4 * cost_func6(state, action)
    return 1.0 * cost_func4(state, action) + 0.4 * cost_func6(state, action)


def plot_trajectory(x_trajectory):
    agent1_pos = x_trajectory[:, 0:2]  # (T, 2)
    agent2_pos = x_trajectory[:, 4:6]  # (T, 2)
    T = agent1_pos.shape[0]
    # Create frames for animation
    frames = []
    for t in range(T):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=agent1_pos[:t+1, 0], y=agent1_pos[:t+1, 1],
                        mode="lines+markers", name="Agent 1",
                        line=dict(color="blue")),
                go.Scatter(x=agent2_pos[:t+1, 0], y=agent2_pos[:t+1, 1],
                        mode="lines+markers", name="Agent 2",
                        line=dict(color="red"))
            ],
            name=str(t)
        ))

    # Initial figure setup
    fig = go.Figure(
        data=[
            go.Scatter(x=[agent1_pos[0, 0]], y=[agent1_pos[0, 1]],
                    mode="markers+text", name="Agent 1", text=["Start"],
                    marker=dict(color="blue", size=10)),
            go.Scatter(x=[agent2_pos[0, 0]], y=[agent2_pos[0, 1]],
                    mode="markers+text", name="Agent 2", text=["Start"],
                    marker=dict(color="red", size=10)),
        ],
        layout=go.Layout(
            xaxis=dict(range=[min(agent1_pos[:,0].min().item()-1, agent2_pos[:,0].min().item()-1), max(agent1_pos[:,0].max().item()+1, agent2_pos[:,0].max().item()+1)]),
            yaxis=dict(range=[min(agent1_pos[:,1].min().item()-1, agent2_pos[:,1].min().item()-1), max(agent1_pos[:,1].max().item()+1, agent2_pos[:,1].max().item()+1)]),
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[dict(label="Play", method="animate", args=[None])]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(t)]], label=str(t)) for t in range(T)],
                transition=dict(duration=0),
                x=0.1, xanchor="left",
                y=0, yanchor="top"
            )]
        ),
        frames=frames
    )

    fig.write_html(f"trajectory_animation_{time.now()}.html")


cost_funcs = [true_cost_p0, true_cost_p1]
dynamics_func = dyn

num_sims = 1
steps = 300
plan_steps = 300

sim_param = SimulationParams(steps,-1,plan_steps)
nl_game = NonlinearGame(dyn, cost_funcs, x_dims, x_dim, u_dims, u_dim, 2)

# Intersection of agents
agent1_init = [-5, 0]
agent2_init = [0, -5]
#agent1_goal = [5, 0] 
goal_p0 = torch.tensor([5, 0])
# agent2_goal = [0, 5]
goal_p1 = torch.tensor([0, 5])
# center = torch.tensor([-10, 0, 0, 0, -10, 0, 0, 0])
# std_devs = torch.tensor([2, 2, 1, 1, 2, 2, 1, 1]).sqrt()
# x_inits = [center + std_devs * torch.randn(8) for _ in range(num_sims)]
# x_inits = [center for _ in range(num_sims)]
# x_inits = [torch.tensor([-5, 0, 0, 0, 0, -5, 0, 0]),
#           torch.tensor([0, -5, 0, 0, -5, 0, 0, 0])]
# x_inits = [torch.tensor(agent1_init + [0,0] + agent2_init + [0,0]),
#            torch.tensor(agent2_init + [0,0] + agent1_init + [0,0])]
x_inits = [torch.tensor(agent1_init + [0,0] + agent2_init + [0,0]),
           torch.tensor(agent2_init + [0,0] + agent1_init + [0,0])]

print("agent1_init:", agent1_init, "agent1_goal:", goal_p0)
print("agent2_init:", agent2_init, "agent2_goal:", goal_p1)
print("x_inits:", x_inits)


x_trajectories, u_trajectories = generate_sim(x_inits[0], torch.tensor([5.0, 1.0, 2.0, 1.0,      5.0, 1.0, 1.0, 2.0]), 300, x_dim, u_dim, 2, 1)

# x_trajectories = []
# u_trajectories = []
# for i, x_init in enumerate(x_inits):
#     print(i, x_init)

#     dem_results, _, _ = generate_simulations(sim_param,
#                                         nl_game,
#                                         x_init,
#                                         1, 2)
    

#     x_trajectories.append(dem_results.x_trajs[0])
#     u_trajectories.append(dem_results.u_trajs[0])
#     print(dem_results.x_trajs[0])#, dem_results.u_trajs[0])
    
#     with open("x_trajectories.pkl", "wb") as file:
#         pickle.dump(x_trajectories, file)
#     with open("u_trajectories.pkl", "wb") as file:
#         pickle.dump(u_trajectories, file)

# # with open("x_trajectories.pkl", "rb") as file:
# #     x_trajectories = pickle.load(file)

plot_trajectory(x_trajectories[-1])

# print(x.shape, u.shape)

# w = ma_irl(dynamics_func, [cost_func1, cost_func2, cost_func3, cost_func4, cost_func5, cost_func6], x_trajectories, u_trajectories, 10, [[0,1,2],[3,4,5]], 2)

# print(w)