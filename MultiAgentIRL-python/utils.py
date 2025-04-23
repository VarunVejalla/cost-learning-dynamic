import numpy as np
import torch
import torch.autograd.functional as functional

class SimulationParams:
    def __init__(self, steps, horizon, plan_steps):
        self.steps = steps
        self.horizon = horizon
        self.plan_steps = plan_steps

class SimulationResults:
    def __init__(self, x_trajs, u_trajs):
        self.x_trajs = x_trajs
        self.u_trajs = u_trajs

class NonlinearGame:
    
    def __init__(self, dynamics_func, cost_funcs, x_dims, x_dim, u_dims, u_dim, num_agents):
        """_summary_

        Args:
            dynamics_func (function from whole state and all actions to new state): _description_
            cost_funcs (functions from ): _description_
            x_dims (_type_): _description_
            x_dim (_type_): _description_
            u_dims (_type_): _description_
            u_dim (_type_): _description_
            num_agents (_type_): _description_
        """
        self.dynamics = dynamics_func
        
        self.x_dims = x_dims
        self.x_dim = x_dim
        self.u_dims = u_dims
        self.u_dim = u_dim
        self.cost_funcs = cost_funcs
        self.num_agents = num_agents
   

def generate_simulations(sim_param:SimulationParams, nl_game:NonlinearGame, x_init:torch.Tensor, num_sim:int, num_players:int):
    """_summary_

    Args:
        sim_param (SimulationParams): _description_
        nl_game (NonlinearGame): _description_
        x_init (one dimensional tensor): _description_
        num_sim (int): _description_
        num_players (int): _description_
    """
    
    # TODO: Kyle
    
    steps = sim_param.steps
    plan_steps = sim_param.plan_steps
    x_dim = nl_game.x_dim
    u_dim = nl_game.u_dim
    x_trajs, u_trajs = [], []
    x_trajs_data = torch.zeros((num_sim, steps+1, x_dim))
    u_trajs_data = torch.zeros((num_sim, steps+1, u_dim))
    for i in range(num_sim):
        if (i+1)%3 == 0:
            print(" --- generating sim number : ", i, " ---")
        
        x_history = torch.zeros((steps+1, x_dim))
        x_history[0] = x_init
        u_history = torch.zeros((steps, u_dim))
        
        for t in range(steps):
            Nlist_all, alphalist_all, cov_all, x_nominal, u_nominal = solve_iLQGame(sim_param=sim_param, nl_game=nl_game, x_init=x_history[t])
            delta_x = x_history[t] - x_nominal[0]
            u_dists = []
            for ip in range(num_players):
                # TODO: clean this up
                u_mean = -Nlist_all[ip][len(Nlist_all[ip])-1] * delta_x - alphalist_all[ip][len(alphalist_all[ip])-1]
                # TODO: should the covariance be something like Symmetric(cov_all[ip][end])??
                u_cov = torch.eye(2)
                u_dist = torch.distributions.MultivariateNormal(u_mean, covariance_matrix=u_cov)
                u_dists.append(u_dist)
            control = []
            for u_dist in u_dists:
                control.append(u_dist.sample())
            control = torch.stack(control)
            
            u_history[t] = control + u_nominal[0]
            x_history[t+1] = nl_game.dynamics(x_history[t], u_history[t])
        
        x_trajs.append(x_history)
        u_trajs.append(u_history)
        x_trajs_data[i] = x_history
        u_trajs_data[i, :steps] = u_history
    
    return SimulationResults(x_trajs, u_trajs), x_trajs_data, u_trajs_data

# def lqgame_QRE(dynamic_dicts, cost_dicts):
#     # TODO: John
#     # from Varun: feel free to change the format of what these look like
#     #               - solve_iLQGame is what writes to these dictionaries
    
#     A = dynamic_dicts["A"]
#     B = dynamic_dicts["B"]
#     Q = cost_dicts["Q"]
#     l = cost_dicts["l"]
#     R = cost_dicts["R"]
    
#     num_agents = len(B)
    
#     T = len(A)
#     n = B[0][0].shape[0]
    
#     m = []
#     for i in range(num_agents):
#         if len(B[i][0].shape) == 1:
#             m.append(1)
#         else:
#             m.append(B[i][0].shape[1])
    
#     P = [[] for _ in range(num_agents)]
#     alpha = [[] for _ in range(num_agents)]
#     cov = [[] for _ in range(num_agents)]
    
#     Z = [[] for _ in range(num_agents)]
#     F = []
#     zeta = [[] for _ in range(num_agents)]
#     beta = []
    
#     for i in range(num_agents):
#         Z[i].append(Q[[i][-1]])
#         zeta[i].append(l[i][-1])
    
#     for t in range(T-1, -1, -1):
#         # TODO
#         continue
    
#     return P, alpha, cov

def lqgame_QRE(dynamic_dicts, cost_dicts):
    # TODO: I have no idea how to test this, but I do believe
    # that I implemented it pretty rigourously identical to the 
    # paper it's definitions. If you would like me to explain 
    # something let me know! - John

    As = dynamic_dicts["A"]
    Bs = dynamic_dicts["B"]
    Qs = cost_dicts["Q"]
    ls = cost_dicts["l"]
    Rs = cost_dicts["R"]
    
    num_agents = len(Bs)
    
    T = len(As)
    n = Bs[0][0].shape[0]
    
    m = []
    for i in range(num_agents):
        if len(Bs[i][0].shape) == 1:
            m.append(1)
        else:
            m.append(Bs[i][0].shape[1])
    
    Ps = [[] for _ in range(num_agents)]
    alphas = [[] for _ in range(num_agents)]
    covs = [[] for _ in range(num_agents)]
    
    Zs = [[] for _ in range(num_agents)]
    Fs = []
    zetas = [[] for _ in range(num_agents)]
    betas = []
    
    # for doing the linear quadratic game backwards passes, 
    # initialize the terminal state to the terminal costs and
    # then begin to iterate backwards for T iterations
    for i in range(num_agents):
        Zs[i].append(Qs[i][len(Qs[i])-1])
        zetas[i].append(ls[i][len(ls[i])-1])
    
    sum_m = sum(m)
    
    for t in range(T-1, -1, -1):
        Z_n = []
        for i in range(num_agents):
            Z_n.append(Zs[i][len(Zs[i])-1])
        S = torch.zeros((sum_m,sum_m))
        
        start, end = 0,0
        for i in range(num_agents):
            start, end = end, end + m[i]
            start_j, end_j = 0,0
            for j in range(num_agents):
                start_j, end_j = end_j, end_j + m[i]
                
                if i == j:
                    S[start:end, start_j:end_j] = Rs[i,i] + Bs[i][t].T @ Z_n[i] @ Bs[j][t]
                else:
                    S[start:end, start_j:end_j] = Bs[i][t].T @ Z_n[i] @ Bs[j][t]
        
        YN = torch.zeros((sum_m, n))
        start, end = 0,0
        for i in range(num_agents):
            start, end = end, end + m[i]
            YN[start:end] = Bs[i].T @ Z_n[i] @ As[t]
        
        temp_P = torch.linalg.solve(S, YN)# temp_P = S\YN
        start, end = 0,0
        P = []
        for i in range(num_agents):
            start, end = end, end + m[i]
            P.append(temp_P[start:end])
        
        for i in range(num_agents):
            Ps[i].append(P[i])
        
        start, end = 0,0
        for i in range(num_agents):
            start, end = end, end + m[i]
            covs[i].append(torch.linalg.inv(S[start:end, start:end]))
        
        zeta_n = []
        
        for i in range(num_agents):
            zeta_n.append(zetas[i][len(zetas[i]-1)])
        YA = torch.zeros((sum_m, 1))
        start, end = 0,0
        for i in range(num_agents):
            start, end = end, end + m[i]
            YA[start:end] = Bs[i][t].T @ zeta_n[i]
        
        # TODO
        # temp_alpha = S\YA
        temp_alpha = torch.linalg.solve(S, YA)
        alpha = []
        start, end = 0,0
        for i in range(num_agents):
            start, end = end, end + m[i]
            
            alpha.append(temp_alpha[start:end])
            alphas[i].append(alpha[i])
        
        F = As[t] - torch.sum(Bs[i][t] @ P[i] for i in range(num_agents))
        Fs.append(F)
        
        beta = - torch.sum(Bs[i][t] @ alpha[i] for i in range(num_agents))
        zeta = []
        for i in range(num_agents):
            zeta.append(F.T @ Z_n[i] @ beta + F.T @ zeta_n[i] + torch.stack([P[j].T @ Rs[i][j][t] @ alpha[j] for j in range(num_agents)]).sum(dim=0) + ls[i][t])
        
        for i in range(num_agents):
            zetas[i].append(zeta[i])
        Z = []
        for i in range(num_agents):
            Z.append(F.T @ Z_n[i] @ F + torch.stack([P[j].T @ Rs[i][j][t] @ P[j] for j in range(num_agents)]).sum(dim=0) + Qs[i][t])
            Zs[i].append(Z[i])
        
    return Ps, alphas, covs

def solve_iLQGame(sim_param:SimulationParams, nl_game:NonlinearGame, x_init:torch.Tensor):
    # TODO: Varun
    num_player = len(nl_game.u_dims)
    
    
    plan_steps = sim_param.plan_steps
    x_dim = nl_game.x_dim
    u_dim = nl_game.u_dim
    u_dims = nl_game.u_dims
    
    x_trajectory = torch.zeros((plan_steps+1, x_dim))
    x_trajectory[0] = x_init
    u_trajectory = torch.zeros((plan_steps, u_dim))
    
    for t in range(plan_steps):
        x_trajectory[t+1] = nl_game.dynamics(x_trajectory[t], u_trajectory[t])
    
    tol = 1.0
    itr = 1
    err = 10
    max_itr = 50
    x_trajectory_prev = x_trajectory
    u_trajectory_prev = u_trajectory
    N = [[] for _ in range(num_player)]
    alpha = [[] for _ in range(num_player)]
    cov = [[] for _ in range(num_player)]
    
    while err > tol and itr < max_itr:
        Dynamics = {}
        Dynamics["A"] = []
        Dynamics["B"] = [[] for _ in range(num_player)]

        Costs = {}
        Costs["Q"] = [[] for _ in range(num_player)]
        Costs["l"] = [[] for _ in range(num_player)]
        Costs["R"] = [[[] for _ in range(num_player)] for _ in range(num_player)]
        
        for t in range(plan_steps):
            dynamics_input = torch.cat([x_trajectory_prev[t], u_trajectory_prev[t]])
            jac = functional.jacobian(nl_game.dynamics, dynamics_input)
            A = jac[:, :x_dim]
            start_index, end_index = 0,x_dim
            
            for i in range(num_player):
                start_index, end_index = end_index, end_index+u_dims[i]
                Dynamics["B"][i].append(jac[:, start_index:end_index])
            
            gradients = []
            hessians = []
            cost_input = torch.cat([x_trajectory_prev[t], u_trajectory[t]])
            for i in range(num_player):
                gradients.append(functional.jacobian(nl_game.cost_funcs[i], cost_input))
                hessians.append(functional.hessian(nl_game.cost_funcs[i], cost_input))
            
            Q = [hessians[i][:x_dim, :x_dim] for i in range(num_player)]
            for i in range(num_player):
                r = torch.min(torch.linalg.eigvals(Q[i]))
                if r <= 0.0:
                    Q[i] += (torch.abs(r) + 1e-3) * torch.eye(x_dim)
            
            for i in range(num_player):
                Costs["Q"][i].append(Q[i])
                Costs["l"][i].append(gradients[i][:x_dim])
                start_index, end_index = 0, 2*x_dim
                for j in range(num_player):
                    start_index, end_index = end_index, end_index+u_dims[j]
                    Costs["R"][i][j] = hessians[i][start_index:end_index, start_index:end_index]
            
        gradients = []
        hessians = []
        cost_input = torch.cat([x_trajectory_prev[plan_steps], u_trajectory[plan_steps]])
        for i in range(num_player):
            gradients.append(functional.jacobian(nl_game.cost_funcs[i], cost_input))
            hessians.append(functional.hessian(nl_game.cost_funcs[i], cost_input))
        Q = [hessians[i][:x_dim, :x_dim] for i in range(num_player)]
        for i in range(num_player):
            r = torch.min(torch.linalg.eigvals(Q[i]))
            if r <= 0.0:
                Q[i] += (torch.abs(r) + 1e-3) * torch.eye(x_dim)
        for i in range(num_player):
            Costs["Q"][i].append(Q[i])
            Costs["l"][i].append(gradients[i][:x_dim])
        
        
        N, alpha, cov = lqgame_QRE(Dynamics, Costs)
        
        step_size = 1.0
        done = False
        while not done:
            u_trajectory = torch.zeros((plan_steps, u_dim))
            x_trajectory = torch.zeros((plan_steps, x_dim))
            x_trajectory[0,:] = x_init
            for t in range(plan_steps):
                delta_x = x_trajectory[t] - x_trajectory_prev[t]
                
                reverse_index = len(N[agent])-t-1
                
                start_index = 0
                end_index = 0
                
                
                for agent in range(num_player):
                    start_index,end_index = end_index,end_index+u_dims[agent]
                    u_trajectory[t, start_index:end_index] = (-N[agent][reverse_index] * delta_x - alpha[agent][reverse_index] * step_size) + u_trajectory_prev[t,start_index:end_index]
                
                x_trajectory[t+1] = nl_game.dynamics(x_trajectory[t], u_trajectory[t])
            if torch.max(torch.abs(x_trajectory - x_trajectory_prev)) > 1.0:
                step_size /= 2
            else:
                done = True
        
        err = torch.sum(torch.abs(x_trajectory_prev - x_trajectory))
        itr += 1
        x_trajectory_prev = x_trajectory
        u_trajectory_prev = u_trajectory
    
    return N, alpha, cov, x_trajectory_prev, u_trajectory_prev

def cost_func_p1(state, action):
    return torch.linalg.norm(state) + torch.linalg.norm(action)

def cost_func_p2(state, action):
    return torch.linalg.norm(state) + torch.linalg.norm(action)

def dyn(state, action):
    return state + torch.concat([action, action])



x_init = torch.Tensor([[0,0,0,0],[1,1,0,0]])

x_dims = [4,4]
x_dim = 8

u_dims = [2,2]
u_dim = 4

cost_funcs = [cost_func_p1, cost_func_p2]
dynamics_func = dyn

sim_param = SimulationParams(100,-1,5)
nl_game = NonlinearGame(dyn, cost_funcs, x_dims, x_dim, u_dims, u_dim, 2)




generate_simulations(sim_param, nl_game, x_init, 2, 2)