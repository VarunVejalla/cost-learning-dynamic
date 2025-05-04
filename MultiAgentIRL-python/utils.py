import os
import numpy as np
import torch
import torch.autograd.functional as functional
from torch.distributions.multivariate_normal import MultivariateNormal
import argparse
import matplotlib.pyplot as plt

# OPTIMIZE = True
# OPTIMIZE = os.environ.get('optimize', 'False').lower() in ['true', '1', 'yes']
# print(OPTIMIZE)

parser = argparse.ArgumentParser(description="Whether or not to run the optimized code")
parser.add_argument("-o", "--optimize", default=False, help="Whether or not to run the optmized code")
args = parser.parse_args()
OPTIMIZE = args.optimize in ["True", "true", "1", "yes"]
OPTIMIZE = False

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
   

def generate_simulations(sim_param:SimulationParams, 
                         nl_game:NonlinearGame, 
                         x_init:torch.Tensor, 
                         traj_ref:torch.Tensor,
                         num_sim:int, 
                         num_players:int):
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
            Nlist_all, alphalist_all, cov_all, x_nominal, u_nominal = solve_iLQGame(sim_param=sim_param, 
                                                                                    nl_game=nl_game, 
                                                                                    x_init=x_history[t],
                                                                                    traj_ref=traj_ref[t:t+plan_steps])
            delta_x = x_history[t] - x_nominal[0]
            u_dists = []
            for ip in range(num_players):
                # TODO: clean this up
                u_mean = -Nlist_all[ip][len(Nlist_all[ip])-1] @ delta_x - alphalist_all[ip][len(alphalist_all[ip])-1]
                # TODO: should the covariance be something like Symmetric(cov_all[ip][end])??
                
                # u_cov = torch.eye(nl_game.u_dims[ip])
                u_cov = torch.mean(torch.stack(cov_all[ip]), dim=0)
                u_dist = torch.distributions.MultivariateNormal(u_mean, covariance_matrix=u_cov)
                u_dists.append(u_dist)
            control = []
            for u_dist in u_dists:
                control.append(u_dist.sample())
            control = torch.cat(control)
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

# def lqgame_QRE(dynamic_dicts, cost_dicts):
#     # TODO: I have no idea how to test this, but I do believe
#     # that I implemented it pretty rigourously identical to the 
#     # paper it's definitions. If you would like me to explain 
#     # something let me know! - John

#     As = dynamic_dicts["A"]
    
#     Bs = dynamic_dicts["B"]
    
    
    
#     Qs = cost_dicts["Q"]
#     ls = cost_dicts["l"]
#     Rs = cost_dicts["R"]
    
#     num_agents = len(Bs)
    
#     T = len(As)
#     n = Bs[0][0].shape[0]
    
#     m = []
#     for i in range(num_agents):
#         if len(Bs[i][0].shape) == 1:
#             m.append(1)
#         else:
#             m.append(Bs[i][0].shape[1])

#     Ps = [[] for _ in range(num_agents)]
#     alphas = [[] for _ in range(num_agents)]
#     covs = [[] for _ in range(num_agents)]
    
#     Zs = [[] for _ in range(num_agents)]
#     Fs = []
#     zetas = [[] for _ in range(num_agents)]
#     betas = []
    
#     # for doing the linear quadratic game backwards passes, 
#     # initialize the terminal state to the terminal costs and
#     # then begin to iterate backwards for T iterations
#     for i in range(num_agents):
#         Zs[i].append(Qs[i][len(Qs[i])-1])
#         zetas[i].append(ls[i][len(ls[i])-1])
    
#     sum_m = sum(m)
    
#     for t in range(T-1, -1, -1):
#         Z_n = []
#         for i in range(num_agents):
#             Z_n.append(Zs[i][len(Zs[i])-1])
#         S = torch.zeros((sum_m,sum_m))
        
#         start, end = 0,0
#         for i in range(num_agents):
#             start, end = end, end + m[i]
#             start_j, end_j = 0,0
#             for j in range(num_agents):
#                 start_j, end_j = end_j, end_j + m[i]
                
#                 if i == j:
#                     S[start:end, start_j:end_j] = Rs[i][i][t] + Bs[i][t].T @ Z_n[i] @ Bs[j][t]
#                 else:
#                     S[start:end, start_j:end_j] = Bs[i][t].T @ Z_n[i] @ Bs[j][t]
        
#         YN = torch.zeros((sum_m, n))
#         start, end = 0,0
#         for i in range(num_agents):
#             start, end = end, end + m[i]
            
#             YN[start:end] = Bs[i][t].T @ Z_n[i] @ As[t]
        
#         temp_P = torch.linalg.solve(S, YN)# temp_P = S\YN
#         start, end = 0,0
#         P = []
#         for i in range(num_agents):
#             start, end = end, end + m[i]
#             P.append(temp_P[start:end])
        
#         for i in range(num_agents):
#             Ps[i].append(P[i])
        
#         start, end = 0,0
#         for i in range(num_agents):
#             start, end = end, end + m[i]
#             covs[i].append(torch.linalg.inv(S[start:end, start:end]))
        
#         zeta_n = []
        
#         for i in range(num_agents):
#             zeta_n.append(zetas[i][len(zetas[i])-1])
#         YA = torch.zeros((sum_m,))
#         start, end = 0,0
#         for i in range(num_agents):
#             start, end = end, end + m[i]
#             YA[start:end] = Bs[i][t].T @ zeta_n[i]
        
#         temp_alpha = torch.linalg.solve(S, YA)
#         alpha = []
#         start, end = 0,0
#         for i in range(num_agents):
#             start, end = end, end + m[i]
            
#             alpha.append(temp_alpha[start:end])
#             alphas[i].append(alpha[i])
        
#         F = As[t] - torch.stack([Bs[i][t] @ P[i] for i in range(num_agents)]).sum(dim=0)
#         Fs.append(F)
        
#         beta = - torch.stack([Bs[i][t] @ alpha[i] for i in range(num_agents)]).sum(dim=0)
#         betas.append(beta)
        
#         for i in range(num_agents):
#             # TODO: should we be adding ls[i][t] or not?
#             zetas[i].append(F.T @ Z_n[i] @ beta + F.T @ zeta_n[i] + torch.stack([P[j].T @ Rs[i][j][t] @ alpha[j] for j in range(num_agents)]).sum(dim=0) + ls[i][t])
#         for i in range(num_agents):
#             Zs[i].append(F.T @ Z_n[i] @ F + torch.stack([P[j].T @ Rs[i][j][t] @ P[j] for j in range(num_agents)]).sum(dim=0) + Qs[i][t])
    
    
#     # # # verifying eq (28)
#     # for t in range(1, T):
#     #     for i in range(num_agents):
#     #         delta = Fs[t].T @ Zs[i][t] @ Fs[t] - Zs[i][t+1] + Qs[i][t]
#     #         for j in range(num_agents):
#     #             delta += Ps[j][t].T @ Rs[i][j][t] @ Ps[j][t]
            
#     #         # for some reason, this is not 0 or even close to 0
#     #         print(torch.linalg.norm(delta))
    
#     # verifying eq (29)
#     # for t in range(1, T):
#     #     for i in range(num_agents):
#     #         delta = Fs[t].T @ (zetas[i][t-1] + Zs[i][t-1]@betas[t]) - zetas[i][t]
#     #         for j in range(num_agents):
#     #             delta += Ps[j][t].T @ Rs[i][j][t] @ alphas[j][t]
#     #         print(torch.linalg.norm(delta))
    
#     # verifying eq (30)
#     # for t in range(T):
#     #     delta = As[t] - Fs[t]
#     #     for j in range(num_agents):
#     #         delta -= Bs[j][t] @ Ps[j][t]
#     #     print(torch.linalg.norm(delta))
    
#     # verifying eq (31)
#     # for t in range(T):
#     #     delta = -betas[t]
#     #     for j in range(num_agents):
#     #         delta -= Bs[j][t]@alphas[j][t]
#     #     print(torch.linalg.norm(delta))
    
    
#     # print("dynamic dicts", dynamic_dicts)
#     # print("cost dicts", cost_dicts)
#     # print("Ps", Ps)
#     # print("alphas", alphas)
#     # print('covs', covs)
#     # print("Zs", Zs)
#     # print("Fs", Fs)
#     # print("zetas", zetas)
#     # print("betas", betas)
    
    
#     return Ps, alphas, covs

def lqgame_QRE(dynamic_dicts, cost_dicts):
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
        m.append(1 if len(Bs[i][0].shape) == 1 else Bs[i][0].shape[1])

    sum_m = sum(m)

    # Preallocate all lists
    Ps = [[None for _ in range(T)] for _ in range(num_agents)]
    alphas = [[None for _ in range(T)] for _ in range(num_agents)]
    covs = [[None for _ in range(T)] for _ in range(num_agents)]
    Zs = [[None for _ in range(T + 1)] for _ in range(num_agents)]
    zetas = [[None for _ in range(T + 1)] for _ in range(num_agents)]
    Fs = [None for _ in range(T)]
    betas = [None for _ in range(T)]

    # Initialize terminal cost (t = T)
    for i in range(num_agents):
        Zs[i][T] = Qs[i][T]
        zetas[i][T] = ls[i][T]

    for t in reversed(range(T)):
        Z_n = [Zs[i][t + 1] for i in range(num_agents)]

        S = torch.zeros((sum_m, sum_m))
        start, end = 0, 0
        for i in range(num_agents):
            start, end = end, end + m[i]
            start_j, end_j = 0, 0
            for j in range(num_agents):
                start_j, end_j = end_j, end_j + m[j]
                if i == j:
                    S[start:end, start_j:end_j] = Rs[i][i][t] + Bs[i][t].T @ Z_n[i] @ Bs[j][t]
                else:
                    S[start:end, start_j:end_j] = Bs[i][t].T @ Z_n[i] @ Bs[j][t]

        YN = torch.zeros((sum_m, n))
        start, end = 0, 0
        for i in range(num_agents):
            start, end = end, end + m[i]
            YN[start:end] = Bs[i][t].T @ Z_n[i] @ As[t]

        temp_P = torch.linalg.solve(S, YN)
        start, end = 0, 0
        for i in range(num_agents):
            start, end = end, end + m[i]
            Ps[i][t] = temp_P[start:end]
            covs[i][t] = torch.linalg.inv(S[start:end, start:end])

        zeta_n = [zetas[i][t + 1] for i in range(num_agents)]
        YA = torch.zeros((sum_m,))
        start, end = 0, 0
        for i in range(num_agents):
            start, end = end, end + m[i]
            YA[start:end] = Bs[i][t].T @ zeta_n[i]

        temp_alpha = torch.linalg.solve(S, YA)
        start, end = 0, 0
        for i in range(num_agents):
            start, end = end, end + m[i]
            alphas[i][t] = temp_alpha[start:end]

        F = As[t] - sum(Bs[j][t] @ Ps[j][t] for j in range(num_agents))
        Fs[t] = F

        beta = -sum(Bs[j][t] @ alphas[j][t] for j in range(num_agents))
        betas[t] = beta

        for i in range(num_agents):
            zetas[i][t] = (
                F.T @ Z_n[i] @ beta
                + F.T @ zeta_n[i]
                + sum(Ps[j][t].T @ Rs[i][j][t] @ alphas[j][t] for j in range(num_agents))
                #+ ls[i][t]
            )
            Zs[i][t] = (
                F.T @ Z_n[i] @ F
                + sum(Ps[j][t].T @ Rs[i][j][t] @ Ps[j][t] for j in range(num_agents))
                + Qs[i][t]
            )
    
    # verifying eq 25 (ieee)
    # for t in range(T):
    #     for i in range(num_agents):
    #         LHS = (Rs[i][i][t] + Bs[i][t].T @ Zs[i][t+1] @ Bs[i][t]) @ Ps[i][t]
    #         for j in range(num_agents):
    #             if j == i:
    #                 continue
    #             LHS += Bs[i][t].T @ Zs[i][t+1] @ Bs[j][t] @ Ps[j][t]
            
    #         RHS = Bs[i][t].T @ Zs[i][t+1] @ As[t]
    #         print(torch.linalg.norm(LHS-RHS))
    
    # verifying eq 26 (ieee)
    # for t in range(T):
    #     for i in range(num_agents):
    #         LHS = (Rs[i][i][t] + Bs[i][t].T @ Zs[i][t+1] @ Bs[i][t]) @ alphas[i][t]
    #         for j in range(num_agents):
    #             if j == i:
    #                 continue
    #             LHS += Bs[i][t].T @ Zs[i][t+1] @ Bs[j][t] @ alphas[j][t]
            
    #         RHS = Bs[i][t].T @ zetas[i][t+1]
    #         print(torch.linalg.norm(LHS-RHS))
            
    
    # verifying eq 27 (ieee)
    # for t in range(T):
    #     for i in range(num_agents):
    #         delta = Fs[t].T @ Zs[i][t+1] @ Fs[t] + Qs[i][t] - Zs[i][t]
    #         for j in range(num_agents):
    #             delta += Ps[j][t].T @ Rs[i][j][t] @ Ps[j][t]
    #         print(torch.linalg.norm(delta))
    
    # verifying eq 28 (ieee)
    # for t in range(T):
    #     for i in range(num_agents):
    #         delta = Fs[t].T @ (zetas[i][t+1] + Zs[i][t+1] @ betas[t]) - zetas[i][t]
    #         for j in range(num_agents):
    #             delta += Ps[j][t].T @ Rs[i][j][t] @ alphas[j][t]
    #         print(torch.linalg.norm(delta))
    

    return Ps, alphas, covs#, Zs, zetas, Fs, betas


def combine_hessian_blocks(hessian_blocks):
    row_blocks = []
    for row in hessian_blocks:
        row_blocks.append(torch.cat(row, dim=1))
    full_hessian = torch.cat(row_blocks, dim=0)
    return full_hessian

def solve_iLQGame(sim_param: SimulationParams, nl_game: NonlinearGame, x_init: torch.Tensor, traj_ref: torch.Tensor):
    num_player = len(nl_game.u_dims)
    plan_steps = sim_param.plan_steps
    x_dim = nl_game.x_dim
    u_dim = nl_game.u_dim
    u_dims = nl_game.u_dims

    x_trajectory = torch.zeros((plan_steps + 1, x_dim))
    x_trajectory[0] = x_init
    u_trajectory = torch.zeros((plan_steps, u_dim))
    
    for t in range(plan_steps):
        x_trajectory[t + 1] = nl_game.dynamics(x_trajectory[t], u_trajectory[t])

    tol = 1.0
    err = float('inf')
    max_itr = 10
    itr = 0
    x_trajectory_prev = x_trajectory.clone()
    u_trajectory_prev = u_trajectory.clone()
    
    while err > tol and itr < max_itr:
        Dynamics = {
            "A": torch.empty((plan_steps, x_dim, x_dim)),
            "B": [torch.empty((plan_steps, x_dim, u_dims[i])) for i in range(num_player)],
        }
        Costs = {
            "Q": [torch.empty((plan_steps + 1, x_dim, x_dim)) for _ in range(num_player)],
            "l": [torch.empty((plan_steps + 1, x_dim)) for _ in range(num_player)],
            "R": [[torch.empty((plan_steps, u_dims[j], u_dims[i])) for j in range(num_player)] for i in range(num_player)],
        }

        for t in range(plan_steps):
            x_t = x_trajectory_prev[t].detach().requires_grad_()
            u_t = u_trajectory_prev[t].detach().requires_grad_()
            jac_xu = torch.cat(functional.jacobian(nl_game.dynamics, (x_t, u_t)), dim=-1)

            Dynamics["A"][t] = jac_xu[:, :x_dim]
            col_start = x_dim
            for i in range(num_player):
                col_end = col_start + u_dims[i]
                Dynamics["B"][i][t] = jac_xu[:, col_start:col_end]
                col_start = col_end

            gradients, hessians = [], []
            for i in range(num_player):
                grad = torch.cat(functional.jacobian(nl_game.cost_funcs[i], (x_t, u_t)), dim=-1)
                hess = combine_hessian_blocks(functional.hessian(nl_game.cost_funcs[i], (x_t, u_t)))
                gradients.append(grad)
                hessians.append(hess)

            for i in range(num_player):
                Q = hessians[i][:x_dim, :x_dim]
                eig_min = torch.min(torch.linalg.eigvalsh(Q))
                if eig_min <= 0:
                    Q += (torch.abs(eig_min) + 1e-3) * torch.eye(x_dim)
                Costs["Q"][i][t] = Q
                Costs["l"][i][t] = gradients[i][:x_dim]
                offset = x_dim
                for j in range(num_player):
                    u_j_dim = u_dims[j]
                    Costs["R"][i][j][t] = hessians[i][offset:offset+u_j_dim, offset:offset+u_j_dim]
                    offset += u_j_dim

        # Terminal cost
        x_T = x_trajectory_prev[plan_steps].detach().requires_grad_()
        u_Tm1 = u_trajectory_prev[plan_steps-1].detach().requires_grad_()
        for i in range(num_player):
            grad = torch.cat(functional.jacobian(nl_game.cost_funcs[i], (x_T, u_Tm1)), dim=-1)
            hess = combine_hessian_blocks(functional.hessian(nl_game.cost_funcs[i], (x_T, u_Tm1)))
            Q = hess[:x_dim, :x_dim]
            eig_min = torch.min(torch.linalg.eigvalsh(Q))
            if eig_min <= 0:
                Q += (torch.abs(eig_min) + 1e-3) * torch.eye(x_dim)
            Costs["Q"][i][plan_steps] = Q
            Costs["l"][i][plan_steps] = grad[:x_dim]

        N, alpha, cov = lqgame_QRE(Dynamics, Costs)

        step_size = 1.0
        done = False
        while not done:
            u_trajectory = torch.zeros((plan_steps, u_dim))
            x_trajectory = torch.zeros((plan_steps + 1, x_dim))
            x_trajectory[0] = x_init

            for t in range(plan_steps):
                delta_x = x_trajectory[t] - x_trajectory_prev[t]
                start_idx = 0
                for agent in range(num_player):
                    end_idx = start_idx + u_dims[agent]
                    u_nominal = u_trajectory_prev[t, start_idx:end_idx]
                    u_correction = -N[agent][t] @ delta_x - alpha[agent][t] * step_size
                    u_trajectory[t, start_idx:end_idx] = u_nominal + u_correction
                    start_idx = end_idx

                x_trajectory[t + 1] = nl_game.dynamics(x_trajectory[t], u_trajectory[t])

            if torch.max(torch.abs(x_trajectory - x_trajectory_prev)) > 1.0:
                step_size /= 2
            else:
                done = True

        err = torch.sum(torch.abs(x_trajectory - x_trajectory_prev))
        x_trajectory_prev = x_trajectory.clone()
        u_trajectory_prev = u_trajectory.clone()
        itr += 1

    return N, alpha, cov, x_trajectory_prev, u_trajectory_prev




def cost_func_p0(state, action):
    
    
    pos_p0 = state[0:2]  # x0, y0
    pos_p1 = state[4:6]  # x1, y1
    act_p0 = action[0:2]  # ax0, ay0

    goal_p0 = torch.tensor([1.0, 1.0], device=state.device, dtype=state.dtype)

    dist_to_goal = torch.linalg.norm(pos_p0 - goal_p0)
    dist_to_other = torch.linalg.norm(pos_p0 - pos_p1)
    eps = 1e-6
    action_cost = torch.sqrt(torch.sum(act_p0 ** 2) + eps)


    cost = 0.5 * dist_to_goal + 2.0 / (dist_to_other + 1e-6) + action_cost
    return cost

def cost_func_p1(state, action):
    pos_p0 = state[0:2]  # x0, y0
    pos_p1 = state[4:6]  # x1, y1
    act_p1 = action[2:4]  # ax1, ay1

    goal_p1 = torch.tensor([2.0, 2.0], device=state.device, dtype=state.dtype)

    dist_to_goal = torch.linalg.norm(pos_p1 - goal_p1)
    dist_to_other = torch.linalg.norm(pos_p1 - pos_p0)
    eps = 1e-6
    action_cost = torch.sqrt(torch.sum(act_p1 ** 2) + eps)

    cost = dist_to_goal + 1.0 / (dist_to_other + 1e-6) + action_cost
    return cost

def dyn(state, action):
    delta_time = 0.5
    
    pos0 = state[0:2]
    vel0 = state[2:4]
    pos1 = state[4:6]
    vel1 = state[6:8]

    acc0 = action[0:2]
    acc1 = action[2:4]

    new_pos0 = pos0 + vel0 * delta_time
    new_vel0 = vel0 + acc0 * delta_time
    new_pos1 = pos1 + vel1 * delta_time
    new_vel1 = vel1 + acc1 * delta_time

    return torch.cat([new_pos0, new_vel0, new_pos1, new_vel1])


if __name__ == "__main__":
    x_init = torch.Tensor([0.1,0.1,0.1,0.1, 1,1,0.5,0.5])

    x_dims = [4,4]
    x_dim = 8

    u_dims = [2,2]
    u_dim = 4

    cost_funcs = [cost_func_p0, cost_func_p1]
    dynamics_func = dyn

    sim_param = SimulationParams(1,-1,20)
    nl_game = NonlinearGame(dyn, cost_funcs, x_dims, x_dim, u_dims, u_dim, 2)

    S, x, u = generate_simulations(sim_param, nl_game, x_init, 2, 2)

    print(x)