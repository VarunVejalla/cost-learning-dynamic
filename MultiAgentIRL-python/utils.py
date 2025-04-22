import numpy as np
import torch

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
    
    def __init__(self, dynamics_func, cost_funcs, x_dims, x_dim, u_dims, u_dim, agent_to_functions, num_agents):
        """_summary_

        Args:
            dynamics_func (function from whole state and all actions to new state): _description_
            cost_funcs (functions from ): _description_
            x_dims (_type_): _description_
            x_dim (_type_): _description_
            u_dims (_type_): _description_
            u_dim (_type_): _description_
            agent_to_functions (_type_): _description_
            num_agents (_type_): _description_
        """
        self.dynamics = dynamics_func
        
        self.x_dims = x_dims
        self.x_dim = x_dim
        self.u_dims = u_dims
        self.u_dim = u_dim
        self.cost_funcs = cost_funcs
        self.agent_to_functions = agent_to_functions
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
            # TODO
            continue
        
        x_trajs.append(x_history)
        u_trajs.append(u_history)
        x_trajs_data[i] = x_history
        u_trajs_data[i, :steps] = u_history
    
    return SimulationResults(x_trajs, u_trajs), x_trajs_data, u_trajs_data

def lqgame_QRE(dynamic_dicts, cost_dicts):
    # TODO: I have no idea how to test this, but I do believe
    # that I implemented it pretty rigourously identical to the 
    # paper it's definitions. If you would like me to explain 
    # something let me know! - John

    A = dynamic_dicts["A"]
    B = dynamic_dicts["B"]
    Q = cost_dicts["Q"]
    l = cost_dicts["l"]
    R = cost_dicts["R"]
    
    num_agents = len(B)
    
    T = len(A)
    n = B[0][0].shape[0]
    
    m = []
    for i in range(num_agents):
        if len(B[i][0].shape) == 1:
            m.append(1)
        else:
            m.append(B[i][0].shape[1])
    
    P = [[] for _ in range(num_agents)]
    alpha = [[] for _ in range(num_agents)]
    cov = [[] for _ in range(num_agents)]
    
    Z = [[] for _ in range(num_agents)]
    F = []
    zeta = [[] for _ in range(num_agents)]
    beta = []
    
    # for doing the linear quadratic game backwards passes, 
    # initialize the terminal state to the terminal costs and
    # then begin to iterate backwards for T iterations
    def lqgame_QRE(dynamic_dicts, cost_dicts):
    A = dynamic_dicts["A"]
    B = dynamic_dicts["B"]
    Q = cost_dicts["Q"]
    l = cost_dicts["l"]
    R = cost_dicts["R"]
    
    num_agents = len(B)
    T = len(A)
    n = B[0][0].shape[0]
    
    m = [B[i][0].shape[1] if len(B[i][0].shape) > 1 else 1 for i in range(num_agents)]
    
    P = [[] for _ in range(num_agents)]
    alpha = [[] for _ in range(num_agents)]
    cov = [[] for _ in range(num_agents)]
    Z = [[] for _ in range(num_agents)]
    zeta = [[] for _ in range(num_agents)]
    F = []
    beta = []

    for i in range(num_agents):
        Z[i].append(Q[i][-1])
        zeta[i].append(l[i][-1])
        P[i].append(torch.zeros((m[i], n)))
        alpha[i].append(torch.zeros((m[i], 1)))  # Adjust shape if needed

    for t in range(T - 1, -1, -1):
        F_t = A[t]
        beta_t = 0

        for i in range(num_agents):
            Z_next = Z[i][-1]
            zeta_next = zeta[i][-1]

            M1 = R[i][i][t] + B[i][t].T @ Z_next @ B[i][t]
            M2 = torch.zeros((m[i], n))
            for j in range(num_agents):
                if j == i: continue
                M2 += B[j][t].T @ P[j][-1]
            M2 = B[i][t].T @ Z_next @ M2
            rhs1 = B[i][t].T @ Z_next @ A[t]
            P_t = torch.linalg.solve(M1, rhs1 - M2)
            P[i].append(P_t)

            M4 = torch.zeros((m[i], 1))
            for j in range(num_agents):
                if j == i: continue
                M4 += B[j][t].T @ alpha[j][-1]
            M4 = B[i][t].T @ Z_next @ M4
            rhs2 = B[i][t].T @ zeta_next
            alpha_t = torch.linalg.solve(M1, rhs2 - M4)
            alpha[i].append(alpha_t)

            F_t -= B[i][t] @ P_t
            beta_t -= B[i][t] @ alpha_t

        F.append(F_t)
        beta.append(beta_t)

        for i in range(num_agents):
            Z_next = Z[i][-1]
            zeta_next = zeta[i][-1]

            Z_it = F_t.T @ Z_next @ F_t
            zeta_it = F_t.T @ (zeta_next + Z_next @ beta_t)

            for j in range(num_agents):
                Z_it += P[j][-1].T @ R[i][j][t] @ P[j][-1]
                zeta_it += P[j][-1].T @ R[i][j][t] @ alpha[j][-1]

            Z_it += Q[i][t]
            zeta_it += l[i][t]

            Z[i].append(Z_it)
            zeta[i].append(zeta_it)

            cov_t = torch.linalg.inv(R[i][i][t] + B[i][t].T @ Z_next @ B[i][t])
            cov[i].append(cov_t)

    # Optional: reverse time order
    P = [p[::-1] for p in P]
    alpha = [a[::-1] for a in alpha]
    cov = [c[::-1] for c in cov]

    return P, alpha, cov


def solve_iLQGame(sim_param:SimulationParams, nl_game:NonlinearGame, x_init:torch.Tensor):
    # TODO: Varun
    num_player = len(nl_game.u_dims)
    
    
    plan_steps = sim_param.plan_steps
    x_dim = nl_game.x_dim
    u_dim = nl_game.u_dim
    u_dims = nl_game.udims
    
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
    
    while abs(err) > tol and itr < max_itr:
        Dynamics = {}
        Dynamics["A"] = []
        Dynamics["B"] = [[] for _ in range(num_player)]

        Costs = {}
        Costs["Q"] = [[] for _ in range(num_player)]
        Costs["l"] = [[] for _ in range(num_player)]
        Costs["R"] = [[[] for _ in range(num_player)] for _ in range(num_player)]
        
        for t in range(plan_steps):
            # TODO
            continue
        
        # TODO
        
        N, alpha, cov = lqgame_QRE(Dynamics, Costs)
        
        step_size = 1.0
        done = False
        while not done:
            u_trajectory = torch.zeros((plan_steps, u_dim))
            x_trajectory = torch.zeros((plan_steps, x_dim))
            x_trajectory[0,:] = x_init
            for t in range(plan_steps):
                delta_x = x_trajectory[t] - x_trajectory_prev[t]
                
                # TODO
                reverse_index = "fsda"
                
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
    
    
    pass