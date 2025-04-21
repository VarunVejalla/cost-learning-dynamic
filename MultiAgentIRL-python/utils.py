import numpy as np

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

def generate_simulations(sim_param, nl_game, x_init, num_sim, num_players):
    """_summary_

    Args:
        sim_param (SimulationParams): _description_
        nl_game (NonlinearGame): _description_
        x_init (one dimensional numpy array): _description_
        num_sim (int): _description_
        num_players (int): _description_
    """
    # TODO: Kyle
    pass

def lqgame_QRE(dynamics, costs, agent_to_functions, num_agents):
    # TODO: John
    pass

def solve_iLQGame(sim_param, nl_game, x_init):
    # TODO: Varun
    pass