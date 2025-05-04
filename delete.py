import torch
from time import perf_counter_ns
import torch.autograd.functional as functional

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

def cost_func(state, action):
    return 0.1 * cost_func1(state, action) + 0.5 * cost_func2(state, action) + 0.9 * cost_func3(state, action)

f = lambda state, action: 0.1 * cost_func1(state, action) + 0.5 * cost_func2(state, action) + 0.9 * cost_func3(state, action)

start_time = perf_counter_ns()

for _ in range(1000):
    state = torch.randn(8, requires_grad=True)
    action = torch.randn(4, requires_grad=True)
    
    result = functional.hessian(cost_func, (state, action))

end_time = perf_counter_ns()

print(end_time-start_time)

# 10489876400