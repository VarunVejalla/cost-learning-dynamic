# utils.jl
using PyCall
using LinearAlgebra
using ForwardDiff

struct NonlinearGame
    state_dim::Int
    ctrl_dim::Int
    state_dims::Vector{Int}
    ctrl_dims::Vector{Int}
    DT::Float64
    radius::Float64
end

py"""
import torch
import torch.nn as nn

class CostNetwork(nn.Module):
    def __init__(self, state_dim, ctrl_dim, num_agents, embedding_dim):
        super(CostNetwork, self).__init__()
        self.input_dim = state_dim + ctrl_dim  # St + all ui,t
        self.num_agents = num_agents
        self.embedding_dim = embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim * num_agents)  # wi for each agent
        )
    
    def forward(self, state, controls):
        x = torch.cat((state, controls), dim=-1)
        w = self.layers(x)  # Output: [w1, w2, w3]
        return w.view(self.num_agents, self.embedding_dim)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, embedding_dim, ctrl_dim_per_agent):
        super(PolicyNetwork, self).__init__()
        self.input_dim = state_dim + embedding_dim  # St + wi
        self.output_dim = ctrl_dim_per_agent
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )
    
    def forward(self, state, w_i):
        x = torch.cat((state, w_i), dim=-1)
        return self.layers(x)

class DynamicsNetwork(nn.Module):
    def __init__(self, embedding_dim, ctrl_dim, state_dim):
        super(DynamicsNetwork, self).__init__()
        self.input_dim = embedding_dim + ctrl_dim  # wi + ui,t
        self.output_dim = state_dim // 3  # Per-agent state (e.g., 4D)
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )
    
    def forward(self, w_i, u_i):
        x = torch.cat((w_i, u_i), dim=-1)
        return self.layers(x)
"""

const torch = pyimport("torch")
CostNetwork = py"CostNetwork"
PolicyNetwork = py"PolicyNetwork"
DynamicsNetwork = py"DynamicsNetwork"

#=======================
define simulation parameters
========================#
struct SimulationParams
    steps      :: Int64
    horizon    :: Float64
    plan_steps :: Int64
end


function SimulationParams(;steps = 60,
                           horizon = 6.0,
                           plan_steps = 10)
    return SimulationParams(steps, horizon, plan_steps)
end


#=======================
define a nonlinear game
========================#
struct NonlinearGame
    # dynamics and cost function
    # dt::Float64
    dynamics_func::Function
    cost_funcs :: Array{Function}

    #
    state_dims :: Array{Int}
    state_dim :: Int     # total dimension of states
    ctrl_dims :: Array{Int}
    ctrl_dim :: Int

    radius :: Float64
end


function NonlinearGame(state_dims::Array{Int},
                       ctrl_dims::Array{Int},
                       DT::Float64;
                       theta::Array{Float64,1})

    state_dim = sum(state_dims)
    ctrl_dim = sum(ctrl_dims)

    radius = 0.25
    function cost1(s)
        # s total dimension: state_dim * 2 + ctrl_dim
        @assert(length(s) == state_dim * 2 + ctrl_dim)

        state = s[1:state_dim]
        ref = s[state_dim+1:state_dim*2]
        control1 = s[state_dim*2+1: state_dim*2+ctrl_dims[1]]
        control2 = s[state_dim*2+ctrl_dims[1]+1: state_dim*2+ctrl_dim]

        dist = sqrt((state[1] - state[5])^2 + (state[2] - state[6])^2) - (2 * radius)

        return theta[1] * (state[1:4] - ref[1:4])' * (state[1:4] - ref[1:4]) +
               theta[2] * control1' * control1 +
               theta[3]/((0.2*dist + 1)^10)
    end

    function cost2(s)
        # s total dimension: state_dim * 2 + ctrl_dim
        @assert(length(s) == state_dim * 2 + ctrl_dim)
        state = s[1:state_dim]
        ref = s[state_dim+1:state_dim*2]
        control1 = s[state_dim*2+1: state_dim*2+ctrl_dims[1]]
        control2 = s[state_dim*2+ctrl_dims[1]+1: state_dim*2+ctrl_dim]

        dist = sqrt((state[1] - state[5])^2 + (state[2] - state[6])^2) - (2 * radius)
        return theta[4] * (state[5:end] - ref[5:end])' * (state[5:end] - ref[5:end]) +
               theta[5] * control2' * control2 +
               theta[6]/((0.2*dist + 1)^10)
    end

    function dynamics_func(s)
        @assert(length(s) == state_dim + ctrl_dim)
        x1, y1, v1, theta1, x2, y2, v2, theta2 = s[1:state_dim]
        acc1, yr1, acc2, yr2 = s[state_dim+1:end]

        x1_new = x1 + v1 * cos(theta1) * DT
        y1_new = y1 + v1 * sin(theta1) * DT
        v1_new = v1+ acc1 * DT
        theta1_new = theta1 + yr1 * DT

        x2_new = x2 + v2 * cos(theta2) * DT
        y2_new = y2 + v2 * sin(theta2) * DT
        v2_new = v2 + acc2 * DT
        theta2_new = theta2 + yr2 * DT

        return [x1_new, y1_new, v1_new, theta1_new, x2_new, y2_new, v2_new, theta2_new]
    end


    return NonlinearGame(dynamics_func, [cost1, cost2], state_dims, state_dim, ctrl_dims, ctrl_dim, radius)
end


function NonlinearGame(state_dims::Array{Int},
                      ctrl_dims::Array{Int};
                      dynamics_func::Function,
                      cost_funcs::Array{Function})
    @assert length(cost_funcs) == length(state_dims)
    @assert length(cost_funcs) == length(ctrl_dims)
    state_dim = sum(state_dims)
    ctrl_dim = sum(ctrl_dims)
    radius = 0.25
    return NonlinearGame(dynamics_func, cost_funcs, state_dims, state_dim, ctrl_dims, ctrl_dim, radius)
end


# return NonlinearGame instance for three player case
function define_game(state_dims, ctrl_dims, DT, theta)
    state_dim = sum(state_dims)
    ctrl_dim = sum(ctrl_dims)
    radius = 0.25
    function dynamics_func(s)
        @assert(length(s) == state_dim + ctrl_dim)
        x1, y1, v1, theta1, x2, y2, v2, theta2, x3, y3, v3, theta3 = s[1:state_dim]
        acc1, yr1, acc2, yr2, acc3, yr3 = s[state_dim+1:end]

        x1_new = x1 + v1 * cos(theta1) * DT
        y1_new = y1 + v1 * sin(theta1) * DT
        v1_new = v1+ acc1 * DT
        theta1_new = theta1 + yr1 * DT

        x2_new = x2 + v2 * cos(theta2) * DT
        y2_new = y2 + v2 * sin(theta2) * DT
        v2_new = v2 + acc2 * DT
        theta2_new = theta2 + yr2 * DT

        x3_new = x3 + v3 * cos(theta3) * DT
        y3_new = y3 + v3 * sin(theta3) * DT
        v3_new = v3 + acc3 * DT
        theta3_new = theta3 + yr3 * DT

        return [x1_new, y1_new, v1_new, theta1_new, x2_new, y2_new, v2_new, theta2_new, x3_new, y3_new, v3_new, theta3_new]
    end

    function cost1(s)

        @assert(length(s) == state_dim * 2 + ctrl_dim)
        state_dim1 = state_dims[1]
        state_dim2 = state_dims[2]
        ctrl_dim1 = ctrl_dims[1]

        state = s[1:state_dim]
        ref = s[state_dim+1:state_dim*2]

        ego_state = state[1:state_dim1]
        ego_ref = ref[1:state_dim1]

        control1 = s[state_dim*2+1:state_dim*2+ctrl_dim1]

        dist12 = sqrt((ego_state[1] - state[state_dim1+1])^2 + (ego_state[2] - state[state_dim1+2])^2) - (2 * radius)
        dist13 = sqrt((ego_state[1] - state[state_dim1+state_dim2+1])^2 + (ego_state[2] - state[state_dim1+state_dim2+2])^2) - (2 * radius)

        return theta[1] * (ego_state - ego_ref)' * (ego_state - ego_ref) +
               theta[2] * control1' * control1 +
               theta[3]/((0.2*dist12 + 1)^10) +
               theta[3]/((0.2*dist13 + 1)^10)
    end

    function cost2(s)
        @assert(length(s) == state_dim * 2 + ctrl_dim)
        state_dim1 = state_dims[1]
        state_dim2 = state_dims[2]
        ctrl_dim1 = ctrl_dims[1]
        ctrl_dim2 = ctrl_dims[2]

        state = s[1:state_dim]
        ref = s[state_dim+1:state_dim*2]
        # control1 = s[state_dim*2+1:state_dim*2+ctrl_dim1]

        ego_state = state[state_dim1+1:state_dim1+state_dim2]
        ego_ref = ref[state_dim1+1:state_dim1+state_dim2]

        control2 = s[state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2]

        dist21 = sqrt((ego_state[1] - state[1])^2 + (ego_state[2] - state[1])^2) - (2 * radius)
        dist23 = sqrt((ego_state[1] - state[state_dim1+state_dim2+1])^2 + (ego_state[2] - state[state_dim1+state_dim2+2])^2) - (2 * radius)

        return theta[4] * (ego_state - ego_ref)' * (ego_state - ego_ref) +
               theta[5] * control2' * control2 +
               theta[6]/((0.2*dist21 + 1)^10) +
               theta[6]/((0.2*dist23 + 1)^10)
    end

    function cost3(s)
        @assert(length(s) == state_dim * 2 + ctrl_dim)
        state_dim1 = state_dims[1]
        state_dim2 = state_dims[2]
        ctrl_dim1 = ctrl_dims[1]
        ctrl_dim2 = ctrl_dims[2]

        state = s[1:state_dim]
        ref = s[state_dim+1:state_dim*2]

        ego_state = state[state_dim1+state_dim2+1:end]
        ego_ref = ref[state_dim1+state_dim2+1:end]
        control3 = s[state_dim*2+ctrl_dim1+ctrl_dim2+1:end]

        dist31 = sqrt((ego_state[1] - state[1])^2 + (ego_state[2] - state[2])^2) - (2 * radius)
        dist32 = sqrt((ego_state[1] - state[state_dim1+1])^2 + (ego_state[2] - state[state_dim1+1])^2) - (2 * radius)

        return theta[7] * (ego_state - ego_ref)' * (ego_state - ego_ref) +
               theta[8] * control3' * control3 +
               theta[9]/((0.2*dist31 + 1)^10) +
               theta[9]/((0.2*dist32 + 1)^10)
    end



    return NonlinearGame(state_dims, ctrl_dims;
                    dynamics_func=dynamics_func,
                    cost_funcs=[cost1, cost2, cost3])
end

function get_feature_counts_three(game, x_trajectories, u_trajectories, x_ref, feature_k)
    println("get_feature_counts_three")
    println("x_trajectories: ", x_trajectories)
    println("size(x_trajectories): ", size(x_trajectories))
    println("u_trajectories: ", u_trajectories)
    println("size(u_trajectories): ", size(u_trajectories))
    println("x_ref: ", x_ref)
    println("feature_k: ", feature_k)
    feature_counts = zeros(feature_k)
    num = length(x_trajectories)
    state_dim1 = game.state_dims[1]
    state_dim2 = game.state_dims[2]
    ctrl_dim1 = game.ctrl_dims[1]
    ctrl_dim2 = game.ctrl_dims[2]
    radius = game.radius

    for i = 1:num
        xtraj = x_trajectories[i]
        utraj = u_trajectories[i]
        steps = size(utraj)[1]
        for t = 1:steps+1
            # Extract states for each agent
            state1 = xtraj[t, 1:state_dim1]               # Agent 1
            state2 = xtraj[t, 1+state_dim1:state_dim1+state_dim2]  # Agent 2
            state3 = xtraj[t, 1+state_dim1+state_dim2:end]  # Agent 3
            # Extract reference states
            ref1 = x_ref[t, 1:state_dim1]
            ref2 = x_ref[t, 1+state_dim1:state_dim1+state_dim2]
            ref3 = x_ref[t, 1+state_dim1+state_dim2:end]
            # Extract controls (if not final step)
            if t <= steps
                control1 = utraj[t, 1:ctrl_dim1]
                control2 = utraj[t, 1+ctrl_dim1:ctrl_dim1+ctrl_dim2]
                control3 = utraj[t, 1+ctrl_dim1+ctrl_dim2:end]
            end
            # Compute pairwise distances minus radii
            dist12 = sqrt((state1[1] - state2[1])^2 + (state1[2] - state2[2])^2) - (2 * radius)
            dist13 = sqrt((state1[1] - state3[1])^2 + (state1[2] - state3[2])^2) - (2 * radius)
            dist23 = sqrt((state2[1] - state3[1])^2 + (state2[2] - state3[2])^2) - (2 * radius)

            # Manually defined features
            feature_counts[1] += (state1 - ref1)' * (state1 - ref1)  # Agent 1: Tracking error
            feature_counts[3] += 1.0/((0.2*dist12 + 1)^10) + 1.0/((0.2*dist13 + 1)^10)  # Agent 1: Safety
            feature_counts[4] += (state2 - ref2)' * (state2 - ref2)  # Agent 2: Tracking error
            feature_counts[6] += 1.0/((0.2*dist12 + 1)^10) + 1.0/((0.2*dist23 + 1)^10)  # Agent 2: Safety
            feature_counts[7] += (state3 - ref3)' * (state3 - ref3)  # Agent 3: Tracking error
            feature_counts[9] += 1.0/((0.2*dist13 + 1)^10) + 1.0/((0.2*dist23 + 1)^10)  # Agent 3: Safety
            if t <= steps
                feature_counts[2] += control1' * control1  # Agent 1: Control effort
                feature_counts[5] += control2' * control2  # Agent 2: Control effort
                feature_counts[8] += control3' * control3  # Agent 3: Control effort 
            end
        end
    end

    avg_feature_counts = 1.0/num * feature_counts
    return avg_feature_counts
end

function get_feature_counts(game, x_trajectories, u_trajectories, x_ref, feature_k)

    num = length(x_trajectories)
    state_dim1 = game.state_dims[1]
    ctrl_dim1 = game.ctrl_dims[1]
    radius = game.radius

    feature_counts = zeros(feature_k, num)

    for i = 1:num
        xtraj = x_trajectories[i]
        utraj = u_trajectories[i]
        steps = size(utraj)[1]
        for t = 1:steps+1
            state1 = xtraj[t, 1:state_dim1]
            state2 = xtraj[t, 1+state_dim1:end]
            ref1 = x_ref[t, 1:state_dim1]
            ref2 = x_ref[t, 1+state_dim1:end]
            if t <= steps
                control1 = utraj[t, 1:ctrl_dim1]
                control2 = utraj[t, 1+ctrl_dim1:end]
            end
            dist = sqrt((state1[1] - state2[1])^2 + (state1[2] - state2[2])^2) - (2 * radius)

            # Manually defined features
            feature_counts[1, i] += (state1 - ref1)' * (state1 - ref1)  # Agent 1: Tracking error
            feature_counts[3, i] += 1.0/((0.2*dist + 1)^10)             # Agent 1: Safety
            feature_counts[4, i] += (state2 - ref2)' * (state2 - ref2)  # Agent 2: Tracking error
            feature_counts[6, i] += 1.0/((0.2*dist + 1)^10)             # Agent 2: Safety
            if t <= steps
                feature_counts[2, i] += control1' * control1            # Agent 1: Control effort
                feature_counts[5, i] += control2' * control2            # Agent 2: Control effort
            end
        end
    end

    avg_feature_counts = 1.0/num * sum(feature_counts, dims=2)
    return avg_feature_counts, feature_counts
end

function define_game(state_dims, ctrl_dims, DT, theta)
    state_dim = sum(state_dims)
    ctrl_dim = sum(ctrl_dims)
    radius = 0.5
    return NonlinearGame(state_dim, ctrl_dim, state_dims, ctrl_dims, DT, radius)
end

# Compute costs and policies
function compute_costs_and_policies(game, state, controls, cost_network, policy_networks)
    state_torch = torch.tensor(state, dtype=torch.float32)
    controls_torch = torch.tensor(controls, dtype=torch.float32)
    w = cost_network(state_torch, controls_torch)  # [num_agents, embedding_dim]
    u_pred = zeros(game.ctrl_dim)
    costs = zeros(game.num_agents)
    for i = 1:game.num_agents
        u_pred[(i-1)*game.ctrl_dims[i]+1:i*game.ctrl_dims[i]] = policy_networks[i](state_torch, w[i]).detach().numpy()
        costs[i] = torch.sum(w[i] * w[i])  # Example cost: ||wi||^2
    end
    return w, u_pred, costs
end

function dynamics_nn(game, state, controls, w, dynamics_networks)
    state_torch = torch.tensor(state, dtype=torch.float32)
    controls_torch = torch.tensor(controls, dtype=torch.float32)
    next_state = zeros(game.state_dim)
    for i = 1:game.num_agents
        start_idx = sum(game.state_dims[1:i-1]) + 1
        end_idx = sum(game.state_dims[1:i])
        u_i = controls_torch[(i-1)*game.ctrl_dims[i]+1:i*game.ctrl_dims[i]]
        next_state[start_idx:end_idx] = dynamics_networks[i](w[i], u_i).detach().numpy()
    end
    return next_state
end

# Generate simulations with NN policy
function generate_simulations_nn(sim_param, nl_game, x_init, traj_ref, num_sim, cost_network, policy_networks, dynamics_networks)
    steps = sim_param.steps
    state_dim = nl_game.state_dim
    ctrl_dim = nl_game.ctrl_dim
    x_trajectories = []
    u_trajectories = []

    for _ in 1:num_sim
        x_traj = zeros(steps + 1, state_dim)
        u_traj = zeros(steps, ctrl_dim)
        x_traj[1, :] = x_init
        
        for t = 1:steps
            w, u_pred, _ = compute_costs_and_policies(nl_game, x_traj[t, :], u_traj[max(1, t-1), :], cost_network, policy_networks)
            u_traj[t, :] = u_pred
            x_traj[t + 1, :] = dynamics_nn(nl_game, x_traj[t, :], u_traj[t, :], w, dynamics_networks)
        end
        push!(x_trajectories, x_traj)
        push!(u_trajectories, u_traj)
    end
    
    results = (state_trajectories=x_trajectories, ctrl_trajectories=u_trajectories)
    return results, x_trajectories, u_trajectories
end

#=======================
an iterative LQ game solver
input: sim_param::SimulationParams,
       nl_game::NonlinearGame,
       x_init::Array{Float64},
       traj_ref::Array{Float64}
========================#
function solve_iLQGame(;sim_param::SimulationParams,
                       nl_game::NonlinearGame,
                       x_init::Array{Float64},
                       traj_ref::Array{Float64})

    num_player = length(nl_game.ctrl_dims)

    plan_steps = sim_param.plan_steps
    state_dim = nl_game.state_dim
    ctrl_dim = nl_game.ctrl_dim
    ctrl_dim1 = nl_game.ctrl_dims[1]
    ctrl_dim2 = nl_game.ctrl_dims[2]



    # ###########################
    # initialize the iteration (forward simulation)
    # ###########################
    x_trajectory = zeros(plan_steps+1, state_dim)
    x_trajectory[1,:] = x_init
    u_trajectory = zeros(plan_steps, ctrl_dim)

    # forward simulation
    for t=1:plan_steps
        x_trajectory[t+1, :] = nl_game.dynamics_func([x_trajectory[t, :]; u_trajectory[t, :]])
    end

    ##################
    # iteration
    ################
    if num_player == 2
        tol = 1.0
        itr = 1
        err = 10
        max_itr = 50

        x_trajectory_prev = x_trajectory
        u_trajectory_prev = u_trajectory
        N1, N2 = [], []
        alpha1, alpha2 = [], []
        cov1, cov2 = [], []
        while (abs(err) > tol && itr < max_itr)
            # if plot_flag
            #     ax.plot(x_trajectory[:,1], x_trajectory[:,2], "yellow")
            #     ax.plot(x_trajectory[:,5], x_trajectory[:,6], "yellow")
            # end

            # println(" at iteration ", itr, " and error ", err)

            ###########################
            #  linear quadratic approximation
            # ###########################
            Dynamics = Dict{String, Array{Array{Float64}}}()
            Dynamics["A"] = []
            Dynamics["B1"] = []
            Dynamics["B2"] = []

            Costs = Dict{String, Array{Array{Float64}}}()
            Costs["Q1"] = []
            Costs["l1"] = []
            Costs["Q2"] = []
            Costs["l2"] = []
            Costs["R11"] = []
            Costs["R12"] = []
            Costs["R21"] = []
            Costs["R22"] = []

            for t=1:plan_steps
                jac = ForwardDiff.jacobian(nl_game.dynamics_func, [x_trajectory_prev[t,:]; u_trajectory_prev[t,:]])
                A = jac[:,1:state_dim]
                B1 = jac[:, state_dim+1:state_dim+nl_game.ctrl_dims[1]]
                B2 = jac[:, state_dim+nl_game.ctrl_dims[1]+1:end]

                push!(Dynamics["A"], A)
                push!(Dynamics["B1"], B1)
                push!(Dynamics["B2"], B2)


                grads1 = ForwardDiff.gradient(nl_game.cost_funcs[1], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])
                hess1 = ForwardDiff.hessian(nl_game.cost_funcs[1], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])
                grads2 = ForwardDiff.gradient(nl_game.cost_funcs[2], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])
                hess2 = ForwardDiff.hessian(nl_game.cost_funcs[2], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])

                Q1 = hess1[1:state_dim, 1:state_dim]
                min_eig_Q1 = minimum(eigvals(Q1))
                if min_eig_Q1 <= 0.0
                    Q1 += (abs(min_eig_Q1) + 1e-3) * I
                end
                Q2 = hess2[1:state_dim, 1:state_dim]
                min_eig_Q2 = minimum(eigvals(Q2))
                if min_eig_Q2 <= 0.0
                    Q2 += (abs(min_eig_Q2) + 1e-3) * I
                end
                push!(Costs["Q1"], Q1)
                push!(Costs["Q2"], Q2)
                push!(Costs["l1"], grads1[1:state_dim])
                push!(Costs["l2"], grads2[1:state_dim])

                push!(Costs["R11"], hess1[state_dim*2+1:state_dim*2+ctrl_dim1, state_dim*2+1:state_dim*2+ctrl_dim1])
                push!(Costs["R12"], hess1[state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim,
                                          state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim])
                push!(Costs["R21"], hess2[state_dim*2+1:state_dim*2+ctrl_dim1, state_dim*2+1:state_dim*2+ctrl_dim1])
                push!(Costs["R22"], hess2[state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim,
                                          state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim])
            end
            grads1 = ForwardDiff.gradient(nl_game.cost_funcs[1], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])
            hess1 = ForwardDiff.hessian(nl_game.cost_funcs[1], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])
            grads2 = ForwardDiff.gradient(nl_game.cost_funcs[2], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])
            hess2 = ForwardDiff.hessian(nl_game.cost_funcs[2], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])


            Q1 = hess1[1:state_dim, 1:state_dim]
            min_eig_Q1 = minimum(eigvals(Q1))
            if min_eig_Q1 <= 0.0
                Q1 += (abs(min_eig_Q1) + 1e-3) * I
            end
            Q2 = hess2[1:state_dim, 1:state_dim]
            min_eig_Q2 = minimum(eigvals(Q2))
            if min_eig_Q2 <= 0.0
                Q2 += (abs(min_eig_Q2) + 1e-3) * I
            end
            push!(Costs["Q1"], Q1)
            push!(Costs["Q2"], Q2)
            push!(Costs["l1"], grads1[1:state_dim])
            push!(Costs["l2"], grads2[1:state_dim])


            # ###########################
            # backward computation
            # ###########################
            # backtrack on gamma parameters
            N1, N2, alpha1, alpha2, cov1, cov2 = lqgame_QRE(Dynamics, Costs)


            ###########################
            # forward simulation
            # ###########################
            step_size = 1.0
            done = false
            while !done
                u_trajectory = zeros(plan_steps, ctrl_dim)
                x_trajectory = zeros(plan_steps + 1, state_dim)
                x_trajectory[1,:] = x_init
                for t=1:plan_steps
                    u_trajectory[t,:] = [(-N1[end-t+1] * (x_trajectory[t, :] - x_trajectory_prev[t,:]) - alpha1[end-t+1] * step_size)...,
                                         (-N2[end-t+1] * (x_trajectory[t, :] - x_trajectory_prev[t,:]) - alpha2[end-t+1] * step_size)...] +
                                         u_trajectory_prev[t, :]
                    x_trajectory[t+1, :] = nl_game.dynamics_func([x_trajectory[t, :]; u_trajectory[t,:]])
                end

                if maximum(abs.(x_trajectory - x_trajectory_prev)) > 1.0
                    step_size /= 2
                else
                    done = true
                end

            end

            ###########################
            # book keeping and convergence test
            # ###########################
            err = sum(abs.(x_trajectory_prev - x_trajectory))
            itr += 1
            x_trajectory_prev = x_trajectory
            u_trajectory_prev = u_trajectory
        end


        return [N1, N2], [alpha1, alpha2], [cov1, cov2], x_trajectory_prev, u_trajectory_prev
    else
        tol = 1.0
        itr = 1
        err = 10
        max_itr = 50

        x_trajectory_prev = x_trajectory
        u_trajectory_prev = u_trajectory
        N1, N2, N3 = [], [], []
        alpha1, alpha2, alpha3 = [], [], []
        cov1, cov2, cov3 = [], [], []
        while (abs(err) > tol && itr < max_itr)
            # if plot_flag
            #     ax.plot(x_trajectory[:,1], x_trajectory[:,2], "yellow")
            #     ax.plot(x_trajectory[:,5], x_trajectory[:,6], "yellow")
            # end

            # println(" at iteration ", itr, " and error ", err)

            ###########################
            #  linear quadratic approximation
            # ###########################
            Dynamics = Dict{String, Array{Array{Float64}}}()
            Dynamics["A"] = []
            Dynamics["B1"] = []
            Dynamics["B2"] = []
            Dynamics["B3"] = []

            Costs = Dict{String, Array{Array{Float64}}}()
            Costs["Q1"] = []
            Costs["l1"] = []
            Costs["Q2"] = []
            Costs["l2"] = []
            Costs["Q3"] = []
            Costs["l3"] = []
            Costs["R11"] = []
            Costs["R12"] = []
            Costs["R13"] = []
            Costs["R21"] = []
            Costs["R22"] = []
            Costs["R23"] = []
            Costs["R31"] = []
            Costs["R32"] = []
            Costs["R33"] = []

            for t=1:plan_steps
                jac = ForwardDiff.jacobian(nl_game.dynamics_func, [x_trajectory_prev[t,:]; u_trajectory_prev[t,:]])
                A = jac[:,1:state_dim]
                B1 = jac[:, state_dim+1:state_dim+ctrl_dim1]
                B2 = jac[:, state_dim+ctrl_dim1+1:state_dim+ctrl_dim1+ctrl_dim2]
                B3 = jac[:, state_dim+ctrl_dim1+ctrl_dim2+1:end]
                push!(Dynamics["A"], A)
                push!(Dynamics["B1"], B1)
                push!(Dynamics["B2"], B2)
                push!(Dynamics["B3"], B3)

                grads1 = ForwardDiff.gradient(nl_game.cost_funcs[1], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])
                hess1 = ForwardDiff.hessian(nl_game.cost_funcs[1], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])
                grads2 = ForwardDiff.gradient(nl_game.cost_funcs[2], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])
                hess2 = ForwardDiff.hessian(nl_game.cost_funcs[2], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])
                grads3 = ForwardDiff.gradient(nl_game.cost_funcs[3], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])
                hess3 = ForwardDiff.hessian(nl_game.cost_funcs[3], [x_trajectory_prev[t,:]; traj_ref[t,:]; u_trajectory[t,:]])

                Q1 = hess1[1:state_dim, 1:state_dim]
                min_eig_Q1 = minimum(abs.(eigvals(Q1)))
                if min_eig_Q1 <= 0.0
                    Q1 += (abs(min_eig_Q1) + 1e-3) * I
                end
                Q2 = hess2[1:state_dim, 1:state_dim]
                min_eig_Q2 = minimum(eigvals(Q2))
                if min_eig_Q2 <= 0.0
                    Q2 += (abs(min_eig_Q2) + 1e-3) * I
                end

                Q3 = hess3[1:state_dim, 1:state_dim]
                min_eig_Q3 = minimum(abs.(eigvals(Q3)))
                if min_eig_Q3 <= 0.0
                    Q3 += (abs(min_eig_Q3) + 1e-3) * I
                end
                push!(Costs["Q1"], Q1)
                push!(Costs["Q2"], Q2)
                push!(Costs["Q3"], Q3)
                push!(Costs["l1"], grads1[1:state_dim])
                push!(Costs["l2"], grads2[1:state_dim])
                push!(Costs["l3"], grads3[1:state_dim])

                push!(Costs["R11"], hess1[state_dim*2+1:state_dim*2+ctrl_dim1, state_dim*2+1:state_dim*2+ctrl_dim1])
                push!(Costs["R12"], hess1[state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2,
                                          state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2])
                push!(Costs["R13"], hess1[state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim,
                                          state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim])
                push!(Costs["R21"], hess2[state_dim*2+1:state_dim*2+ctrl_dim1, state_dim*2+1:state_dim*2+ctrl_dim1])
                push!(Costs["R22"], hess2[state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2,
                                          state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2])
                push!(Costs["R23"], hess2[state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim,
                                          state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim])
                push!(Costs["R31"], hess3[state_dim*2+1:state_dim*2+ctrl_dim1, state_dim*2+1:state_dim*2+ctrl_dim1])
                push!(Costs["R32"], hess3[state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2,
                                          state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2])
                push!(Costs["R33"], hess3[state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim,
                                          state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim])
            end
            grads1 = ForwardDiff.gradient(nl_game.cost_funcs[1], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])
            hess1 = ForwardDiff.hessian(nl_game.cost_funcs[1], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])
            grads2 = ForwardDiff.gradient(nl_game.cost_funcs[2], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])
            hess2 = ForwardDiff.hessian(nl_game.cost_funcs[2], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])
            grads3 = ForwardDiff.gradient(nl_game.cost_funcs[3], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])
            hess3 = ForwardDiff.hessian(nl_game.cost_funcs[3], [x_trajectory_prev[end,:]; traj_ref[end,:]; u_trajectory[end,:]])

            Q1 = hess1[1:state_dim, 1:state_dim]
            min_eig_Q1 = minimum(abs.(eigvals(Q1)))
            if min_eig_Q1 <= 0.0
                Q1 += (abs(min_eig_Q1) + 1e-3) * I
            end
            Q2 = hess2[1:state_dim, 1:state_dim]
            min_eig_Q2 = minimum(eigvals(Q2))
            if min_eig_Q2 <= 0.0
                Q2 += (abs(min_eig_Q2) + 1e-3) * I
            end
            Q3 = hess3[1:state_dim, 1:state_dim]
            min_eig_Q3 = minimum(abs.(eigvals(Q3)))
            if min_eig_Q3 <= 0.0
                Q3 += (abs(min_eig_Q3) + 1e-3) * I
            end
            push!(Costs["Q1"], Q1)
            push!(Costs["Q2"], Q2)
            push!(Costs["Q3"], Q3)
            push!(Costs["l1"], grads1[1:state_dim])
            push!(Costs["l2"], grads2[1:state_dim])
            push!(Costs["l3"], grads3[1:state_dim])


            # ###########################
            # backward computation
            # ###########################
            # backtrack on gamma parameters
            N1, N2, N3, alpha1, alpha2, alpha3, cov1, cov2, cov3 = lqgame_QRE_3player(Dynamics, Costs)

            ###########################
            # forward simulation
            # ###########################
            step_size = 1.0
            done = false
            while !done
                u_trajectory = zeros(plan_steps, ctrl_dim)
                x_trajectory = zeros(plan_steps + 1, state_dim)
                x_trajectory[1,:] = x_init
                for t=1:plan_steps
                    delta_x = x_trajectory[t, :] - x_trajectory_prev[t,:]
                    u_trajectory[t,1:ctrl_dim1] =
                                (-N1[end-t+1] * delta_x - alpha1[end-t+1] * step_size) + u_trajectory_prev[t,1:ctrl_dim1]
                    u_trajectory[t,ctrl_dim1+1:ctrl_dim1+ctrl_dim2] =
                                (-N2[end-t+1] * delta_x - alpha2[end-t+1] * step_size) + u_trajectory_prev[t,ctrl_dim1+1:ctrl_dim1+ctrl_dim2]
                    u_trajectory[t,ctrl_dim1+ctrl_dim2+1:ctrl_dim] =
                                (-N3[end-t+1] * delta_x - alpha3[end-t+1] * step_size) + u_trajectory_prev[t,ctrl_dim1+ctrl_dim2+1:ctrl_dim]
                    x_trajectory[t+1, :] = nl_game.dynamics_func([x_trajectory[t, :]; u_trajectory[t,:]])
                end

                if maximum(abs.(x_trajectory - x_trajectory_prev)) > 1.0
                    step_size /= 2
                else
                    done = true
                end

            end

            ###########################
            # book keeping and convergence test
            # ###########################
            err = sum(abs.(x_trajectory_prev - x_trajectory))
            itr += 1
            x_trajectory_prev = x_trajectory
            u_trajectory_prev = u_trajectory
        end


        return [N1, N2, N3], [alpha1, alpha2, alpha3], [cov1, cov2, cov3], x_trajectory_prev, u_trajectory_prev
    end

end

struct SimulationResults
    state_trajectories :: Array
    ctrl_trajectories :: Array
end

function generate_simulations(;sim_param::SimulationParams,
                               nl_game::NonlinearGame,
                               x_init::Array{Float64},
                               traj_ref::Array{Float64},
                               num_sim::Int,
                               plot_flag::Bool=false)
    num_player = length(nl_game.ctrl_dims)
    steps = sim_param.steps
    plan_steps = sim_param.plan_steps
    state_dim = nl_game.state_dim
    ctrl_dim = nl_game.ctrl_dim
    x_trajectories, u_trajectories = [], []
    x_trajectories_data = zeros(num_sim, steps+1, state_dim)
    u_trajectories_data = zeros(num_sim, steps+1, ctrl_dim)

    # receiding horizon planning in one simulation
    # need to replan online because of stochastic policy
    # fig, ax = subplots(1, 1)
    for i = 1:num_sim
        if i%3 == 0
           println(" --- generating sim number : ", i, " ---")
        end
        x_history = zeros(steps+1, state_dim)
        x_history[1,:] = x_init
        u_history = zeros(steps, ctrl_dim)

        for t = 1:steps
            if plot_flag
                   ax.clear()
                   ax.scatter(-4, 0, marker="X", s=100, c="y")
                   ax.scatter(0, -4, marker="X", s=100, c="m")
                   ax.scatter(traj_ref[t,1], traj_ref[t,2], alpha=0.1)
                   ax.scatter(traj_ref[t,5], traj_ref[t,6], alpha=0.1)

                   ax.set_xlabel("x")
                   ax.set_ylabel("y")
                   ax.set_xlim(-5, 5)
                   ax.set_xticks([-3,-1, 1, 3])
                   ax.set_ylim(-5, 5)
                   ax.set_yticks([-3,-1, 1, 3])
                   pause(0.1)
            end

            Nlist_all, alphalist_all, cov_all, x_nominal, u_nominal =
                    solve_iLQGame(sim_param=sim_param, nl_game=nl_game, x_init=x_history[t,:], traj_ref=traj_ref[t:t+plan_steps,:])

            delta_x = x_history[t, :] - x_nominal[1,:]
            u_dists = []
            for ip = 1:num_player
                u_mean = -Nlist_all[ip][end] * delta_x - alphalist_all[ip][end]
                # u_dist = MvNormal(u_mean, Symmetric(cov_all[ip][end]))
                u_dist = MvNormal(u_mean, Symmetric(Matrix(I, 2, 2)))
                push!(u_dists, u_dist)
            end

            control = []
            for u_dist in u_dists
                append!(control, rand(u_dist))
            end

            u_history[t,:] = control + u_nominal[1, :]
            x_history[t+1, :] = nl_game.dynamics_func([x_history[t, :]; u_history[t, :]])
            if plot_flag
                ax.plot(traj_ref[t:t+plan_steps,1], traj_ref[t:t+plan_steps,2], "purple")
                ax.plot(traj_ref[t:t+plan_steps,5], traj_ref[t:t+plan_steps,6], "purple")
                ax.plot(x_nominal[:,1], x_nominal[:,2], "red")
                ax.plot(x_nominal[:,5], x_nominal[:,6], "red")
                pause(0.1)
            end
       end

       push!(x_trajectories, x_history)
       push!(u_trajectories, u_history)
       x_trajectories_data[i,:,:] = x_history
       u_trajectories_data[i,1:steps,:] = u_history
    end


    return SimulationResults(x_trajectories, u_trajectories), x_trajectories_data, u_trajectories_data
end
