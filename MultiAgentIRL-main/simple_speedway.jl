# file:     simple_speedway.jl
# author:   mingyuw@stanford.edu

# Nonlinear game with 3 players

# Set matplotlib backend for plotting
ENV["MPLBACKEND"] = "tkagg"
using PyPlot
using ForwardDiff, LinearAlgebra
using Distributions, StatsBase
using Dates
using JLD2, FileIO
using Pkg
using PyCall
using Pickle

# Include external utility files
include("lqgame.jl")  # Contains game-solving functions (e.g., lqgame_QRE)
include("utils.jl")   # Contains helper functions (e.g., generate_simulations_nn)

# Define Python function to load pickle files via PyCall
py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""
load_pickle = py"load_pickle"

# Learning rate decay function
function rate(lr0, itr)
    return lr0 / (1.0 + 0.1 * itr)  # Decays learning rate over iterations
end

# Multi-agent IRL function for 3-player nonlinear game
function ma_irl(;sync_update=true, single_update=10, scale=true, eta=0.0001, plot=true, max_itr=5000, sample_size=100, dem_num=200)
    println("==========================")

    data = load_pickle("data/demo_trajectories_rld.pickle")
    x_trajectories = data["x"]
    u_trajectories = data["u"]

    steps = size(x_trajectories[1])[1]
    horizon = 6.0
    plan_steps = 10
    state_dims = [12, 12, 12]
    ctrl_dims = [6, 6, 6]
    DT = horizon / steps

    theta_true = [5.0, 1.0, 10.0, 10.0, 0.5, 10.0, 5.0, 0.5, 8.0]
    sim_param = SimulationParams(steps=steps, horizon=horizon, plan_steps=plan_steps)
    game = define_game(state_dims, ctrl_dims, DT, zeros(length(theta_true)))

    current_time = Dates.now()
    data = Dict("sim_param" => sim_param, "state_dims" => state_dims, "ctrl_dims" => ctrl_dims, "DT" => DT)

    if plot
        fig_theta, axs_theta = subplots(3, 3, figsize=(6,4))
        fig_f, axs_f = subplots(3, 3, figsize=(6,4))
        fig_dem, ax_dem = subplots(1, 1)
        ax_dem.axis("equal")
        ax_dem.set_xlim(-15, 0)
        ax_dem.set_ylim(-2, 12)
        ax_dem.set_xlabel("X Position")
        ax_dem.set_ylabel("Y Position")
        ax_dem.set_title("Demonstration Trajectories")
    end

    x_init = [-12.3 7.2 5.20 0 -1.0 7.3 5.20 -pi -8.8 10.59 5.20 -pi/2]

    # Initialize NNs (shared cost network)
    num_agents = 3
    state_dim = sum(state_dims)
    ctrl_dim = sum(ctrl_dims)
    embedding_dim = 4
    cost_network = CostNetwork(state_dims[1], ctrl_dims[1], embedding_dim)  # Shared across agents
    policy_networks = [PolicyNetwork(state_dims[1], embedding_dim, ctrl_dims[i]) for i = 1:num_agents]
    dynamics_networks = [DynamicsNetwork(embedding_dim, ctrl_dims[i], state_dims[i]) for i = 1:num_agents]
    optimizer = torch.optim.Adam(
        vcat([Dict("params" => cost_network.parameters())],
             [Dict("params" => pn.parameters()) for pn in policy_networks],
             [Dict("params" => dn.parameters()) for dn in dynamics_networks]),
        lr=0.001
    )

    x_ref = zeros(steps * 2, state_dims)
    x_ref[1, :] = x_init
    for i = 1:steps * 2 - 1
        w, u_pred, _ = compute_costs_and_policies(game, x_ref[i, :], zeros(game.ctrl_dim), cost_network, policy_networks)
        x_ref[i + 1, :] = dynamics_nn(game, x_ref[i, :], u_pred, w, dynamics_networks)
    end
    println("==========================")

    fname_bc = string("data/", "data_bc", ".jld2")
    x_refs_bc = x_ref
    @save fname_bc x_trajectories x_refs_bc
    println(" Saved BC data")

    feature_k = length(theta_true)
    if plot
        for i = 1:length(x_trajectories)
            ax_dem.plot(x_trajectories[i][:, 1], x_trajectories[i][:, 2], alpha=0.3, color="red")
            ax_dem.plot(x_trajectories[i][:, 5], x_trajectories[i][:, 6], alpha=0.3, color="blue")
            ax_dem.plot(x_trajectories[i][:, 9], x_trajectories[i][:, 10], alpha=0.3, color="green")
        end
        ax_dem.legend(["Agent 1", "Agent 2", "Agent 3"])
        pause(0.1)
    end

    avg_dem_feature_counts = get_feature_counts_three(game, x_trajectories, u_trajectories, x_ref, feature_k)
    scale_vector = scale ? avg_dem_feature_counts ./ 100 : ones(size(avg_dem_feature_counts))
    sc_avg_dem_feature_counts = avg_dem_feature_counts ./ scale_vector
    println(" this is avg dem feature counts ", avg_dem_feature_counts)

    data["true theta"] = theta_true
    data["feature_counts_demonstration"] = avg_dem_feature_counts
    data["demonstration_xtrajectory"] = x_trajectories
    data["demonstration_utrajectory"] = u_trajectories
    data["x_reference"] = x_ref
    fname = string("data/", current_time, ".jld2")
    # @save fname data

    # IRL loop
    w_est = zeros(max_itr + 1, num_agents, embedding_dim)
    feature_counts = zeros(max_itr, feature_k)

    for itr = 1:max_itr
        println(" ------------- in iteration ", itr , " ------------- ")

        results, x_trajectories_sim, u_trajectories_sim = generate_simulations_nn(
            sim_param, game, x_init, x_ref, sample_size, cost_network, policy_networks, dynamics_networks
        )

        dynamics_loss = 0.0
        equilibrium_loss = 0.0
        w_curr = zeros(num_agents, embedding_dim)
        for i = 1:sample_size
            x_traj = x_trajectories_sim[i]
            u_traj = u_trajectories_sim[i]
            for t = 1:steps
                w, u_pred, costs = compute_costs_and_policies(game, x_traj[t, :], u_traj[t, :], cost_network, policy_networks)
                s_next_pred = dynamics_nn(game, x_traj[t, :], u_traj[t, :], w, dynamics_networks)
                dynamics_loss += sum((s_next_pred - x_traj[t + 1, :]) .^ 2)
                dem_u = i <= length(u_trajectories) ? u_trajectories[i][t, :] : u_traj[t, :]
                equilibrium_loss += sum(costs) + sum((u_pred - dem_u) .^ 2)
                if i == 1 && t == 1
                    w_curr = torch.stack(w).detach().numpy()
                end
            end
        end

        avg_pro_feature_counts = get_feature_counts_three(game, x_trajectories_sim, u_trajectories_sim, x_ref, feature_k)
        feature_counts[itr, :] = avg_pro_feature_counts

        loss = torch.tensor(dynamics_loss / sample_size + equilibrium_loss / sample_size, requires_grad=true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        w_est[itr + 1, :, :] = w_curr

        println("            this is avg dem feature counts ", avg_dem_feature_counts)
        println("            this is avg proposed feature counts ", avg_pro_feature_counts)
        println("            this is our current w estimation ", w_curr)

        data["w_est"] = w_est[1:itr + 1, :, :]
        data["feature_counts_proposed"] = feature_counts[1:itr, :]
        fname = string("data/", current_time, ".jld2")
        # @save fname data

        datapath = expanduser("~/Research/bluecity_example/data/")
        gname = string(datapath, "learned_x_trajectories_rld_simple", ".jld2")
        @save gname x_trajectories_sim
    end

    if plot
        for i = 1:3
            for j = 1:min(embedding_dim, 3)
                idx = (i - 1) * 3 + j
                axs_theta[i, j].plot(w_est[:, i, j], label="Estimated w[$i,$j]")
                axs_theta[i, j].set_title("w[$i,$j]")
                axs_theta[i, j].set_xlabel("Iteration")
                axs_theta[i, j].set_ylabel("Value")
                axs_theta[i, j].legend()

                axs_f[i, j].plot(feature_counts[:, idx], label="Proposed")
                axs_f[i, j].plot(avg_dem_feature_counts[idx] * ones(max_itr), "--", label="Demonstration")
                axs_f[i, j].set_title("Feature $idx")
                axs_f[i, j].set_xlabel("Iteration")
                axs_f[i, j].set_ylabel("Count")
                axs_f[i, j].legend()
            end
        end
        fig_theta.tight_layout()
        fig_f.tight_layout()
        pause(0.1)
    end
end

#===========================#
# Main execution
#============================#
ma_irl(sync_update=false,
       single_update=1,
       scale=true,
       eta=0.01,
       max_itr=10,
       sample_size=10,
       dem_num=10,
       plot=true)