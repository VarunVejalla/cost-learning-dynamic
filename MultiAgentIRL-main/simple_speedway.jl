# file:     multiplayer.jl
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
include("lqgame.jl")  # Assumed to contain game-solving functions (e.g., lqgame_QRE)
include("utils.jl")   # Assumed to contain helper functions (e.g., generate_simulations)

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
function ma_irl(;sync_update=true,       # If true, update all players' parameters simultaneously
                single_update=10,        # Number of updates per player when sync_update=false
                scale=true,              # Whether to scale feature counts
                eta=0.0001,              # Initial learning rate
                plot=true,               # Whether to generate plots
                max_itr=5000,            # Maximum number of IRL iterations
                sample_size=100,         # Number of simulated trajectories per iteration
                dem_num=200)             # Number of demonstration trajectories

    println("==========================")

    # Load demonstration trajectories from pickle file
    data = load_pickle("data/demo_trajectories_rld.pickle")
    x_trajectories = data["x"]  # State trajectories
    u_trajectories = data["u"]  # Control trajectories

    # Simulation parameters
    steps = size(x_trajectories[1])[1]  # Number of time steps (e.g., 64)
    horizon = 6.0                       # Total time horizon in seconds
    plan_steps = 10                     # Number of planning steps

    state_dims = [4, 4, 4]              # State dimensions for 3 agents (e.g., [x, y, vx, vy])
    ctrl_dims = [2, 2, 2]               # Control dimensions for 3 agents (e.g., [ax, ay])
    DT = horizon / steps                # Time step size

    # True cost parameters for demonstration generation
    theta_true = [5.0, 1.0, 10.0,      # Agent 1: state cost, ctrl cost 1, ctrl cost 2
                  10.0, 0.5, 10.0,     # Agent 2
                  5.0, 0.5, 8.0]       # Agent 3

    # Initialize simulation and game objects
    sim_param = SimulationParams(steps=steps, horizon=horizon, plan_steps=plan_steps)
    game = define_game(state_dims, ctrl_dims, DT, theta_true)  # Define nonlinear game dynamics and costs

    # Save metadata
    current_time = Dates.now()
    data = Dict()
    data["sim_param"] = sim_param
    data["state_dims"] = state_dims
    data["ctrl_dims"] = ctrl_dims
    data["DT"] = DT

    # Set up plotting if enabled
    if plot
        fig_theta, axs_theta = subplots(3, 3, figsize=(6,4))  # For theta evolution
        fig_f, axs_f = subplots(3, 3, figsize=(6,4))          # For feature counts
        fig_dem, ax_dem = subplots(1, 1)                      # For trajectory visualization
        ax_dem.axis("equal")
        ax_dem.set_xlim(-15, 0)
        ax_dem.set_ylim(-2, 12)
        ax_dem.set_xlabel("X Position")                       # Add X-axis label
        ax_dem.set_ylabel("Y Position")                       # Add Y-axis label
        ax_dem.set_title("Demonstration Trajectories")        # Add title
    end

    # Initial state for 3 agents: [x, y, vx, theta] per agent
    x_init = [-12.3 7.2 5.20 0      # Agent 1: right
              -1.0 7.3 5.20 -pi     # Agent 2: left
              -8.8 10.59 5.20 -pi/2] # Agent 3: down

    # Generate reference trajectory (no control applied)
    x_ref = zeros(steps * 2, game.state_dim)
    x_ref[1, :] = x_init
    for i = 1:steps * 2 - 1
        x_ref[i + 1, :] = game.dynamics_func([x_ref[i, :]; zeros(game.ctrl_dim)])
    end
    println("==========================")

    # Save boundary condition data
    fname_bc = string("data/", "data_bc", ".jld2")
    x_refs_bc = x_ref
    @save fname_bc x_trajectories x_refs_bc
    println(" Saved BC data")

    #=============#
    # Generate demonstrations (loaded from pickle)
    #=============#
    feature_k = length(theta_true)  # Number of features (9 for 3 agents)

    # dem_results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
    #                                    nl_game=game,
    #                                    x_init=x_init,
    #                                    traj_ref=x_ref,
    #                                    num_sim=dem_num)

    # x_trajectories = dem_results.state_trajectories
    # u_trajectories = dem_results.ctrl_trajectories
    # u_trajectories = [zeros(Float64, size(x_trajectories_data[1])[1]-1, 6) for _ in 1:length(x_trajectories)]

    # Plot demonstration trajectories
    if plot
        for i = 1:length(x_trajectories)
            ax_dem.plot(x_trajectories[i][:, 1], x_trajectories[i][:, 2], alpha=0.3, color="red")
            ax_dem.plot(x_trajectories[i][:, 5], x_trajectories[i][:, 6], alpha=0.3, color="blue")
            ax_dem.plot(x_trajectories[i][:, 9], x_trajectories[i][:, 10], alpha=0.3, color="green")
        end
        ax_dem.legend()  # Add legend to distinguish agents
        pause(0.1)
    end

    # Compute demonstration feature counts
    avg_dem_feature_counts = get_feature_counts_three(game, x_trajectories, u_trajectories, x_ref, feature_k)
    scale_vector = scale ? avg_dem_feature_counts ./ 100 : ones(size(avg_dem_feature_counts))  # Scaling factor
    sc_avg_dem_feature_counts = avg_dem_feature_counts ./ scale_vector
    println(" this is avg dem feature counts ", avg_dem_feature_counts)

    # Save demonstration data
    data["true theta"] = theta_true
    data["feature_counts_demonstration"] = avg_dem_feature_counts
    data["demonstration_xtrajectory"] = x_trajectories
    data["demonstration_utrajectory"] = u_trajectories
    data["x_reference"] = x_ref

    fname = string("data/", current_time, ".jld2")
    # @save fname data  # Uncomment to save

    #=============# 
    # IRL iteration loop
    #=============#
    rand_init = Product(Uniform.([0.2, 0.2, 2.0, 0.2, 0.2, 2.0, 0.2, 0.2, 2.0],  # Lower bounds
                                 [0.5, 0.5, 5.0, 0.5, 0.5, 5.0, 0.5, 0.5, 5.0]))  # Upper bounds
    theta_curr = rand(rand_init) .* [sample([-1, 1]) for i = 1:feature_k] + [0, 0, 0, 5.0, 1.0, 10.0, 0, 0, 0]  # Random initialization with offset

    theta_avg = theta_curr  # Running average of theta
    theta_est = zeros(max_itr + 1, feature_k)  # Store theta estimates over iterations
    theta_est[1, :] = theta_curr
    theta_smooth = zeros(max_itr + 1, feature_k)  # Smoothed theta estimates
    theta_smooth[1, :] = theta_avg
    feature_counts = zeros(max_itr, feature_k)  # Store feature counts

    for itr = 1:max_itr
        println(" ------------- in iteration ", itr , " ------------- ")
        avg_pro_feature_counts = zeros(feature_k)
        lr = rate(eta, itr)  # Update learning rate

        # Update parameters for Agent 2 (example; others are commented out)
        for player2_itr = 1:single_update
            curr_game = define_game(state_dims, ctrl_dims, DT, theta_curr)  # Redefine game with current theta
            results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
                                                                                    nl_game=curr_game,
                                                                                    x_init=x_init,
                                                                                    traj_ref=x_ref,
                                                                                    num_sim=sample_size)

            x_trajectories = results.state_trajectories
            u_trajectories = results.ctrl_trajectories
            avg_pro_feature_counts = get_feature_counts_three(curr_game, x_trajectories, u_trajectories, x_ref, feature_k)
            sc_avg_pro_feature_counts = avg_pro_feature_counts ./ scale_vector

            # Gradient descent update for Agent 2's parameters (theta[4:6])
            theta_curr[4:6] = theta_curr[4:6] - lr * (sc_avg_dem_feature_counts[4:6] - sc_avg_pro_feature_counts[4:6])
            theta_curr = max.(0.0, theta_curr)  # Ensure non-negative theta
        end

        # Store results
        feature_counts[itr, :] = avg_pro_feature_counts
        theta_est[itr + 1, :] = theta_curr
        theta_avg = itr >= 10 ? mean(theta_est[itr - 9:itr + 1, :], dims=1) : mean(theta_est[1:itr + 1, :], dims=1)
        theta_smooth[itr + 1, :] = theta_avg

        # Print progress
        println("            this is avg dem feature counts ", avg_dem_feature_counts)
        println("            this is avg proposed feature counts ", avg_pro_feature_counts)
        println("            this is our current theta estimation ", theta_curr, " and averaged over time ", theta_avg)

        # Save iteration data
        data["theta_est"] = theta_est[1:itr + 1, :]
        data["feature_counts_proposed"] = feature_counts[1:itr, :]
        fname = string("data/", current_time, ".jld2")
        # @save fname data  # Uncomment to save

        datapath = expanduser("~/Research/bluecity_example/data/")
        gname = string(datapath, "learned_x_trajectories_rld_simple", ".jld2")
        @save gname x_trajectories  # Save learned trajectories
    end

    # Plot theta and feature counts (labels added)
    if plot
        for i = 1:3
            for j = 1:3
                idx = (i - 1) * 3 + j
                axs_theta[i, j].plot(theta_est[:, idx], label="Estimated")
                axs_theta[i, j].plot(theta_true[idx] * ones(max_itr + 1), "--", label="True")
                axs_theta[i, j].set_title("Theta $idx")
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
ma_irl(sync_update=false,     # Update players sequentially
       single_update=1,       # Number of updates per player
       scale=true,            # Normalize features
       eta=0.01,              # Starting learning rate
       max_itr=10,            # Max iterations
       sample_size=10,        # Samples per iteration
       dem_num=10,            # Number of demonstrations
       plot=true)             # Enable plotting