# file:     multiplayer.jl
# author:   mingyuw@stanford.edu

# nonlinear game with 3 players

ENV["MPLBACKEND"]="tkagg"
# pygui(true)
using PyPlot
using ForwardDiff, LinearAlgebra
using Distributions, StatsBase
using Dates
using JLD2, FileIO
using Pkg
using PyCall
using Pickle

# Pkg.add("PyCall")

include("lqgame.jl")
include("utils.jl")

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

load_pickle = py"load_pickle"

function rate(lr0, itr)
    return lr0/(1.0 + 0.1 * itr)
end


function ma_irl(;sync_update=true,
                single_update=10,
                scale=true,
                eta=0.0001,
                plot=true,
                max_itr=5000,
                sample_size=100,
                dem_num=200)

    println("==========================")

    # Load the pickle module from Python
    # pickle = pyimport("pickle")
    
    # Specify the path to the pickle file
    data = load_pickle("data/demo_trajectories_rld.pickle")
    x_trajectories = data["x"]
    u_trajectories = data["u"]

    steps = size(x_trajectories[1])[1] #64
    horizon = 6.0
    plan_steps = 10


    state_dims = [4, 4, 4]
    ctrl_dims = [2, 2, 2]
    DT = horizon/steps

    # theta_true = [5.0, 1.0, 10.0,      5.0, 0.5, 10.0,    1.0, 0.5, 8.0] # red and green ok, blue (agent 2, left) spread
    # theta_true = [5.0, 1.0, 10.0,      5.0, 0.5, 10.0,    5.0, 0.5, 8.0] #  green ok, blue/red (agent 1,2) spread
    theta_true = [5.0, 1.0, 10.0,      10.0, 0.5, 10.0,    5.0, 0.5, 8.0] # 
    # theta_true = [1.0, 1.0, 10.0,      1.0, 0.5, 10.0,    1.0, 1.0, 8.0] # 

    sim_param = SimulationParams(steps=steps, horizon=horizon, plan_steps=plan_steps)
    game = define_game(state_dims, ctrl_dims, DT, theta_true)
    # save file and figures
    current_time = Dates.now()
    data = Dict()
    data["sim_param"] = sim_param
    data["state_dims"] = state_dims
    data["ctrl_dims"] = ctrl_dims
    data["DT"] = DT

    if plot
        fig_theta, axs_theta = subplots(3, 3, figsize=(6,4))
        fig_f, axs_f = subplots(3, 3, figsize=(6,4))
        fig_dem, ax_dem = subplots(1, 1)
        ax_dem.axis("equal")
        ax_dem.set_xlim(-15, 0)
        ax_dem.set_ylim(-2, 12)
    end

    # x_init = [-4.0 0.0 1.33 0.0     2.0 3.46 1.33 -2.0/3*pi   2.0 -3.46 1.33 2.0/3*pi]
    # x_init = [-4.0 0.0 1.33 0.0     0.0 4.0 1.33 -pi/2   0.0 -4.0 1.33 pi/2]
    x_init = [-12.3 7.2 5.20 0     -1.0 7.3 5.20 -pi   -8.8 10.59 5.20 -pi/2] # right, left, down
    # x_init = [-8.0 -2.0 5.03 pi/2     -9.0 10.5 5.03 -pi/2   -1.0 7.3 5.20 -pi] # up, down, left
    # x_init = [-8.0 -2.0 5.03 pi/2     -9.0 10.5 5.03 -pi/2   -12.3 7.2 5.03 0] # up, down, right
    # x_init = [-12.3 7.2 5.20 0     -1.0 7.3 5.20 -pi   -8.0 -2.0 5.20 pi/2]  # right, left, up

    x_ref = zeros(steps*2, game.state_dim)
    x_ref[1,:] = x_init
    for i=1:steps*2 - 1
        x_ref[i+1,:] = game.dynamics_func([x_ref[i,:]; zeros(game.ctrl_dim)])
    end
    println("==========================")
    fname_bc = string("data/", "data_bc", ".jld2")
    x_refs_bc = x_ref
    @save fname_bc x_trajectories x_refs_bc
    println(" Saved BC data")

    #=============
    generate demonstrations
    ==============#
    feature_k = length(theta_true)

    # dem_results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
    #                                    nl_game=game,
    #                                    x_init=x_init,
    #                                    traj_ref=x_ref,
    #                                    num_sim=dem_num)

    # x_trajectories = dem_results.state_trajectories
    # u_trajectories = dem_results.ctrl_trajectories
    # u_trajectories = [zeros(Float64, size(demo_trajectories[1])[1]-1, 6) for _ in 1:length(x_trajectories)]
    

    
    if plot
        for i = 1:length(x_trajectories)
            ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="red")
            ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="blue")
            ax_dem.plot(x_trajectories[i][:,9], x_trajectories[i][:,10], alpha=0.3, color="green")
        end
        pause(0.1)
    end
    # @info game
    avg_dem_feature_counts = get_feature_counts_three(game, x_trajectories, u_trajectories, x_ref, feature_k)
    if scale
        scale_vector = avg_dem_feature_counts./100
    else
        scale_vector = ones(size(avg_dem_feature_counts))    # do not scale
    end


    sc_avg_dem_feature_counts = avg_dem_feature_counts./scale_vector
    println(" this is avg dem feature counts ", avg_dem_feature_counts)

    data["true theta"] = theta_true
    data["feature_counts_demonstration"] = avg_dem_feature_counts
    data["demonstration_xtrajectory"] = x_trajectories
    data["demonstration_utrajectory"] = u_trajectories
    data["x_reference"] = x_ref

    fname = string("data/", current_time, ".jld2")
    # @save fname data

    #=============
    iteration on likelihood maximization
    ==============#
    rand_init = Product(Uniform.([0.2, 0.2, 2.0, 0.2, 0.2, 2.0, 0.2, 0.2, 2.0],
                [0.5, 0.5, 5.0, 0.5, 0.5, 5.0, 0.5, 0.5, 5.0]))
    # theta_curr = rand(rand_init) .* [sample([-1,1]) for i = 1:feature_k] + theta_true
    theta_curr = rand(rand_init) .* [sample([-1,1]) for i = 1:feature_k] + [0, 0, 0, 5.0, 1.0, 10.0,   0, 0, 0] # 

    theta_avg = theta_curr

    theta_est = zeros(max_itr+1, feature_k)
    theta_est[1,:] = theta_curr
    theta_smooth = zeros(max_itr+1, feature_k)
    theta_smooth[1,:] = theta_avg
    feature_counts = zeros(max_itr, feature_k)


    for itr = 1:max_itr
        println(" ------------- in iteration ", itr , " ------------- ")
        avg_pro_feature_counts = zeros(feature_k)
        lr = rate(eta, itr)

        # for player1_itr = 1:single_update
        #     # update weights on player 1
        #     curr_game = define_game(state_dims, ctrl_dims, DT, theta_curr)
        #     results,  x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
        #                                         nl_game=curr_game,
        #                                         x_init=x_init,
        #                                         traj_ref=x_ref,
        #                                         num_sim=sample_size)

        #     x_trajectories = results.state_trajectories
        #     u_trajectories = results.ctrl_trajectories
        #     if plot
        #         for i = 1:length(x_trajectories)
        #             ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="blue")
        #             ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="blue")
        #             ax_dem.plot(x_trajectories[i][:,9], x_trajectories[i][:,10], alpha=0.3, color="blue")
        #         end
        #         pause(0.1)
        #     end
        #     avg_pro_feature_counts = get_feature_counts_three(curr_game, x_trajectories, u_trajectories, x_ref, feature_k)
        #     sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

        #     theta_curr[1:3] = theta_curr[1:3] - lr * (sc_avg_dem_feature_counts[1:3] - sc_avg_pro_feature_counts[1:3])
        #     theta_curr = max.(0.0, theta_curr)
        # end

        for player2_itr = 1:single_update
            # update weights on player 1
            curr_game = define_game(state_dims, ctrl_dims, DT, theta_curr)
            results,  x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
                                                nl_game=curr_game,
                                                x_init=x_init,
                                                traj_ref=x_ref,
                                                num_sim=sample_size)

            x_trajectories = results.state_trajectories
            u_trajectories = results.ctrl_trajectories
            avg_pro_feature_counts = get_feature_counts_three(curr_game, x_trajectories, u_trajectories, x_ref, feature_k)
            sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

            theta_curr[4:6] = theta_curr[4:6] - lr * (sc_avg_dem_feature_counts[4:6] - sc_avg_pro_feature_counts[4:6])
            theta_curr = max.(0.0, theta_curr)
        end
        # for player3_itr = 1:single_update
        #     # update weights on player 1
        #     curr_game = define_game(state_dims, ctrl_dims, DT, theta_curr)
        #     results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
        #                                         nl_game=curr_game,
        #                                         x_init=x_init,
        #                                         traj_ref=x_ref,
        #                                         num_sim=sample_size)

        #     x_trajectories = results.state_trajectories
        #     u_trajectories = results.ctrl_trajectories
        #     avg_pro_feature_counts = get_feature_counts_three(curr_game, x_trajectories, u_trajectories, x_ref, feature_k)
        #     sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

        #     theta_curr[7:end] = theta_curr[7:end] - lr * (sc_avg_dem_feature_counts[7:end] - sc_avg_pro_feature_counts[7:end])
        #     theta_curr = max.(0.0, theta_curr)
        # end

        feature_counts[itr,:] = avg_pro_feature_counts
        theta_est[itr+1,:] = theta_curr

        if itr >= 10
            theta_avg = mean(theta_est[itr-9:itr+1,:], dims=1)
        else
            theta_avg = mean(theta_est[1:itr+1,:], dims=1)
        end
        theta_smooth[itr+1,:] = theta_avg

        println("            this is avg dem feature counts ", avg_dem_feature_counts)
        println("            this is avg proposed feature counts ", avg_pro_feature_counts)
        println("            this is our current theta estimation ", theta_curr, " and averaged over time ", theta_avg)

        # save data along the way
        data["theta_est"] = theta_est[1:itr+1, :]
        data["feature_counts_proposed"] = feature_counts[1:itr, :]

        fname = string("data/", current_time, ".jld2")
        # @save fname data
        datapath = expanduser("~/Research/bluecity_example/data/")
        gname = string(datapath, "learned_x_trajectories_rld_simple", ".jld2")
        @save gname x_trajectories
    end
end



#===========================
main
============================#

ma_irl(sync_update=false,     # if true, update param for both players at the same time
       single_update=1,       # if sync_update = false, then update each player single_update times before updating the other player
       # single_update=1,
       scale=true,            # normalize features
       eta=0.01,              # parameter for learning rate, starting value
       max_itr=10,          # maximum iteration
       sample_size = 10,      # number of samples in each update to approximate feature counts
       # sample_size=2,
       dem_num=10,
       # plot=false)
       plot=true)