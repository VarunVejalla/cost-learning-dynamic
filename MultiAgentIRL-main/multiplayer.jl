# file:     multiplayer.jl
# author:   mingyuw@stanford.edu

# nonlinear game with 3 players


using PyPlot
using ForwardDiff, LinearAlgebra
using Distributions, StatsBase
using Dates
using JLD2, FileIO


include("lqgame.jl")
include("utils.jl")



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

    steps = 60
    horizon = 6.0
    plan_steps = 10


    state_dims = [4, 4, 4]
    ctrl_dims = [2, 2, 2]
    DT = horizon/steps

    theta_true = [1.0, 1.0, 8.0,      0.5, 0.5, 10.0,    0.5, 0.5, 8.0]

    sim_param = SimulationParams(steps=steps, horizon=horizon, plan_steps=plan_steps)
    game = define_game(state_dims, ctrl_dims, DT, theta_true)
    # save file and figures
    current_time = Dates.now()

    if plot
        fig_theta, axs_theta = subplots(3, 3, figsize=(6,4))
        fig_f, axs_f = subplots(3, 3, figsize=(6,4))
        fig_dem, ax_dem = subplots(1, 1)
        ax_dem.axis("equal")
        ax_dem.set_xlim(-5, 5)
        ax_dem.set_ylim(-5, 5)
    end

    x_init = [-4.0 0.0 1.33 0.0     2.0 3.46 1.33 -2.0/3*pi   2.0 -3.46 1.33 2.0/3*pi]
    x_ref = zeros(steps*2, game.state_dim)
    x_ref[1,:] = x_init
    for i=1:steps*2 - 1
        x_ref[i+1,:] = game.dynamics_func([x_ref[i,:]; zeros(game.ctrl_dim)])
    end

    #=============
    generate demonstrations
    ==============#
    demo_data = zeros(dem_num, steps+1, game.state_dim * 2 + game.ctrl_dim)
    feature_k = length(theta_true)

    dem_results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
                                       nl_game=game,
                                       x_init=x_init,
                                       traj_ref=x_ref,
                                       num_sim=dem_num)

    x_trajectories = dem_results.state_trajectories
    u_trajectories = dem_results.ctrl_trajectories
    demo_data[:,:,1:game.state_dim] = x_trajectories_data
    demo_data[:,:,game.state_dim*2+1:end] = u_trajectories_data
    for di=1:dem_num
        demo_data[di,:, 1+game.state_dim:game.state_dim*2] = x_ref[1:steps+1,:]
    end
    if plot
        for i = 1:length(x_trajectories)
            ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="blue")
            ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="blue")
            ax_dem.plot(x_trajectories[i][:,9], x_trajectories[i][:,10], alpha=0.3, color="blue")
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

    fname = string("cioc_data/threeplayer.h5")
    @save fname demo_data

end



#===========================
main
============================#

ma_irl(sync_update=false,     # if true, update param for both players at the same time
       single_update=5,       # if sync_update = false, then update each player single_update times before updating the other player
       # single_update=1,
       scale=true,            # normalize features
       eta=0.01,              # parameter for learning rate, starting value
       max_itr=5000,          # maximum iteration
       sample_size = 50,      # number of samples in each update to approximate feature counts
       # sample_size=2,
       dem_num=500,
       plot=true)
