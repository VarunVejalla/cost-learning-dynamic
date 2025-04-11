# file:     multiplayer.jl
# author:   mingyuw@stanford.edu

# nonlinear game with 3 players

ENV["MPLBACKEND"]="tkagg"
# pygui(true)
using PyPlot
using Flux
using PyCall
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

    println("==========================")
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
        ax_dem.set_xlim(-5, 5)
        ax_dem.set_ylim(-5, 5)
    end

    # x_init = [-4.0 0.0 1.33 0.0     2.0 3.46 1.33 -2.0/3*pi   2.0 -3.46 1.33 2.0/3*pi]
    x_init = [-4.0 0.0 1.33 0.0     0.0 4.0 1.33 -pi/2   0.0 -4.0 1.33 pi/2]
    # x_init = [-4.0 -4.0 1.33 pi/4     4.0 4.0 1.33 -3*pi/4   0.0 -4.0 1.33 pi/2]

    x_ref = zeros(steps*2, game.state_dim)
    x_ref[1,:] = x_init
    for i=1:steps*2 - 1
        x_ref[i+1,:] = game.dynamics_func([x_ref[i,:]; zeros(game.ctrl_dim)])
    end
    println("==========================")

    #=============
    generate demonstrations
    ==============#
    feature_k = length(theta_true)

    dem_results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
                                       nl_game=game,
                                       x_init=x_init,
                                       traj_ref=x_ref,
                                       num_sim=dem_num)

    x_trajectories = dem_results.state_trajectories
    u_trajectories = dem_results.ctrl_trajectories
    
    fname_bc = string("data/", "data_bc", ".jld2")
    x_refs_bc = x_ref
    @save fname_bc x_trajectories x_refs_bc
    println(" Saved BC data")
    datapath = expanduser("~/Research/bluecity_trajectories/data/")
    fname = string(datapath, "GT", ".jld2")
    @save fname x_trajectories
    println(" Saved GT data")
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

    #=============
    iteration on likelihood maximization
    ==============#
    rand_init = Product(Uniform.([0.2, 0.2, 2.0, 0.2, 0.2, 2.0, 0.2, 0.2, 2.0],
                [0.5, 0.5, 5.0, 0.5, 0.5, 5.0, 0.5, 0.5, 5.0]))
    theta_curr = rand(rand_init) .* [sample([-1,1]) for i = 1:feature_k] + theta_true
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

        if sync_update
            curr_game = define_game(state_dims, ctrl_dims, DT, theta_curr)
            results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
                                              nl_game=curr_game,
                                              x_init=x_init,
                                              traj_ref=x_ref,
                                              num_sim=sample_size)
            x_trajectories = results.state_trajectories
            u_trajectories = results.ctrl_trajectories
            avg_pro_feature_counts = get_feature_counts_three(curr_game, x_trajectories, u_trajectories, x_ref, eature_k)
            sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

            theta_curr = theta_curr - lr * (sc_avg_dem_feature_counts - sc_avg_pro_feature_counts)
            theta_curr = max.(0.0, theta_curr)
        else
            # asynchronous update
            for player1_itr = 1:single_update
                # update weights on player 1
                curr_game = define_game(state_dims, ctrl_dims, DT, theta_curr)
                results,  x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
                                                  nl_game=curr_game,
                                                  x_init=x_init,
                                                  traj_ref=x_ref,
                                                  num_sim=sample_size)

                x_trajectories = results.state_trajectories
                u_trajectories = results.ctrl_trajectories
                if plot
                    for i = 1:length(x_trajectories)
                        ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="red")
                        ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="blue")
                        ax_dem.plot(x_trajectories[i][:,9], x_trajectories[i][:,10], alpha=0.3, color="green")
                    end
                    pause(0.1)
                end
                avg_pro_feature_counts = get_feature_counts_three(curr_game, x_trajectories, u_trajectories, x_ref, feature_k)
                sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

                theta_curr[1:3] = theta_curr[1:3] - lr * (sc_avg_dem_feature_counts[1:3] - sc_avg_pro_feature_counts[1:3])
                theta_curr = max.(0.0, theta_curr)
            end

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
            for player3_itr = 1:single_update
                # update weights on player 1
                curr_game = define_game(state_dims, ctrl_dims, DT, theta_curr)
                results, x_trajectories_data, u_trajectories_data = generate_simulations(sim_param=sim_param,
                                                  nl_game=curr_game,
                                                  x_init=x_init,
                                                  traj_ref=x_ref,
                                                  num_sim=sample_size)

                x_trajectories = results.state_trajectories
                u_trajectories = results.ctrl_trajectories
                avg_pro_feature_counts = get_feature_counts_three(curr_game, x_trajectories, u_trajectories, x_ref, feature_k)
                sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

                theta_curr[7:end] = theta_curr[7:end] - lr * (sc_avg_dem_feature_counts[7:end] - sc_avg_pro_feature_counts[7:end])
                theta_curr = max.(0.0, theta_curr)
            end
        end
        datapath = expanduser("~/Research/bluecity_trajectories/data/")
        hname = string(datapath, "Learned", ".jld2")
        @save hname x_trajectories
        println(" Saved Learned data")
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
        if plot
            for i = 1:feature_k
                axs_theta[i].clear()
                axs_theta[i].plot(theta_est[1:itr+1, i])
                axs_theta[i].plot(theta_smooth[1:itr+1, i])
                axs_theta[i].plot(theta_true[i] * ones(itr+1))

                axs_f[i].clear()
                axs_f[i].plot(feature_counts[1:itr, i])
                axs_f[i].plot(avg_dem_feature_counts[i] * ones(itr))
            end
        end

        # save data along the way
        data["theta_est"] = theta_est[1:itr+1, :]
        data["feature_counts_proposed"] = feature_counts[1:itr, :]

        fname = string("data/", current_time, ".jld2")
        @save fname data
    end

end



#===========================
main
============================#

ma_irl(sync_update=false,     # if true, update param for both players at the same time
       single_update=5,       # if sync_update = false, then update each player single_update times before updating the other player
       # single_update=1,
       scale=true,            # normalize features
       eta=0.01,              # parameter for learning rate, starting value
       max_itr=1,          # maximum iteration
       sample_size = 10,      # number of samples in each update to approximate feature counts
       # sample_size=2,
       dem_num=10,
       plot=false)
