# coupled.jl
# mingyuw@stanford.edu

# try to couple the problem and eliminates the degenerate case


using PyPlot
using LinearAlgebra, Distributions, StatsBase
using Dates
using JLD2, FileIO

include("lqgame.jl")

state_dim_1 = 4
ctrl_dim_1 = 2
state_dim_2 = 4
ctrl_dim_2 = 2
state_dim = state_dim_1 + state_dim_2
ctrl_dim = ctrl_dim_1 + ctrl_dim_2


plan_steps = 20
horizon = 2.0   # [s]
DT = horizon/plan_steps


#=
coefficients for dynamics
=#
A = zeros(state_dim, state_dim)
A[1:state_dim_1, 1:state_dim_1] = [1 0 DT 0; 0 1 0 DT; 0 0 1 0; 0 0 0 1]
A[state_dim_1+1:end, state_dim_1+1:end] = [1 0 DT 0; 0 1 0 DT; 0 0 1 0; 0 0 0 1]
B1 = zeros(state_dim, ctrl_dim_1)
B1[1:state_dim_1,:] = [0 0; 0 0; DT 0; 0 DT]
B2 = zeros(state_dim, ctrl_dim_2)
B2[state_dim_1+1:end,:] = [0 0; 0 0; DT 0; 0 DT]


function dynamics_forward(s)
    state = s[1:state_dim]
    ctrl1 = s[state_dim+1:state_dim+ctrl_dim_1]
    ctrl2 = s[state_dim+ctrl_dim_1+1:end]

    return A * state + B1 * ctrl1 + B2 * ctrl2
end



function set_up_system(theta)
    w_state1 = zeros(state_dim, state_dim)
    w_state1[1:state_dim_1, 1:state_dim_1] = (theta[1] + theta[2]) * Matrix{Float64}(I, state_dim_1, state_dim_1)
    w_state1[1+state_dim_1:end, 1:state_dim_1] = theta[2] * Matrix{Float64}(I, state_dim_2, state_dim_1)
    w_state1[1:state_dim_1, 1+state_dim_1:end] = theta[2] * Matrix{Float64}(I, state_dim_1, state_dim_2)
    w_state1[1+state_dim_1:end, 1+state_dim_1:end] = (theta[1] + theta[2]) * Matrix{Float64}(I, state_dim_2, state_dim_2)
    w_ctrl11 = theta[3] * Matrix{Float64}(I, ctrl_dim_1, ctrl_dim_1)
    w_ctrl12 = theta[4] * Matrix{Float64}(I, ctrl_dim_2, ctrl_dim_2)

    w_state2 = zeros(state_dim, state_dim)
    w_state2[1:state_dim_1, 1:state_dim_1] = (theta[5] + theta[6]) * Matrix{Float64}(I, state_dim_1, state_dim_1)
    w_state2[1+state_dim_1:end, 1:state_dim_1] = theta[6] * Matrix{Float64}(I, state_dim_2, state_dim_1)
    w_state2[1:state_dim_1, 1+state_dim_1:end] = theta[6] * Matrix{Float64}(I, state_dim_1, state_dim_2)
    w_state2[1+state_dim_1:end, 1+state_dim_1:end] = (theta[5] + theta[6]) * Matrix{Float64}(I, state_dim_2, state_dim_2)
    w_ctrl21 = theta[7] * Matrix{Float64}(I, ctrl_dim_1, ctrl_dim_1)
    w_ctrl22 = theta[8] * Matrix{Float64}(I, ctrl_dim_2, ctrl_dim_2)

    Dynamics = Dict{String, Array{Array{Float64}}}()
    Dynamics["A"] = [A for i = 1:plan_steps]
    Dynamics["B1"] = [B1 for i = 1:plan_steps]
    Dynamics["B2"] = [B2 for i = 1:plan_steps]
    Costs = Dict{String, Array{Array{Float64}}}()
    Costs["Q1"] = [w_state1 for i = 1:plan_steps+1]
    Costs["l1"] = [zeros(state_dim) for i = 1:plan_steps+1]
    Costs["Q2"] = [w_state2 for i = 1:plan_steps+1]
    Costs["l2"] = [zeros(state_dim) for i = 1:plan_steps+1]
    Costs["R11"] = [w_ctrl11 for i = 1:plan_steps]
    Costs["R12"] = [w_ctrl12 for i = 1:plan_steps]
    Costs["R21"] = [w_ctrl21 for i = 1:plan_steps]
    Costs["R22"] = [w_ctrl22 for i = 1:plan_steps]
    return Dynamics, Costs
end



function generate_sim(x_init, theta, num = 200)

    x_trajectories, u_trajectories = [], []

    # compute the Quantal response equilibrium
    Dynamics, Costs = set_up_system(theta)
    N1, N2, alpha1, alpha2, cov1, cov2 = lqgame_QRE(Dynamics, Costs)

    # run simulations to get optimal equilibrium trajectories
    for i = 1:num
        x_history = zeros(plan_steps+1, state_dim)
        x_history[1,:] = x_init
        u_history = zeros(plan_steps, ctrl_dim)
        for t = 1:plan_steps
            u_mean1 = -N1[end-t+1] * x_history[t, :] - alpha1[end-t+1]
            u_dist1 = MvNormal(u_mean1, cov1[end-t+1])
            u_mean2 = -N2[end-t+1] * x_history[t, :] - alpha2[end-t+1]
            u_dist2 = MvNormal(u_mean2, cov2[end-t+1])

            u_history[t,:] = [rand(u_dist1)..., rand(u_dist2)...]
            x_history[t+1, :] = dynamics_forward([x_history[t, :]; u_history[t, :]])
        end
        push!(x_trajectories, x_history)
        push!(u_trajectories, u_history)
    end
    return x_trajectories, u_trajectories
end



function get_feature_counts(x_trajectories, u_trajectories, feature_k)
    feature_counts = zeros(feature_k)
    num = length(x_trajectories)

    coeff = zeros(state_dim, state_dim)
    coeff[1:state_dim_1, 1:state_dim_1] = Matrix{Float64}(I, state_dim_1, state_dim_1)
    coeff[1+state_dim_1:end, 1:state_dim_1] = Matrix{Float64}(I, state_dim_2, state_dim_1)
    coeff[1:state_dim_1, 1+state_dim_1:end] = Matrix{Float64}(I, state_dim_1, state_dim_2)
    coeff[1+state_dim_1:end, 1+state_dim_1:end] = Matrix{Float64}(I, state_dim_2, state_dim_2)
    for i = 1:num
        xtraj = x_trajectories[i]
        utraj = u_trajectories[i]
        for t = 1:plan_steps
            feature_counts[1] += xtraj[t,:]' * xtraj[t,:]
            feature_counts[2] += xtraj[t,:]' * coeff * xtraj[t,:]
            feature_counts[3] += utraj[t,1:ctrl_dim_1]' * utraj[t,1:ctrl_dim_1]
            feature_counts[4] += utraj[t,ctrl_dim_1+1:end]' * utraj[t,ctrl_dim_1+1:end]

            feature_counts[5] += xtraj[t,:]' * xtraj[t,:]
            feature_counts[6] += xtraj[t,:]' * coeff * xtraj[t,:]
            feature_counts[7] += utraj[t,1:ctrl_dim_1]' * utraj[t,1:ctrl_dim_1]
            feature_counts[8] += utraj[t,ctrl_dim_1+1:end]' * utraj[t,ctrl_dim_1+1:end]
        end
        feature_counts[1] += xtraj[end,:]' * xtraj[end,:]
        feature_counts[2] += xtraj[end,:]' * coeff * xtraj[end,:]
        feature_counts[5] += xtraj[end,:]' * xtraj[end,:]
        feature_counts[6] += xtraj[end,:]' * coeff * xtraj[end,:]
    end
    avg_feature_counts = 1.0/num * feature_counts
    return avg_feature_counts
end


function rate(lr0, itr)
    return lr0/(1.0 + 0.02 * itr)
end


function plot_lr(eta)
    fig = figure()
    ax = fig.add_subplot(111)
    rates = []

    for i = 1:5000
        push!(rates, rate(eta, i))
    end
    ax.clear()
    ax.plot(rates)
    println(rates)
    show()
end

function ma_irl(;sync_update=true, single_update=10, scale=true, eta=0.0001, plot=true, max_itr=5000)

    current_time = Dates.now()
    data = Dict()


    fig_theta, axs_theta = subplots(2, 4, figsize=(8,4))
    fig_f, axs_f = subplots(2, 4, figsize=(8,4))
    fig_dem = figure()
    ax_dem = fig_dem.add_subplot(111)
    ax_dem.axis("equal")
    ax_dem.set_xlim(-11, 11)
    ax_dem.set_ylim(-1, 11)

    dem_num = 3000
    feature_k = 8   # feature dimension

    x_init = [10 10 0 0 -10 10 0 0]
    #=============
    generate demonstrations
    ==============#
    theta_true = [5.0, 1.0, 2.0, 1.0,      5.0, 1.0, 1.0, 2.0]
    x_trajectories, u_trajectories = generate_sim(x_init, theta_true, dem_num)
    if plot
        for i = 1:length(x_trajectories)
            ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="blue")
            ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="blue")
        end
        pause(0.1)
    end

    avg_dem_feature_counts = get_feature_counts(x_trajectories, u_trajectories, feature_k)
    if scale
        scale_vector = avg_dem_feature_counts./100
    else
        scale_vector = ones(size(avg_dem_feature_counts))    # do not scale
    end


    sc_avg_dem_feature_counts = avg_dem_feature_counts./scale_vector
    println(" this is avg dem feature counts ", avg_dem_feature_counts)

    #=============
    iteration on likelihood maximization
    ==============#
    theta_curr = [3.0, 1.0, 1.0, 1.0,       3.0, 1.0, 1.0, 1.0]
    theta_avg = theta_curr


    theta_est = zeros(max_itr+1, feature_k)
    theta_est[1,:] = theta_curr
    theta_smooth = zeros(max_itr+1, feature_k)
    theta_smooth[1,:] = theta_avg
    feature_counts = zeros(max_itr, feature_k)


    num = 200
    for itr = 1:max_itr
        println(" ------------- in iteration ", itr , " ------------- ")
        avg_pro_feature_counts = zeros(feature_k)
        lr = rate(eta, itr)

        if sync_update
            x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)
            avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories, feature_k)
            sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

            theta_curr = theta_curr - lr * (sc_avg_dem_feature_counts - sc_avg_pro_feature_counts)
            theta_curr = max.(0.0, theta_curr)
        else
            # asynchronous update
            for player1_itr = 1:single_update
                # update weights on player 1
                x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)
                avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories, feature_k)
                sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

                theta_curr[1:4] = theta_curr[1:4] - lr * (sc_avg_dem_feature_counts[1:4] - sc_avg_pro_feature_counts[1:4])
                theta_curr = max.(0.0, theta_curr)
            end

            for player2_itr = 1:single_update
                # update weights on player 1
                x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)
                avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories, feature_k)
                sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

                theta_curr[5:end] = theta_curr[5:end] - lr * (sc_avg_dem_feature_counts[5:end] - sc_avg_pro_feature_counts[5:end])
                theta_curr = max.(0.0, theta_curr)
            end
        end

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

        for i = 1:feature_k
            axs_theta[i].clear()
            axs_theta[i].plot(theta_est[1:itr+1, i])
            axs_theta[i].plot(theta_smooth[1:itr+1, i])
            axs_theta[i].plot(theta_true[i] * ones(itr+1))

            axs_f[i].clear()
            axs_f[i].plot(feature_counts[1:itr, i])
            axs_f[i].plot(avg_dem_feature_counts[i] * ones(itr))
        end

        # save data along the way
        data["true theta"] = theta_true
        data["theta_est"] = theta_est[1:itr+1, :]
        data["feature_counts_demonstration"] = avg_dem_feature_counts
        data["feature_counts_proposed"] = feature_counts[1:itr, :]

#         fname = string("data/", current_time, ".jld2")
#         @save fname data
    end

end


#===========================
============================#
# ma_irl(sync_update=true, eta=0.001)
ma_irl(sync_update=true, single_update=20, scale=false, eta=0.01, plot=true, max_itr=4000)

# plot_lr(0.01)
