# simple.jl
# mingyuw@stanford.edu


# note: this problem setup decouples the game, i.e. the policy of one agent does not depend on the state of
# the other agent, need to check through math too...

using PyPlot
using LinearAlgebra, Distributions

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
    w_state1 = theta[1] * Matrix{Float64}(I, state_dim, state_dim)
    w_ctrl11 = theta[2] * Matrix{Float64}(I, ctrl_dim_1, ctrl_dim_1)
    w_ctrl12 = theta[3] * Matrix{Float64}(I, ctrl_dim_2, ctrl_dim_2)

    w_state2 = theta[4] * Matrix{Float64}(I, state_dim, state_dim)
    w_ctrl21 = theta[5] * Matrix{Float64}(I, ctrl_dim_1, ctrl_dim_1)
    w_ctrl22 = theta[6] * Matrix{Float64}(I, ctrl_dim_2, ctrl_dim_2)

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

function return_policy(x_init, theta, num = 200)

    # compute the Quantal response equilibrium
    Dynamics, Costs = set_up_system(theta)
    N1, N2, alpha1, alpha2, cov1, cov2 = lqgame_QRE(Dynamics, Costs)
    return N1, N2, alpha1, alpha2, cov1, cov2
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



function main()
    fig = figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.axis("equal")
    ax.set_xlim(-11, 11)
    ax.set_ylim(-1, 11)


    x_init = [10 10 0 0 -10 10 0 0]
    Dynamics, Costs = set_up_system()
    N1, N2, alpha1, alpha2 = lqgame(Dynamics, Costs)


    u_history = zeros(plan_steps, ctrl_dim)
    x_history = zeros(plan_steps+1, state_dim)
    x_history[1,:] = x_init
    for t = 1:plan_steps
        u_history[t,1:ctrl_dim_1] = -N1[end-t+1] * x_history[t, :] - alpha1[end-t+1]
        u_history[t,ctrl_dim_1+1:ctrl_dim_1+ctrl_dim_2] = -N2[end-t+1] * x_history[t, :] - alpha2[end-t+1]
        x_history[t+1, :] = dynamics_forward([x_history[t, :]; u_history[t, :]])
    end
    ax.plot(x_history[:,1], x_history[:,2])
    ax.plot(x_history[:,5], x_history[:,6])
    pause(1.0)
    show()
end

function get_feature_counts(x_trajectories, u_trajectories)
    feature_counts = zeros(6)
    num = length(x_trajectories)

    for i = 1:num
        xtraj = x_trajectories[i]
        utraj = u_trajectories[i]
        for t = 1:plan_steps
            feature_counts[1] += xtraj[t,:]' * xtraj[t,:]
            feature_counts[2] += utraj[t,1:ctrl_dim_1]' * utraj[t,1:ctrl_dim_1]
            feature_counts[3] += utraj[t,ctrl_dim_1+1:end]' * utraj[t,ctrl_dim_1+1:end]

            feature_counts[4] += xtraj[t,:]' * xtraj[t,:]
            feature_counts[5] += utraj[t,1:ctrl_dim_1]' * utraj[t,1:ctrl_dim_1]
            feature_counts[6] += utraj[t,ctrl_dim_1+1:end]' * utraj[t,ctrl_dim_1+1:end]
        end
        feature_counts[1] += xtraj[end,:]' * xtraj[end,:]
        feature_counts[4] += xtraj[end,:]' * xtraj[end,:]
    end
    avg_feature_counts = 1.0/num * feature_counts
    return avg_feature_counts
end

function ma_irl()
    fig = figure(figsize=(6,4))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    fig_f = figure(figsize=(6,4))
    ax1_f = fig_f.add_subplot(231)
    ax2_f = fig_f.add_subplot(232)
    ax3_f = fig_f.add_subplot(233)
    ax4_f = fig_f.add_subplot(234)
    ax5_f = fig_f.add_subplot(235)
    ax6_f = fig_f.add_subplot(236)

    fig_dem = figure()
    ax_dem = fig_dem.add_subplot(111)
    ax_dem.axis("equal")
    ax_dem.set_xlim(-11, 11)
    ax_dem.set_ylim(-1, 11)

    dem_num = 1000
    feature_k = 6   # feature dimension

    x_init = [10 10 0 0 -10 10 0 0]
    #=============
    generate demonstrations
    ==============#
    theta_true = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    x_trajectories, u_trajectories = generate_sim(x_init, theta_true, dem_num)
    # N11, N21, alpha11, alpha21, cov11, cov21 = return_policy(x_init, theta_true, num)
    for i = 1:length(x_trajectories)
        ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="blue")
        ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="blue")
    end
    pause(1.0)
    # show()
    avg_dem_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
    scale_vector = avg_dem_feature_counts./1000
    sc_avg_dem_feature_counts = avg_dem_feature_counts./scale_vector    # should all be 1000
    println(" this is avg dem feature counts ", avg_dem_feature_counts)

    # # for bebugging and comparison
    # x_init = [10 10 0 0 -10 10 0 0]
    #=============
    generate demonstrations
    ==============#
    # theta_true = [1.0, 2.0, 1.0, 1.0, 2.0, 0.5]
    # x_trajectories, u_trajectories = generate_sim(x_init, theta_true, num)
    # # N12, N22, alpha12, alpha22, cov12, cov22 = return_policy(x_init, theta_true, num)
    # for i = 1:length(x_trajectories)
    #     ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="green")
    #     ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="green")
    # end
    # avg_dem_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
    # println(" this is avg dem feature counts ", avg_dem_feature_counts)
    # pause(1.0)
    #
    # theta_true = [1.0, 2.0, 1.0, 1.0, 5.0, 0.5]
    # x_trajectories, u_trajectories = generate_sim(x_init, theta_true, num)
    # # N13, N23, alpha13, alpha23, cov13, cov23 = return_policy(x_init, theta_true, num)
    # for i = 1:length(x_trajectories)
    #     ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="pink")
    #     ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="pink")
    # end
    # avg_dem_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
    # println(" this is avg dem feature counts ", avg_dem_feature_counts)
    # pause(1.0)

    # println(" first ", N11== N12 && N12 == N13)
    # println(" SECOND ", N21== N22 && N22 == N23)
    # println(" THIRD ", alpha11 == alpha12 && alpha12 == alpha13)
    # println(" forth ", alpha21 == alpha22 && alpha22 == alpha23)
    # println(" fifth ", cov11 == cov12 && cov12 == cov13)
    # println(" sixth ", cov21 == cov22 && cov22 == cov23)

    #=============
    iteration on likelihood maximization
    ==============#
    theta_curr = [1.5, 1.0, 1.0, 1.5, 1.0, 1.0]
    theta1 = [theta_curr[1]]
    theta2 = [theta_curr[2]]
    theta3 = [theta_curr[3]]
    theta4 = [theta_curr[4]]
    theta5 = [theta_curr[5]]
    theta6 = [theta_curr[6]]

    fc1 = []
    fc2 = []
    fc3 = []
    fc4 = []
    fc5 = []
    fc6 = []

    num = 200
    max_itr = 500
    for itr = 1:max_itr
        println(" ------------- in iteration ", itr , " ------------- ")
        eta = 0.0001
        avg_pro_feature_counts = zeros(feature_k)


        for player1_itr = 20
            # update weights on player 1
            x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)
            avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
            sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector


            theta_curr[1:3] = theta_curr[1:3] - eta * (sc_avg_dem_feature_counts[1:3] - sc_avg_pro_feature_counts[1:3])
            # theta_curr = theta_curr - eta * (avg_dem_feature_counts - avg_pro_feature_counts)
            # println("       resulting theta ", theta_curr)
            theta_curr = max.(0.0, theta_curr)
        end
        push!(fc1, avg_pro_feature_counts[1])
        push!(fc2, avg_pro_feature_counts[2])
        push!(fc3, avg_pro_feature_counts[3])
        push!(fc4, avg_pro_feature_counts[4])
        push!(fc5, avg_pro_feature_counts[5])
        push!(fc6, avg_pro_feature_counts[6])
        println("            this is avg dem feature counts ", avg_dem_feature_counts)
        println("       this is avg proposed feature counts ", avg_pro_feature_counts)
        push!(theta1, theta_curr[1])
        push!(theta2, theta_curr[2])
        push!(theta3, theta_curr[3])
        push!(theta4, theta_curr[4])
        push!(theta5, theta_curr[5])
        push!(theta6, theta_curr[6])


        for player2_itr = 20
            # update weights on player 1
            x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)
            avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
            sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector


            theta_curr[4:6] = theta_curr[4:6] - eta * (sc_avg_dem_feature_counts[4:6] - sc_avg_pro_feature_counts[4:6])
            # theta_curr = theta_curr - eta * (avg_dem_feature_counts - avg_pro_feature_counts)
            # println("       resulting theta ", theta_curr)
            theta_curr = max.(0.0, theta_curr)
        end
        push!(fc1, avg_pro_feature_counts[1])
        push!(fc2, avg_pro_feature_counts[2])
        push!(fc3, avg_pro_feature_counts[3])
        push!(fc4, avg_pro_feature_counts[4])
        push!(fc5, avg_pro_feature_counts[5])
        push!(fc6, avg_pro_feature_counts[6])
        println("            this is avg dem feature counts ", avg_dem_feature_counts)
        println("       this is avg proposed feature counts ", avg_pro_feature_counts)
        push!(theta1, theta_curr[1])
        push!(theta2, theta_curr[2])
        push!(theta3, theta_curr[3])
        push!(theta4, theta_curr[4])
        push!(theta5, theta_curr[5])
        push!(theta6, theta_curr[6])

        if true
            ax1.clear()
            ax1.plot(theta1)
            ax1.plot(theta_true[1] * ones(length(theta1)))
            ax2.clear()
            ax2.plot(theta2)
            ax2.plot(theta_true[2] * ones(length(theta2)))
            ax3.clear()
            ax3.plot(theta3)
            ax3.plot(theta_true[3] * ones(length(theta3)))
            ax4.clear()
            ax4.plot(theta4)
            ax4.plot(theta_true[4] * ones(length(theta4)))
            ax5.clear()
            ax5.plot(theta5)
            ax5.plot(theta_true[5] * ones(length(theta5)))
            ax6.clear()
            ax6.plot(theta6)
            ax6.plot(theta_true[6] * ones(length(theta6)))

            ax1_f.clear()
            ax1_f.plot(fc1)
            ax1_f.plot(avg_dem_feature_counts[1] * ones(length(fc1)))
            ax2_f.clear()
            ax2_f.plot(fc2)
            ax2_f.plot(avg_dem_feature_counts[2] * ones(length(fc2)))
            ax3_f.clear()
            ax3_f.plot(fc3)
            ax3_f.plot(avg_dem_feature_counts[3] * ones(length(fc3)))
            ax4_f.clear()
            ax4_f.plot(fc4)
            ax4_f.plot(avg_dem_feature_counts[4] * ones(length(fc4)))
            ax5_f.clear()
            ax5_f.plot(fc5)
            ax5_f.plot(avg_dem_feature_counts[5] * ones(length(fc5)))
            ax6_f.clear()
            ax6_f.plot(fc6)
            ax6_f.plot(avg_dem_feature_counts[6] * ones(length(fc6)))
            pause(1.0)
        end
    end
end


#===========================
============================#
ma_irl()
