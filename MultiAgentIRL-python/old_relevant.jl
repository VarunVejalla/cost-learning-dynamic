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


struct SimulationResults
    state_trajectories :: Array
    ctrl_trajectories :: Array
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

function lqgame_QRE(dynamics::Dict{String, Array{Array{Float64}}},
                    costs::Dict{String, Array{Array{Float64}}})
    # Quantal Response Equilibria for Dynamic Markov Games

    # assume planning horizon T
    # return values:
    #   Ns: list len = T
    #   alphas: list len = T

    # dynamics matrices lists, each list should be of length T

    As = dynamics["A"]
    B1s = dynamics["B1"]
    B2s = dynamics["B2"]

    # cost matrices lists, Qs and ls should be of length T + 1, while Rs should be of length T
    Q1s = costs["Q1"]
    l1s = costs["l1"]
    Q2s = costs["Q2"]
    l2s = costs["l2"]
    R11s = costs["R11"]
    R12s = costs["R12"]
    R21s = costs["R21"]
    R22s = costs["R22"]

    # infer dimensions
    T = length(As)    # planning horizon
    n = size(B1s[1])[1]     # state dimension
    if(length(size(B1s[1])) == 1)
        m1 = 1
    else
        m1 = size(B1s[1])[2]    # control 1 dimension
    end
    if(length(size(B2s[1])) == 1)
        m2 = 1
    else
        m2 = size(B2s[1])[2]    # control 2 dimension
    end


    # return containers
    P1s, alpha1s, cov1s = [], [], []
    P2s, alpha2s, cov2s = [], [], []

    # intermediate computation
    Z1s, Z2s = [], []
    Fs = []
    zeta1s, zeta2s = [], []
    betas = []

    # boundary conditions
    push!(Z1s, Q1s[end])
    push!(Z2s, Q2s[end])
    push!(zeta1s, l1s[end])
    push!(zeta2s, l2s[end])

    # initialize containers
    for t = T:-1:1
        # note that in zero-based time index, the actual time index should be t-1
        # ##########################################
        # 1. solve for P_t^1 and P_t^2
        # ##########################################
        Z1_n = Z1s[end]
        Z2_n = Z2s[end]

        S = zeros(m1+m2, m1+m2)
        S[1:m1, 1:m1]             = R11s[t] + B1s[t]' * Z1_n * B1s[t]
        S[1:m1, m1+1:m1+m2]       = B1s[t]' * Z1_n * B2s[t]
        S[m1+1:m1+m2, 1:m1]       = B2s[t]' * Z2_n * B1s[t]
        S[m1+1:m1+m2, m1+1:m1+m2] = R22s[t] + B2s[t]' * Z2_n * B2s[t]


        YN = zeros(m1+m2, n)
        YN[1:m1, :]        = B1s[t]' * Z1_n * As[t]
        YN[m1+1:m1+m2, :] = B2s[t]' * Z2_n * As[t]
        P = S\YN    # least square computation
        P1 = P[1:m1,:]
        P2 = P[m1+1:m1+m2,:]
        push!(P1s, P1)
        push!(P2s, P2)
        push!(cov1s, inv(S[1:m1, 1:m1]))
        push!(cov2s, inv(S[m1+1:m1+m2, m1+1:m1+m2]))

        # ##########################################
        # 2. solve for the zero order term in FB strategy
        # ##########################################
        zeta1_n = zeta1s[end]
        zeta2_n = zeta2s[end]
        # use least square to solve for alphas, notice that the coefficient matrix (S) is the same as before
        YA = zeros(m1+m2, 1)
        YA[1:m1,:] = B1s[t]'  * zeta1_n
        YA[m1+1:m1+m2,:] = B2s[t]' * zeta2_n
        alpha = S\YA
        alpha1 = alpha[1:m1]
        alpha2 = alpha[m1+1:m1+m2]
        push!(alpha1s, alpha1)
        push!(alpha2s, alpha2)


        # ##########################################
        # 3. compute for next iteration
        # ##########################################
        # solve F for next iteration
        F = As[t] - B1s[t] * P1 - B2s[t] * P2
        push!(Fs, F)

        beta = - B1s[t] * alpha1 - B2s[t] * alpha2
        push!(betas, beta)

        zeta1 = F' * Z1_n * beta + F' * zeta1_n + P1' * R11s[t] * alpha1 + P2' * R12s[t] * alpha2 + l1s[t]
        zeta2 = F' * Z2_n * beta + F' * zeta2_n + P1' * R21s[t] * alpha1 + P2' * R22s[t] * alpha2 + l2s[t]
        push!(zeta1s, zeta1)
        push!(zeta2s, zeta2)

        Z1 = F' * Z1_n * F + P1' * R11s[t] * P1 +  P2' * R12s[t] * P2 + Q1s[t]
        Z2 = F' * Z2_n * F + P2' * R22s[t] * P2 +  P1' * R21s[t] * P1 + Q2s[t]
        push!(Z1s, Z1)
        push!(Z2s, Z2)
    end

    return P1s, P2s, alpha1s, alpha2s, cov1s, cov2s

end

#======================
Quantal equilibrium for 3 player games
=======================#
function lqgame_QRE_3player(dynamics::Dict{String, Array{Array{Float64}}},
                    costs::Dict{String, Array{Array{Float64}}})
    # Quantal Response Equilibria for Dynamic Markov Games

    # assume planning horizon T
    # return values:
    #   Ns: list len = T
    #   alphas: list len = T

    # dynamics matrices lists, each list should be of length T

    As = dynamics["A"]
    B1s = dynamics["B1"]
    B2s = dynamics["B2"]
    B3s = dynamics["B3"]

    # cost matrices lists, Qs and ls should be of length T + 1, while Rs should be of length T
    Q1s = costs["Q1"]
    l1s = costs["l1"]
    Q2s = costs["Q2"]
    l2s = costs["l2"]
    Q3s = costs["Q3"]
    l3s = costs["l3"]
    R11s = costs["R11"]
    R12s = costs["R12"]
    R13s = costs["R13"]
    R21s = costs["R21"]
    R22s = costs["R22"]
    R23s = costs["R23"]
    R31s = costs["R31"]
    R32s = costs["R32"]
    R33s = costs["R33"]

    # infer dimensions
    T = length(As)    # planning horizon
    n = size(B1s[1])[1]     # state dimension
    if(length(size(B1s[1])) == 1)
        m1 = 1
    else
        m1 = size(B1s[1])[2]    # control 1 dimension
    end
    if(length(size(B2s[1])) == 1)
        m2 = 1
    else
        m2 = size(B2s[1])[2]    # control 2 dimension
    end
    if(length(size(B3s[1])) == 1)
        m3 = 1
    else
        m3 = size(B3s[1])[2]    # control 3 dimension
    end


    # return containers
    P1s, alpha1s, cov1s = [], [], []
    P2s, alpha2s, cov2s = [], [], []
    P3s, alpha3s, cov3s = [], [], []

    # intermediate computation
    Z1s, Z2s, Z3s = [], [], []
    Fs = []
    zeta1s, zeta2s, zeta3s = [], [], []
    betas = []

    # boundary conditions
    push!(Z1s, Q1s[end])
    push!(Z2s, Q2s[end])
    push!(Z3s, Q3s[end])
    push!(zeta1s, l1s[end])
    push!(zeta2s, l2s[end])
    push!(zeta3s, l3s[end])

    # initialize containers
    for t = T:-1:1
        # note that in zero-based time index, the actual time index should be t-1
        # ##########################################
        # 1. solve for P_t^1 and P_t^2
        # ##########################################
        Z1_n = Z1s[end]
        Z2_n = Z2s[end]
        Z3_n = Z3s[end]

        S = zeros(m1+m2+m3, m1+m2+m3)
        S[1:m1, 1:m1]             = R11s[t] + B1s[t]' * Z1_n * B1s[t]
        S[1:m1, m1+1:m1+m2]       = B1s[t]' * Z1_n * B2s[t]
        S[1:m1, m1+m2+1:m1+m2+m3] = B1s[t]' * Z1_n * B3s[t]

        S[m1+1:m1+m2, 1:m1]             = B2s[t]' * Z2_n * B1s[t]
        S[m1+1:m1+m2, m1+1:m1+m2]       = R22s[t] + B2s[t]' * Z2_n * B2s[t]
        S[m1+1:m1+m2, m1+m2+1:m1+m2+m3] = B2s[t]' * Z2_n * B3s[t]

        S[m1+m2+1:end, 1:m1]             = B3s[t]' * Z3_n * B1s[t]
        S[m1+m2+1:end, m1+1:m1+m2]       = B3s[t]' * Z3_n * B2s[t]
        # println(" what is this ", size(R33s[t]), " and ", size(B3s[t]' * Z3_n * B3s[t]))
        S[m1+m2+1:end, m1+m2+1:m1+m2+m3] = R33s[t] + B3s[t]' * Z3_n * B3s[t]


        YN = zeros(m1+m2+m3, n)
        YN[1:m1, :]             = B1s[t]' * Z1_n * As[t]
        YN[m1+1:m1+m2, :]       = B2s[t]' * Z2_n * As[t]
        YN[m1+m2+1:m1+m2+m3, :] = B3s[t]' * Z3_n * As[t]
        P = S\YN    # least square computation
        P1 = P[1:m1,:]
        P2 = P[m1+1:m1+m2,:]
        P3 = P[m1+m2+1:m1+m2+m3,:]
        push!(P1s, P1)
        push!(P2s, P2)
        push!(P3s, P3)
        push!(cov1s, inv(S[1:m1, 1:m1]))
        push!(cov2s, inv(S[m1+1:m1+m2, m1+1:m1+m2]))
        push!(cov3s, inv(S[m1+m2+1:m1+m2+m3, m1+m2+1:m1+m2+m3]))

        # ##########################################
        # 2. solve for the zero order term in FB strategy
        # ##########################################
        zeta1_n = zeta1s[end]
        zeta2_n = zeta2s[end]
        zeta3_n = zeta3s[end]
        # use least square to solve for alphas, notice that the coefficient matrix (S) is the same as before
        YA = zeros(m1+m2+m3, 1)
        YA[1:m1,:]             = B1s[t]' * zeta1_n
        YA[m1+1:m1+m2,:]       = B2s[t]' * zeta2_n
        YA[m1+m2+1:m1+m2+m3,:] = B3s[t]' * zeta3_n
        alpha = S\YA
        alpha1 = alpha[1:m1]
        alpha2 = alpha[m1+1:m1+m2]
        alpha3 = alpha[m1+m2+1:m1+m2+m3]
        push!(alpha1s, alpha1)
        push!(alpha2s, alpha2)
        push!(alpha3s, alpha3)


        # ##########################################
        # 3. compute for next iteration
        # ##########################################
        # solve F for next iteration
        F = As[t] - B1s[t] * P1 - B2s[t] * P2 - B3s[t] * P3
        push!(Fs, F)

        beta = - B1s[t] * alpha1 - B2s[t] * alpha2 - B3s[t] * alpha3
        push!(betas, beta)

        zeta1 = F' * Z1_n * beta + F' * zeta1_n + P1' * R11s[t] * alpha1 + P2' * R12s[t] * alpha2 + P3' * R13s[t] * alpha3 + l1s[t]
        zeta2 = F' * Z2_n * beta + F' * zeta2_n + P1' * R21s[t] * alpha1 + P2' * R22s[t] * alpha2 + P3' * R23s[t] * alpha3 + l2s[t]
        zeta3 = F' * Z3_n * beta + F' * zeta3_n + P1' * R31s[t] * alpha1 + P2' * R32s[t] * alpha2 + P3' * R33s[t] * alpha3 + l3s[t]
        push!(zeta1s, zeta1)
        push!(zeta2s, zeta2)
        push!(zeta3s, zeta3)

        Z1 = F' * Z1_n * F + P1' * R11s[t] * P1 +  P2' * R12s[t] * P2 + P3' * R13s[t] * P3 + Q1s[t]
        Z2 = F' * Z2_n * F + P1' * R21s[t] * P1 +  P2' * R22s[t] * P2 + P3' * R23s[t] * P3 + Q2s[t]
        Z3 = F' * Z3_n * F + P1' * R31s[t] * P1 +  P2' * R32s[t] * P2 + P3' * R33s[t] * P3 + Q3s[t]
        push!(Z1s, Z1)
        push!(Z2s, Z2)
        push!(Z3s, Z3)
    end

    return P1s, P2s, P3s, alpha1s, alpha2s, alpha3s, cov1s, cov2s, cov3s

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
                push!(Costs["R21"], hess2[state_dim*2+1:state_dim*2+ctrl_dim1, state_dim*2+1:state_dim*2+ctrl_dim1])
                push!(Costs["R31"], hess3[state_dim*2+1:state_dim*2+ctrl_dim1, state_dim*2+1:state_dim*2+ctrl_dim1])

                push!(Costs["R12"], hess1[state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2,
                                          state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2])
                push!(Costs["R13"], hess1[state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim,
                                          state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim])
                
                push!(Costs["R22"], hess2[state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2,
                                          state_dim*2+ctrl_dim1+1:state_dim*2+ctrl_dim1+ctrl_dim2])
                push!(Costs["R23"], hess2[state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim,
                                          state_dim*2+ctrl_dim1+ctrl_dim2+1:state_dim*2+ctrl_dim])
                

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

