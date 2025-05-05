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
    dem_num=200,
    noise_std=0.1) # New parameter for Gaussian noise standard deviation

    steps = 60
    horizon = 6.0
    plan_steps = 10

    state_dims = [4, 4]
    ctrl_dims = [2, 2]
    DT = horizon/steps

    # cost coefficients, state tracking, control penalty, and collision avoidance
    theta_true = [1.0, 1.0, 8.0, 0.5, 0.5, 10.0]

    sim_param = SimulationParams(steps=steps, horizon=horizon, plan_steps=plan_steps)
    game = NonlinearGame(state_dims, ctrl_dims, DT; theta=theta_true)
    # save file and figures
    current_time = Dates.now()

    fig_theta, axs_theta = subplots(2, 3, figsize=(6,4))
    fig_f, axs_f = subplots(2, 3, figsize=(6,4))
    fig_dem, ax_dem = subplots(1, 1)
    ax_dem.axis("equal")
    ax_dem.set_xlim(-5, 5)
    ax_dem.set_ylim(-5, 5)

    # Base initial state
    x_init_base = [-4.0 0.0 1.33 0.0 0.0 -4.0 1.33 pi/2]
    x_ref = zeros(steps*2, game.state_dim)
    x_ref[1,:] = x_init_base
    for i=1:steps*2 - 1
        x_ref[i+1,:] = game.dynamics_func([x_ref[i,:]; zeros(game.ctrl_dim)])
    end

    #=============#
    # Generate demonstrations with Gaussian noise on x_init
    #=============#
    demo_data = zeros(dem_num, steps+1, game.state_dim * 2+game.ctrl_dim)
    feature_k = length(theta_true)

    # Initialize arrays to store trajectories
    x_trajectories = []
    u_trajectories = []

    # Define Gaussian noise distribution
    noise_dist = Normal(0, noise_std)

    # Generate simulations with noisy x_init for each demonstration
    for di = 1:dem_num
        # Create noisy x_init for this simulation
        println("x_init_base=", x_init_base)
        println("x_init_base.size=", x_init_base.size)
        # println("rand=", rand(noise_dist, length(x_init_base)))
        # println("sum=", x_init_base .+ rand(noise_dist, length(x_init_base)))
        # x_init_noisy = x_init_base .+ rand(noise_dist, length(x_init_base))
        # println("x_init_noisy=", x_init_noisy)
        # println("x_init_noisy.size=", x_init_noisy.size)
        x_init_noisy = x_init_base + reshape(rand(noise_dist, length(x_init_base)), 1, 8)

        # Run simulation with noisy x_init
        dem_results, x_trajectories_data, u_trajectories_data = generate_simulations(
        sim_param=sim_param,
        nl_game=game,
        x_init=x_init_noisy, # Pass noisy initial state
        traj_ref=x_ref,
        num_sim=2 # Run one simulation at a time
        )

        # Store trajectories
        push!(x_trajectories, x_trajectories_data[1])
        push!(u_trajectories, u_trajectories_data[1])

        # Store in demo_data
        println("demo_data[di,:,1:game.state_dim].size=", demo_data[di,:,1:game.state_dim].size)
        println("x_trajectories_data[1].size=", x_trajectories_data[1].size)
        demo_data[di,:,1:game.state_dim] = x_trajectories_data[1]
        demo_data[di,:,1+game.state_dim:game.state_dim*2] = x_ref[1:steps+1,:]
        demo_data[di,:,game.state_dim*2+1:end] = u_trajectories_data[1]
    end

    # Plot trajectories
    if plot
        for i = 1:length(x_trajectories)
            ax_dem.plot(x_trajectories[i][:,1], x_trajectories[i][:,2], alpha=0.3, color="blue")
            ax_dem.plot(x_trajectories[i][:,5], x_trajectories[i][:,6], alpha=0.3, color="blue")
        end
        pause(500)
    end

    avg_dem_feature_counts = get_feature_counts(game, x_trajectories, u_trajectories, x_ref, feature_k)
    if scale
    scale_vector = avg_dem_feature_counts./100
    else
    scale_vector = ones(size(avg_dem_feature_counts)) # do not scale
    end

sc_avg_dem_feature_counts = avg_dem_feature_counts./scale_vector
println("this is avg dem feature counts ", avg_dem_feature_counts)

fname = string("cioc_data/twoplayer-noisy.h5")
@save fname demo_data
end

#===========================#
# Main
#============================#
ma_irl(sync_update=false,
single_update=1,
scale=true,
eta=0.01,
max_itr=5000,
sample_size=10,
dem_num=500,
plot=true,
noise_std=1) # Add noise_std parameter