# simple.jl
# mingyuw@stanford.edu

# note: this problem setup decouples the game, i.e. the policy of one agent does not depend on the state of
# the other agent, need to check through math too...

using PyCall
using PyPlot
using LinearAlgebra, Distributions

cost_encoder = pyimport("../encoders/cost_encoder")  # Import Python module for cost encoding
CostEncoder = py"CostEncoder"
instance = CostEncoder("World")

include("lqgame.jl")  # Include external file defining LQ game functions (e.g., lqgame, lqgame_QRE)

# Define dimensions for states and controls of two agents
state_dim_1 = 4      # State dimension for agent 1 (e.g., x, y, vx, vy)
ctrl_dim_1 = 2       # Control dimension for agent 1 (e.g., ax, ay)
state_dim_2 = 4      # State dimension for agent 2
ctrl_dim_2 = 2       # Control dimension for agent 2
state_dim = state_dim_1 + state_dim_2  # Total state dimension
ctrl_dim = ctrl_dim_1 + ctrl_dim_2     # Total control dimension

# Simulation parameters
plan_steps = 20      # Number of discrete time steps
horizon = 2.0        # Total time horizon in seconds
DT = horizon / plan_steps  # Time step size

#=
coefficients for dynamics
=#
# Initialize system dynamics matrix A (state transition)
A = zeros(state_dim, state_dim)
A[1:state_dim_1, 1:state_dim_1] = [1 0 DT 0; 0 1 0 DT; 0 0 1 0; 0 0 0 1]  # Agent 1: double integrator dynamics
A[state_dim_1+1:end, state_dim_1+1:end] = [1 0 DT 0; 0 1 0 DT; 0 0 1 0; 0 0 0 1]  # Agent 2: same

# Control input matrices for each agent
B1 = zeros(state_dim, ctrl_dim_1)  # Effect of agent 1's control on full state
B1[1:state_dim_1, :] = [0 0; 0 0; DT 0; 0 DT]  # Agent 1: control affects velocity
B2 = zeros(state_dim, ctrl_dim_2)  # Effect of agent 2's control on full state
B2[state_dim_1+1:end, :] = [0 0; 0 0; DT 0; 0 DT]  # Agent 2: same

# Dynamics function: computes next state given current state and controls
function dynamics_forward(s)
    state = s[1:state_dim]  # Extract state vector
    ctrl1 = s[state_dim+1:state_dim+ctrl_dim_1]  # Agent 1's control
    ctrl2 = s[state_dim+ctrl_dim_1+1:end]  # Agent 2's control
    return A * state + B1 * ctrl1 + B2 * ctrl2  # Linear dynamics: x_{t+1} = A*x_t + B1*u1_t + B2*u2_t
end

function set_up_system_nn(state, actions, encoder) 
    # Call Python NN via PyCall
    w = py"encoder(torch.tensor($state), torch.tensor($actions)).detach().numpy()"
    w_state1 = w[1] * Matrix{Float64}(I, state_dim, state_dim)
    w_ctrl11 = w[2] * Matrix{Float64}(I, ctrl_dim_1, ctrl_dim_1)
    # ... (similar for other weights)
    # Return Dynamics, Costs as before
end

# Set up dynamics and cost matrices for the LQ game based on weights theta
function set_up_system(theta)
    # Cost weights for agent 1
    w_state1 = theta[1] * Matrix{Float64}(I, state_dim, state_dim)  # State cost matrix Q1
    w_ctrl11 = theta[2] * Matrix{Float64}(I, ctrl_dim_1, ctrl_dim_1)  # Control cost matrix R11
    w_ctrl12 = theta[3] * Matrix{Float64}(I, ctrl_dim_2, ctrl_dim_2)  # Cross-term cost R12 (unused in decoupled game)

    # Cost weights for agent 2
    w_state2 = theta[4] * Matrix{Float64}(I, state_dim, state_dim)  # State cost matrix Q2
    w_ctrl21 = theta[5] * Matrix{Float64}(I, ctrl_dim_1, ctrl_dim_1)  # Cross-term cost R21 (unused)
    w_ctrl22 = theta[6] * Matrix{Float64}(I, ctrl_dim_2, ctrl_dim_2)  # Control cost matrix R22

    # Dynamics dictionary: constant A, B1, B2 over all time steps
    Dynamics = Dict{String, Array{Array{Float64}}}()
    Dynamics["A"] = [A for i = 1:plan_steps]
    Dynamics["B1"] = [B1 for i = 1:plan_steps]
    Dynamics["B2"] = [B2 for i = 1:plan_steps]

    # Costs dictionary: constant cost matrices over horizon
    Costs = Dict{String, Array{Array{Float64}}}()
    Costs["Q1"] = [w_state1 for i = 1:plan_steps+1]  # State cost for agent 1
    Costs["l1"] = [zeros(state_dim) for i = 1:plan_steps+1]  # Linear state cost (zero here)
    Costs["Q2"] = [w_state2 for i = 1:plan_steps+1]  # State cost for agent 2
    Costs["l2"] = [zeros(state_dim) for i = 1:plan_steps+1]  # Linear state cost (zero)
    Costs["R11"] = [w_ctrl11 for i = 1:plan_steps]  # Control cost for agent 1
    Costs["R12"] = [w_ctrl12 for i = 1:plan_steps]  # Cross-term (decoupled, likely unused)
    Costs["R21"] = [w_ctrl21 for i = 1:plan_steps]  # Cross-term (decoupled, likely unused)
    Costs["R22"] = [w_ctrl22 for i = 1:plan_steps]  # Control cost for agent 2
    return Dynamics, Costs
end

# Compute Quantal Response Equilibrium (QRE) policies for both agents
function return_policy(x_init, theta, num = 200)
    Dynamics, Costs = set_up_system(theta)  # Set up system with given weights
    # Compute QRE: feedback gains (N) and offsets (alpha) with covariances
    N1, N2, alpha1, alpha2, cov1, cov2 = lqgame_QRE(Dynamics, Costs)
    return N1, N2, alpha1, alpha2, cov1, cov2  # Return policy parameters
end

# Generate simulated trajectories using QRE policies
function generate_sim(x_init, theta, num = 200)
    x_trajectories, u_trajectories = [], []  # Store state and control trajectories

    Dynamics, Costs = set_up_system(theta)  # Set up system
    N1, N2, alpha1, alpha2, cov1, cov2 = lqgame_QRE(Dynamics, Costs)  # Get QRE policies

    # Simulate 'num' trajectories
    for i = 1:num
        x_history = zeros(plan_steps+1, state_dim)  # State history
        x_history[1, :] = x_init  # Initial state
        u_history = zeros(plan_steps, ctrl_dim)  # Control history
        for t = 1:plan_steps
            # Compute mean control for agent 1 (feedback policy)
            u_mean1 = -N1[end-t+1] * x_history[t, :] - alpha1[end-t+1]
            u_dist1 = MvNormal(u_mean1, cov1[end-t+1])  # Noisy control distribution
            # Compute mean control for agent 2
            u_mean2 = -N2[end-t+1] * x_history[t, :] - alpha2[end-t+1]
            u_dist2 = MvNormal(u_mean2, cov2[end-t+1])  # Noisy control distribution

            # Sample controls and concatenate
            u_history[t, :] = [rand(u_dist1)..., rand(u_dist2)...]
            # Update state using dynamics
            x_history[t+1, :] = dynamics_forward([x_history[t, :]; u_history[t, :]])
        end
        push!(x_trajectories, x_history)
        push!(u_trajectories, u_history)
    end
    return x_trajectories, u_trajectories  # Return all trajectories
end

# Simulate and plot a single deterministic trajectory (no noise)
function main()
    fig = figure(figsize=(5, 5))  # Create figure
    ax = fig.add_subplot(111)  # Add single subplot
    ax.axis("equal")  # Equal aspect ratio
    ax.set_xlim(-11, 11)  # X-axis limits
    ax.set_ylim(-1, 11)   # Y-axis limits

    x_init = [10 10 0 0 -10 10 0 0]  # Initial state: [x1 y1 vx1 vy1 x2 y2 vx2 vy2]
    Dynamics, Costs = set_up_system([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Default weights
    N1, N2, alpha1, alpha2 = lqgame(Dynamics, Costs)  # Compute deterministic LQ policies

    u_history = zeros(plan_steps, ctrl_dim)  # Control history
    x_history = zeros(plan_steps+1, state_dim)  # State history
    x_history[1, :] = x_init  # Set initial state
    for t = 1:plan_steps
        # Apply feedback policies for both agents
        u_history[t, 1:ctrl_dim_1] = -N1[end-t+1] * x_history[t, :] - alpha1[end-t+1]
        u_history[t, ctrl_dim_1+1:ctrl_dim] = -N2[end-t+1] * x_history[t, :] - alpha2[end-t+1]
        x_history[t+1, :] = dynamics_forward([x_history[t, :]; u_history[t, :]])  # Update state
    end
    # Plot trajectories with legends
    ax.plot(x_history[:, 1], x_history[:, 2], label="Agent 1")  # Agent 1 position (x1, y1)
    ax.plot(x_history[:, 5], x_history[:, 6], label="Agent 2")  # Agent 2 position (x2, y2)
    # ax.legend()  # Add legend to distinguish agents
    pause(1.0)  # Pause to display
    show()  # Show plot
end

# Compute average feature counts from trajectories (for IRL)
function get_feature_counts(x_trajectories, u_trajectories)
    feature_counts = zeros(6)  # Features: [state1, ctrl1, ctrl2, state2, ctrl1, ctrl2]
    num = length(x_trajectories)  # Number of trajectories

    for i = 1:num
        xtraj = x_trajectories[i]  # State trajectory
        utraj = u_trajectories[i]  # Control trajectory
        for t = 1:plan_steps
            # Feature 1: Agent 1 state cost (sum of squared states)
            feature_counts[1] += xtraj[t,:]' * xtraj[t,:]
            # Feature 2: Agent 1 control cost (u1 squared)
            feature_counts[2] += utraj[t,1:ctrl_dim_1]' * utraj[t,1:ctrl_dim_1]
            # Feature 3: Agent 1 control cost (u2 squared)
            feature_counts[3] += utraj[t,ctrl_dim_1+1:end]' * utraj[t,ctrl_dim_1+1:end]

            # Feature 4: Agent 2 state cost (sum of squared states)
            feature_counts[4] += xtraj[t,:]' * xtraj[t,:]
            # Feature 5: Agent 2 control cost (u1 squared)
            feature_counts[5] += utraj[t,1:ctrl_dim_1]' * utraj[t,1:ctrl_dim_1]
            # Feature 6: Agent 2 control cost (u2 squared)
            feature_counts[6] += utraj[t,ctrl_dim_1+1:end]' * utraj[t,ctrl_dim_1+1:end]
        end
        # Include terminal state
        feature_counts[1] += xtraj[end, :]' * xtraj[end, :]
        feature_counts[4] += xtraj[end, :]' * xtraj[end, :]
    end
    avg_feature_counts = 1.0 / num * feature_counts  # Average over trajectories
    return avg_feature_counts
end

# Multi-agent IRL to learn theta from demonstrations
function ma_irl()
    # Set up plots for theta convergence
    fig = figure(figsize=(6, 4))
    ax1 = fig.add_subplot(231)  # theta[1]
    ax2 = fig.add_subplot(232)  # theta[2]
    ax3 = fig.add_subplot(233)  # theta[3]
    ax4 = fig.add_subplot(234)  # theta[4]
    ax5 = fig.add_subplot(235)  # theta[5]
    ax6 = fig.add_subplot(236)  # theta[6]

    # Set up plots for feature counts
    fig_f = figure(figsize=(6, 4))
    ax1_f = fig_f.add_subplot(231)  # feature 1
    ax2_f = fig_f.add_subplot(232)  # feature 2
    ax3_f = fig_f.add_subplot(233)  # feature 3
    ax4_f = fig_f.add_subplot(234)  # feature 4
    ax5_f = fig_f.add_subplot(235)  # feature 5
    ax6_f = fig_f.add_subplot(236)  # feature 6

    # Set up demonstration plot
    fig_dem = figure()
    ax_dem = fig_dem.add_subplot(111)
    ax_dem.axis("equal")
    ax_dem.set_xlim(-11, 11)
    ax_dem.set_ylim(-1, 11)

    dem_num = 1000  # Number of demonstration trajectories
    feature_k = 6   # Number of features

    x_init = [10 10 0 0 -10 10 0 0]  # Initial state
    #=============# 
    # Generate demonstrations
    #=============# 
    theta_true = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # True cost weights
    x_trajectories, u_trajectories = generate_sim(x_init, theta_true, dem_num)  # Generate demos
    for i = 1:length(x_trajectories)
        # Plot demonstration trajectories with transparency
        ax_dem.plot(x_trajectories[i][:, 1], x_trajectories[i][:, 2], alpha=0.3, color="blue", label="Agent 1" * (i == 1 ? "" : "hidden"))
        ax_dem.plot(x_trajectories[i][:, 5], x_trajectories[i][:, 6], alpha=0.3, color="red", label="Agent 2" * (i == 1 ? "" : "hidden"))
    end
    # ax_dem.legend()  # Add legend to distinguish agents
    pause(1.0)

    avg_dem_feature_counts = get_feature_counts(x_trajectories, u_trajectories)  # Demo feature counts
    scale_vector = avg_dem_feature_counts ./ 1000  # Scaling factor
    sc_avg_dem_feature_counts = avg_dem_feature_counts ./ scale_vector  # Scaled features
    println(" this is avg dem feature counts ", avg_dem_feature_counts)

    # Initial guess for theta
    theta_curr = [1.5, 1.0, 1.0, 1.5, 1.0, 1.0]
    theta1 = [theta_curr[1]]  # History of each theta component
    theta2 = [theta_curr[2]]
    theta3 = [theta_curr[3]]
    theta4 = [theta_curr[4]]
    theta5 = [theta_curr[5]]
    theta6 = [theta_curr[6]]
    # theta[1] weights Feature 1 (w_state1).
    # theta[2] weights Feature 2 (w_ctrl11).
    # theta[3] weights Feature 3 (w_ctrl12).
    # theta[4] weights Feature 4 (w_state2).
    # theta[5] weights Feature 5 (w_ctrl21).
    # theta[6] weights Feature 6 (w_ctrl22).

    fc1, fc2, fc3, fc4, fc5, fc6 = [], [], [], [], [], []  # Feature count histories

    num = 200  # Number of simulations per iteration
    max_itr = 500  # Max IRL iterations
    for itr = 1:max_itr
        println(" ------------- in iteration ", itr, " ------------- ")
        eta = 0.0001  # Learning rate
        avg_pro_feature_counts = zeros(feature_k)  # Proposed feature counts

        # Update weights for agent 1
        for player1_itr = 20
            x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)  # Simulate
            avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories)  # Compute features
            sc_avg_pro_feature_counts = avg_pro_feature_counts ./ scale_vector  # Scale features
            # Gradient descent on agent 1's weights
            theta_curr[1:3] = theta_curr[1:3] - eta * (sc_avg_dem_feature_counts[1:3] - sc_avg_pro_feature_counts[1:3])
            theta_curr = max.(0.0, theta_curr)  # Ensure non-negative weights
        end
        # Store feature counts and theta
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

        # Update weights for agent 2
        for player2_itr = 20
            x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)
            avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
            sc_avg_pro_feature_counts = avg_pro_feature_counts ./ scale_vector
            # Gradient descent on agent 2's weights
            theta_curr[4:6] = theta_curr[4:6] - eta * (sc_avg_dem_feature_counts[4:6] - sc_avg_pro_feature_counts[4:6])
            theta_curr = max.(0.0, theta_curr)
        end
        # Store feature counts and theta
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

        # Update plots
        if true
            # Theta plots
            ax1.clear()
            ax1.plot(theta1, label="Learned")
            ax1.plot(theta_true[1] * ones(length(theta1)), label="True")
            ax1.legend()
            ax2.clear()
            ax2.plot(theta2, label="Learned")
            ax2.plot(theta_true[2] * ones(length(theta2)), label="True")
            ax2.legend()
            ax3.clear()
            ax3.plot(theta3, label="Learned")
            ax3.plot(theta_true[3] * ones(length(theta3)), label="True")
            ax3.legend()
            ax4.clear()
            ax4.plot(theta4, label="Learned")
            ax4.plot(theta_true[4] * ones(length(theta4)), label="True")
            ax4.legend()
            ax5.clear()
            ax5.plot(theta5, label="Learned")
            ax5.plot(theta_true[5] * ones(length(theta5)), label="True")
            ax5.legend()
            ax6.clear()
            ax6.plot(theta6, label="Learned")
            ax6.plot(theta_true[6] * ones(length(theta6)), label="True")
            ax6.legend()

            # Feature count plots
            ax1_f.clear()
            ax1_f.plot(fc1, label="Proposed")
            ax1_f.plot(avg_dem_feature_counts[1] * ones(length(fc1)), label="Demo")
            ax1_f.legend()
            ax2_f.clear()
            ax2_f.plot(fc2, label="Proposed")
            ax2_f.plot(avg_dem_feature_counts[2] * ones(length(fc2)), label="Demo")
            ax2_f.legend()
            ax3_f.clear()
            ax3_f.plot(fc3, label="Proposed")
            ax3_f.plot(avg_dem_feature_counts[3] * ones(length(fc3)), label="Demo")
            ax3_f.legend()
            ax4_f.clear()
            ax4_f.plot(fc4, label="Proposed")
            ax4_f.plot(avg_dem_feature_counts[4] * ones(length(fc4)), label="Demo")
            ax4_f.legend()
            ax5_f.clear()
            ax5_f.plot(fc5, label="Proposed")
            ax5_f.plot(avg_dem_feature_counts[5] * ones(length(fc5)), label="Demo")
            ax5_f.legend()
            ax6_f.clear()
            ax6_f.plot(fc6, label="Proposed")
            ax6_f.plot(avg_dem_feature_counts[6] * ones(length(fc6)), label="Demo")
            ax6_f.legend()
            pause(1.0)  # Update display
        end
    end
end

#============================#
ma_irl()  # Run the IRL algorithm