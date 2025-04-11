In the provided simple.jl code, the inverse reinforcement learning (IRL) algorithm is implemented within the ma_irl() function. IRL aims to infer the reward function parameters (theta) that best explain the observed demonstration trajectories by matching their feature counts. Let’s pinpoint where this "inverse" process happens in the code.
Location of the IRL Algorithm

The core of the inverse reinforcement learning occurs in the iteration loop inside the ma_irl() function, where the algorithm iteratively updates the parameter vector theta_curr to minimize the difference between the feature counts of the demonstration trajectories and the simulated trajectories. This process is a form of feature matching, a common approach in IRL.

Here’s the breakdown:
Key Section: IRL Iteration Loop

The IRL algorithm is implemented in the following block of code within ma_irl():
julia
#=============
iteration on likelihood maximization
=============#
theta_curr = [1.5, 1.0, 1.0, 1.5, 1.0, 1.0]  # Initial guess for theta
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

num = 200  # Number of simulated trajectories per iteration
max_itr = 500  # Maximum iterations
for itr = 1:max_itr
    println(" ------------- in iteration ", itr , " ------------- ")
    eta = 0.0001  # Learning rate
    avg_pro_feature_counts = zeros(feature_k)

    # Update weights for Player 1
    for player1_itr = 20
        x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)
        avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
        sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

        # Gradient descent update for Player 1's parameters (theta[1:3])
        theta_curr[1:3] = theta_curr[1:3] - eta * (sc_avg_dem_feature_counts[1:3] - sc_avg_pro_feature_counts[1:3])
        theta_curr = max.(0.0, theta_curr)  # Ensure non-negative weights
    end
    # Store feature counts and updated theta for plotting
    push!(fc1, avg_pro_feature_counts[1])
    push!(fc2, avg_pro_feature_counts[2])
    push!(fc3, avg_pro_feature_counts[3])
    push!(fc4, avg_pro_feature_counts[4])
    push!(fc5, avg_pro_feature_counts[5])
    push!(fc6, avg_pro_feature_counts[6])
    push!(theta1, theta_curr[1])
    push!(theta2, theta_curr[2])
    push!(theta3, theta_curr[3])
    push!(theta4, theta_curr[4])
    push!(theta5, theta_curr[5])
    push!(theta6, theta_curr[6])

    # Update weights for Player 2
    for player2_itr = 20
        x_trajectories, u_trajectories = generate_sim(x_init, theta_curr, num)
        avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
        sc_avg_pro_feature_counts = avg_pro_feature_counts./scale_vector

        # Gradient descent update for Player 2's parameters (theta[4:6])
        theta_curr[4:6] = theta_curr[4:6] - eta * (sc_avg_dem_feature_counts[4:6] - sc_avg_pro_feature_counts[4:6])
        theta_curr = max.(0.0, theta_curr)  # Ensure non-negative weights
    end
    # Store feature counts and updated theta for plotting
    push!(fc1, avg_pro_feature_counts[1])
    push!(fc2, avg_pro_feature_counts[2])
    push!(fc3, avg_pro_feature_counts[3])
    push!(fc4, avg_pro_feature_counts[4])
    push!(fc5, avg_pro_feature_counts[5])
    push!(fc6, avg_pro_feature_counts[6])
    push!(theta1, theta_curr[1])
    push!(theta2, theta_curr[2])
    push!(theta3, theta_curr[3])
    push!(theta4, theta_curr[4])
    push!(theta5, theta_curr[5])
    push!(theta6, theta_curr[6])

    # Visualization code (omitted for brevity)
end
Where the "Inverse" Happens

The "inverse" aspect of IRL—recovering the reward parameters theta from observed behavior—occurs specifically in the gradient descent updates within the loop. Here’s how it works step-by-step:

    Demonstration Feature Counts:
        Earlier in ma_irl(), the demonstration trajectories are generated using theta_true = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]:
        
julia

    x_trajectories, u_trajectories = generate_sim(x_init, theta_true, dem_num)
    avg_dem_feature_counts = get_feature_counts(x_trajectories, u_trajectories)
    scale_vector = avg_dem_feature_counts./1000
    sc_avg_dem_feature_counts = avg_dem_feature_counts./scale_vector
    avg_dem_feature_counts represents the average feature counts (state and control costs) from the expert demonstrations, which serve as the target for IRL.

Simulated Trajectories:

    In each iteration, generate_sim(x_init, theta_curr, num) generates num = 200 trajectories using the current estimate theta_curr.
    avg_pro_feature_counts = get_feature_counts(x_trajectories, u_trajectories) computes the feature counts for these simulated trajectories.

Feature Matching via Gradient Descent:

The key "inverse" step is the update to theta_curr based on the difference between demonstration and simulated feature counts:

For Player 1:
julia

    theta_curr[1:3] = theta_curr[1:3] - eta * (sc_avg_dem_feature_counts[1:3] - sc_avg_pro_feature_counts[1:3])

For Player 2:
julia

    theta_curr[4:6] = theta_curr[4:6] - eta * (sc_avg_dem_feature_counts[4:6] - sc_avg_pro_feature_counts[4:6])

Here, sc_avg_dem_feature_counts (scaled demonstration feature counts) and sc_avg_pro_feature_counts (scaled proposed feature counts) are compared, and theta_curr is adjusted to reduce the difference.
The learning rate eta = 0.0001 controls the step size, and max.(0.0, theta_curr) ensures the weights remain non-negative.
Objective:
    The IRL algorithm seeks to find theta_curr such that the feature counts of the simulated trajectories (avg_pro_feature_counts) match those of the demonstrations (avg_dem_feature_counts). This is the essence of the "inverse" process: inferring the reward parameters that explain the observed behavior.

Why This Is Inverse Reinforcement Learning

    Forward RL: Given a reward function (via theta), compute the optimal policy (e.g., via lqgame_QRE in generate_sim).
    Inverse RL: Given observed behavior (demonstration trajectories), infer the reward function (theta) that makes the policy optimal. This happens in the gradient descent steps above, where theta_curr is updated to align simulated behavior with demonstrations.

Specific Lines

The most critical lines where the "inverse" computation occurs are:

Player 1 Update:
julia

theta_curr[1:3] = theta_curr[1:3] - eta * (sc_avg_dem_feature_counts[1:3] - sc_avg_pro_feature_counts[1:3])

Player 2 Update:
julia

    theta_curr[4:6] = theta_curr[4:6] - eta * (sc_avg_dem_feature_counts[4:6] - sc_avg_pro_feature_counts[4:6])

T
