# try to couple the problem and eliminate the degenerate case

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from datetime import datetime
import pickle  # for saving/loading data

from lqgame import *

state_dim_1 = 4
ctrl_dim_1 = 2
state_dim_2 = 4
ctrl_dim_2 = 2
state_dim = state_dim_1 + state_dim_2
ctrl_dim = ctrl_dim_1 + ctrl_dim_2

plan_steps = 20
horizon = 2.0   # [s]
DT = horizon / plan_steps

# =
# coefficients for dynamics
# =

A = np.zeros((state_dim, state_dim))
A[:state_dim_1, :state_dim_1] = np.array([
    [1, 0, DT, 0],
    [0, 1, 0, DT],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
A[state_dim_1:, state_dim_1:] = np.array([
    [1, 0, DT, 0],
    [0, 1, 0, DT],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B1 = np.zeros((state_dim, ctrl_dim_1))
B1[:state_dim_1, :] = np.array([
    [0, 0],
    [0, 0],
    [DT, 0],
    [0, DT]
])

B2 = np.zeros((state_dim, ctrl_dim_2))
B2[state_dim_1:, :] = np.array([
    [0, 0],
    [0, 0],
    [DT, 0],
    [0, DT]
])

def dynamics_forward(s):
    state = s[:state_dim]
    ctrl1 = s[state_dim:state_dim + ctrl_dim_1]
    ctrl2 = s[state_dim + ctrl_dim_1:]
    return A @ state + B1 @ ctrl1 + B2 @ ctrl2

# === Setup system dynamics and cost weights ===

def set_up_system(theta):
    w_state1 = np.zeros((state_dim, state_dim))
    w_state1[:state_dim_1, :state_dim_1] = (theta[0] + theta[1]) * np.eye(state_dim_1)
    w_state1[state_dim_1:, :state_dim_1] = theta[1] * np.eye(state_dim_2, state_dim_1)
    w_state1[:state_dim_1, state_dim_1:] = theta[1] * np.eye(state_dim_1, state_dim_2)
    w_state1[state_dim_1:, state_dim_1:] = (theta[0] + theta[1]) * np.eye(state_dim_2)
    w_ctrl11 = theta[2] * np.eye(ctrl_dim_1)
    w_ctrl12 = theta[3] * np.eye(ctrl_dim_2)

    w_state2 = np.zeros((state_dim, state_dim))
    w_state2[:state_dim_1, :state_dim_1] = (theta[4] + theta[5]) * np.eye(state_dim_1)
    w_state2[state_dim_1:, :state_dim_1] = theta[5] * np.eye(state_dim_2, state_dim_1)
    w_state2[:state_dim_1, state_dim_1:] = theta[5] * np.eye(state_dim_1, state_dim_2)
    w_state2[state_dim_1:, state_dim_1:] = (theta[4] + theta[5]) * np.eye(state_dim_2)
    w_ctrl21 = theta[6] * np.eye(ctrl_dim_1)
    w_ctrl22 = theta[7] * np.eye(ctrl_dim_2)

    Dynamics = {
        "A": np.array([A] * plan_steps),
        "B1": np.array([B1] * plan_steps),
        "B2": np.array([B2] * plan_steps)
    }
    Costs = {
        "Q1": np.array([w_state1] * (plan_steps + 1)),
        "l1": np.array([np.zeros(state_dim)] * (plan_steps + 1)),
        "Q2": np.array([w_state2] * (plan_steps + 1)),
        "l2": np.array([np.zeros(state_dim)] * (plan_steps + 1)),
        "R11": np.array([w_ctrl11] * plan_steps),
        "R12": np.array([w_ctrl12] * plan_steps),
        "R21": np.array([w_ctrl21] * plan_steps),
        "R22": np.array([w_ctrl22] * plan_steps)
    }
    return Dynamics, Costs


# === Run simulations under QRE and collect trajectories ===

def generate_sim(x_init, theta, num=200):
    x_trajectories = []
    u_trajectories = []

    # Compute the Quantal Response Equilibrium
    Dynamics, Costs = set_up_system(theta)
    
    for key in Dynamics:
        print(key, Dynamics[key].shape)
    for key in Costs:
        print(key, Costs[key].shape)
    
    # print(Dynamics, Costs)
    N1, N2, alpha1, alpha2, cov1, cov2 = lqgame_QRE(Dynamics, Costs)

    for _ in range(num):
        x_history = np.zeros((plan_steps + 1, state_dim))
        x_history[0] = x_init
        u_history = np.zeros((plan_steps, ctrl_dim))

        for t in range(plan_steps):
            t_idx = plan_steps - t - 1
            u_mean1 = -N1[t_idx] @ x_history[t] - alpha1[t_idx]
            u_mean2 = -N2[t_idx] @ x_history[t] - alpha2[t_idx]
            u1 = multivariate_normal.rvs(mean=u_mean1, cov=cov1[t_idx])
            u2 = multivariate_normal.rvs(mean=u_mean2, cov=cov2[t_idx])
            u = np.concatenate([u1, u2])
            u_history[t] = u

            x_history[t + 1] = dynamics_forward(np.concatenate([x_history[t], u]))

        x_trajectories.append(x_history)
        u_trajectories.append(u_history)

    return x_trajectories, u_trajectories


# === Compute empirical feature expectations from sampled trajectories ===

def get_feature_counts(x_trajectories, u_trajectories, feature_k):
    feature_counts = np.zeros(feature_k)
    num = len(x_trajectories)

    coeff = np.zeros((state_dim, state_dim))
    coeff[:state_dim_1, :state_dim_1] = np.eye(state_dim_1)
    coeff[state_dim_1:, :state_dim_1] = np.eye(state_dim_2, state_dim_1)
    coeff[:state_dim_1, state_dim_1:] = np.eye(state_dim_1, state_dim_2)
    coeff[state_dim_1:, state_dim_1:] = np.eye(state_dim_2)

    for xtraj, utraj in zip(x_trajectories, u_trajectories):
        for t in range(plan_steps):
            x = xtraj[t]
            u = utraj[t]
            feature_counts[0] += x @ x
            feature_counts[1] += x @ coeff @ x
            feature_counts[2] += u[:ctrl_dim_1] @ u[:ctrl_dim_1]
            feature_counts[3] += u[ctrl_dim_1:] @ u[ctrl_dim_1:]

            feature_counts[4] += x @ x
            feature_counts[5] += x @ coeff @ x
            feature_counts[6] += u[:ctrl_dim_1] @ u[:ctrl_dim_1]
            feature_counts[7] += u[ctrl_dim_1:] @ u[ctrl_dim_1:]

        x_final = xtraj[-1]
        feature_counts[0] += x_final @ x_final
        feature_counts[1] += x_final @ coeff @ x_final
        feature_counts[4] += x_final @ x_final
        feature_counts[5] += x_final @ coeff @ x_final

    return feature_counts / num


# === Learning rate schedule ===

def rate(lr0, itr):
    return lr0 / (1.0 + 0.02 * itr)

# === Plot learning rate decay ===

def plot_lr(eta):
    rates = [rate(eta, i) for i in range(1, 5001)]
    plt.figure()
    plt.plot(rates)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()
    print(rates)


# === Multi-Agent Inverse Reinforcement Learning ===

def ma_irl(sync_update=True, single_update=10, scale=True, eta=0.0001, plot=True, max_itr=5000):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {}

    feature_k = 8
    dem_num = 3000
    num = 200
    x_init = np.array([10, 10, 0, 0, -10, 10, 0, 0])

    fig_theta, axs_theta = plt.subplots(2, 4, figsize=(12, 6))
    fig_f, axs_f = plt.subplots(2, 4, figsize=(12, 6))
    axs_theta = axs_theta.flatten()
    axs_f = axs_f.flatten()

    fig_dem, ax_dem = plt.subplots()
    ax_dem.set_aspect("equal")
    ax_dem.set_xlim(-11, 11)
    ax_dem.set_ylim(-1, 11)

    # Generate demonstrations
    theta_true = np.array([5.0, 1.0, 2.0, 1.0, 5.0, 1.0, 1.0, 2.0])
    
    x_trajs, u_trajs = generate_sim(x_init, theta_true, dem_num)

    if plot:
        for xtraj in x_trajs:
            ax_dem.plot(xtraj[:, 0], xtraj[:, 1], alpha=0.3, color="blue")
            ax_dem.plot(xtraj[:, 4], xtraj[:, 5], alpha=0.3, color="blue")
        plt.pause(0.1)

    avg_dem_feat = get_feature_counts(x_trajs, u_trajs, feature_k)
    scale_vector = avg_dem_feat / 100.0 if scale else np.ones_like(avg_dem_feat)
    sc_avg_dem_feat = avg_dem_feat / scale_vector
    print("this is avg dem feature counts", avg_dem_feat)

    # Optimization loop
    theta_curr = np.array([3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0])
    theta_avg = theta_curr.copy()

    theta_est = np.zeros((max_itr + 1, feature_k))
    theta_smooth = np.zeros((max_itr + 1, feature_k))
    feature_counts = np.zeros((max_itr, feature_k))

    theta_est[0] = theta_curr
    theta_smooth[0] = theta_avg

    for itr in range(1, max_itr + 1):
        print(f"------------- in iteration {itr} -------------")
        lr = rate(eta, itr)

        if sync_update:
            x_trajs, u_trajs = generate_sim(x_init, theta_curr, num)
            avg_pro_feat = get_feature_counts(x_trajs, u_trajs, feature_k)
            sc_avg_pro_feat = avg_pro_feat / scale_vector

            theta_curr -= lr * (sc_avg_dem_feat - sc_avg_pro_feat)
            theta_curr = np.maximum(0.0, theta_curr)
        else:
            for _ in range(single_update):
                x_trajs, u_trajs = generate_sim(x_init, theta_curr, num)
                avg_pro_feat = get_feature_counts(x_trajs, u_trajs, feature_k)
                sc_avg_pro_feat = avg_pro_feat / scale_vector

                theta_curr[:4] -= lr * (sc_avg_dem_feat[:4] - sc_avg_pro_feat[:4])
                theta_curr = np.maximum(0.0, theta_curr)

            for _ in range(single_update):
                x_trajs, u_trajs = generate_sim(x_init, theta_curr, num)
                avg_pro_feat = get_feature_counts(x_trajs, u_trajs, feature_k)
                sc_avg_pro_feat = avg_pro_feat / scale_vector

                theta_curr[4:] -= lr * (sc_avg_dem_feat[4:] - sc_avg_pro_feat[4:])
                theta_curr = np.maximum(0.0, theta_curr)

        feature_counts[itr - 1] = avg_pro_feat
        theta_est[itr] = theta_curr

        if itr >= 10:
            theta_avg = np.mean(theta_est[itr - 9:itr + 1], axis=0)
        else:
            theta_avg = np.mean(theta_est[:itr + 1], axis=0)
        theta_smooth[itr] = theta_avg

        print("   this is avg dem feature counts", avg_dem_feat)
        print("   this is avg proposed feature counts", avg_pro_feat)
        print("   current theta estimate", theta_curr, "averaged", theta_avg)

        for i in range(feature_k):
            axs_theta[i].clear()
            axs_theta[i].plot(theta_est[:itr + 1, i], label='theta_est')
            axs_theta[i].plot(theta_smooth[:itr + 1, i], label='theta_smooth')
            axs_theta[i].plot([theta_true[i]] * (itr + 1), label='theta_true', linestyle='--')
            axs_theta[i].legend()

            axs_f[i].clear()
            axs_f[i].plot(feature_counts[:itr, i], label='feature')
            axs_f[i].plot([avg_dem_feat[i]] * itr, label='dem_feat', linestyle='--')
            axs_f[i].legend()

        # Optionally save data
        data["true_theta"] = theta_true
        data["theta_est"] = theta_est[:itr + 1]
        data["feature_counts_demonstration"] = avg_dem_feat
        data["feature_counts_proposed"] = feature_counts[:itr]

    plt.show()


# === Call it ===

# plot_lr(0.01)
ma_irl(sync_update=True, single_update=20, scale=False, eta=0.01, plot=True, max_itr=10)