import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt

d = 4  # state dimension per agent
m = 2  # Action dimension per agent
N = 2  # Number of agents

class CostNet(nn.Module):
    def __init__(self, state_dim=d+N*d, action_dim=m, cost_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + cost_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, full_state, action, cost_embed):
        x = torch.cat([state, full_state, action, cost_embed], dim=-1)
        return self.net(x)  # (batch_size, 1)

class PolicyNet(nn.Module):
    def __init__(self, state_dim=d*N, cost_dim=16, action_dim=m*N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + N * cost_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, s_t, c_i):
        x = torch.cat([s_t, c_i], dim=-1)
        return self.net(x)  # (batch_size, action_dim)

class DynamicsNet(nn.Module):
    def __init__(self, state_dim=d, action_dim=m):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)
        return self.net(x)

def preprocess_data(data):
    """
    Load data for a single simulation and preprocess for training.

    Args:
        data: the data to preprocess

    Returns:
        Dictionary with states, next_states, and actions.
    """
    # Extract states and actions
    states1 = torch.tensor(data[:, 0:4], dtype=torch.float32)  # (61, 4)
    states2 = torch.tensor(data[:, 4:8], dtype=torch.float32)  # (61, 4)
    # actions = torch.tensor(data[:, 16:20], dtype=torch.float32)  # (61, 4)

    # Stack states: (61, 2, 4)
    states = torch.stack([states1, states2], dim=1)
    states = states.unsqueeze(0)
    # actions = actions.unsqueeze(0)
    # print(actions.shape)
    # actions = actions.view(1, -1, N, m)
    # print(actions[:, :-1, :].shape)
    batchd = {
        'states': states[:, :-1, :, :],  # (1, 60, 2, 4)
        'next_states': states[:, 1:, :, :],  # (1, 60, 2, 4)
        # 'actions': actions[:, :-1, :, :]  # (1, 60, 2, 2)
    }
    return batchd

def training_step(trajectory_batch, cost_net, dynamics_net, policy_net, cost_embeddings, entropy_weight, ece_weight, accel_weight, embed_weight):
    """
    Perform a training step for multi-agent IRL.

    Args:
        trajectory_batch: dict with keys:
            - states: (B, T, N, 4)
            - next_states: (B, T, N, 4)
        cost_net, dynamics_net, policy_net: Neural network models.
        cost_embeddings: (N, k) cost weights for each agent.
        entropy_weight, ece_weight, accel_weight, embed_weight: Loss weights.

    Returns:
        Total loss.
    """
    states = trajectory_batch['states']  # (B, T, N, d)
    next_states = trajectory_batch['next_states']  # (B, T, N, d)
    # actions = trajectory_batch['actions']  # (B, T, N, m)

    B, T, N, _ = states.shape
    device = states.device

    total_dyn_loss = 0.
    total_ece_loss = 0.
    total_accel_loss = 0.

    for t in range(T):
        s_t = states[:, t]  # (B, N, d)
        s_tp1 = next_states[:, t]  # (B, N, d)

        # Flatten state for dynamics and cost networks
        s_t_flat = s_t.view(B, -1)  # (B, N*d)

        # Predict actions for all agents in one pass
        predicted_actions = policy_net(s_t_flat, cost_embeddings.view(1, -1))  # (B, N*m) # (N, k) --> (B, N*k)
        predicted_actions = predicted_actions.view(B, N, m)  # (B, N, m)

        # Dynamics loss using predicted actions
        # actions_flat = predicted_actions.view(B, -1)  # (B, N*m)
        
        dyn_loss = 0
        for i in range(N):
            s_pred = dynamics_net(s_t[:, i], predicted_actions[:, i])  # (B, N*4)
            s_true = s_tp1[:, i]  # (B, N*4)
            dyn_loss += F.mse_loss(s_pred, s_true)
        
        total_dyn_loss += dyn_loss

        # ECE loss for each agent
        for i in range(N):
            a_i_t = predicted_actions[:, i]  # (B, 2)
            c_i = cost_embeddings[i].unsqueeze(0).expand(B, -1)  # (B, k)

            # Ensure gradients
            # s_t_flat = torch.cat()
            # predicted_actions.requires_grad_(True)
            # s_t.requires_grad_(True)
            # s_t_flat.requires_grad_(True)
            a_i_t = a_i_t.clone().detach().requires_grad_(True)
            # c_i.requires_grad_(True)

            cost = torch.abs(cost_net(s_t[:, i], s_t_flat, a_i_t, c_i).squeeze(-1))  # (B,)
            # print(cost.requires_grad, a_i_t.requires_grad, cost.grad_fn)
            # Compute gradient and Hessian diagonal for entropy term
            grad_a = torch.autograd.grad(cost.sum(), a_i_t, create_graph=True)[0]  # (B, 2)
            hess_diag = []
            for j in range(a_i_t.shape[1]):
                grad2 = torch.autograd.grad(grad_a[:, j].sum(), a_i_t, retain_graph=True)[0][:, j]
                hess_diag.append(grad2)
            hess_diag = torch.stack(hess_diag, dim=1)  # (B, 2)

            log_det_Q = torch.log(torch.clamp(hess_diag, min=1e-6)).sum(dim=1)  # (B,)
            ece_loss = cost - entropy_weight * log_det_Q
            total_ece_loss += ece_loss.mean()

        # Acceleration loss
        v_t = s_t[:, :, 2:]  # (B, N, 2)
        v_tp1 = s_tp1[:, :, 2:]  # (B, N, 2)
        accel = (v_tp1 - v_t)  # (B, N, 2)
        accel_loss = accel.square().sum(dim=-1).mean()
        total_accel_loss += accel_loss

    # Embedding regularization
    embed_loss = (cost_embeddings.norm(dim=-1) - 1).square().mean()

    total_loss = (
        total_dyn_loss +
        ece_weight * total_ece_loss +
        accel_weight * total_accel_loss +
        embed_weight * embed_loss
    )

    return total_loss

def rollout_trajectory(initial_state, policy_net, dynamics_net, cost_embeddings, horizon=60):
    """
    Roll out a trajectory using the learned policy and dynamics models.

    Args:
        initial_state: (N, d) tensor of the starting state for all agents.
        policy_net: trained policy network.
        dynamics_net: trained dynamics model.
        cost_embeddings: learned cost embeddings (N, k).
        horizon: number of time steps to simulate.

    Returns:
        Tensor of shape (horizon+1, N, d) with the predicted trajectory.
    """
    trajectory = [initial_state]
    state = initial_state.unsqueeze(0)  # (1, N, d)

    for _ in range(horizon):
        s_t_flat = state.view(1, -1)  # (1, N*d)
        cost_input = cost_embeddings.view(1, -1)  # (1, N*k)
        action = policy_net(s_t_flat, cost_input).view(N, m)  # (N, m)

        next_state = torch.stack([
            dynamics_net(state[0, i], action[i]) for i in range(N)
        ], dim=0)  # (N, d)

        trajectory.append(next_state)
        state = next_state.unsqueeze(0)

    return torch.stack(trajectory, dim=0)  # (T+1, N, d)

def plot_trajectory(pred_trajectory, true_trajectory):
    """
    Plot predicted and true (x, y) trajectories for each agent.

    Args:
        pred_trajectory: (T, N, d) tensor of predicted states.
        true_trajectory: (T, N, d) tensor of ground truth states.
    """
    T, N, _ = pred_trajectory.shape
    colors = ['r', 'b']
    plt.figure(figsize=(6, 6))

    for i in range(N):
        # Predicted
        x_pred = pred_trajectory[:, i, 0].detach().cpu().numpy()
        y_pred = pred_trajectory[:, i, 1].detach().cpu().numpy()
        plt.plot(x_pred, y_pred, color=colors[i], linestyle='--', label=f'Agent {i+1} Pred')

        # Ground truth
        x_true = true_trajectory[:, i, 0]
        y_true = true_trajectory[:, i, 1]
        plt.plot(x_true, y_true, color=colors[i], linestyle='-', label=f'Agent {i+1} True')

        # Start and end points
        plt.scatter(x_true[0], y_true[0], color=colors[i], marker='o')  # Start
        plt.scatter(x_true[-1], y_true[-1], color=colors[i], marker='x')  # End

    plt.title("Predicted vs True Trajectories from Simulation 300")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()



def train_with_grid_search(filename, cost_dims=[8, 16, 32], num_epochs=30, batch_size=1):
    """
    Perform grid search over cost_embed_dim values, training models and plotting losses.
    """
    # Set up real-time plotting
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Training Loss vs. Epoch for Different cost_embed_dim")
    ax.grid(True)
    lines = []
    labels = []
    best_loss = float('inf')
    best_k = None
    results = []


    with h5py.File(filename, "r") as f:
      data = np.array(f.get("demo_data"))  # (20, 61, 500)
    best_model = None
    # Grid search over cost_embed_dim
    for k in cost_dims:
        print(f"\nTraining with cost_embed_dim = {k}")
        
        # Initialize models
        cost_net = CostNet(cost_dim=k)
        dynamics_net = DynamicsNet()
        policy_net = PolicyNet(cost_dim=k)
        cost_embeddings = nn.Parameter(torch.randn(N, k))
        
        optimizer = torch.optim.Adam(
            list(cost_net.parameters()) +
            list(dynamics_net.parameters()) +
            list(policy_net.parameters()) +
            [cost_embeddings],
            lr=1e-3
        )
        print('made optimizer')

        # Training loop
        epoch_losses = []
        line, = ax.plot([], [], label=f'k={k}')
        lines.append(line)
        labels.append(f'k={k}')

        for epoch in range(num_epochs):
            print(f'starting epoch {epoch}')
            total_loss = 0.
            num_batches = 0

            # Process one simulation at a time (batch_size=1)
            for sim_index in range(10):#data.shape[2]):  # Total 500 simulations
                print(sim_index)
                # Select single simulation and transpose
                batch = data[:, :, sim_index]  # (20, 61)
                batch = batch.T  # (61, 20)
                batchd = preprocess_data(batch)

                loss = training_step(
                    batchd, cost_net, dynamics_net, policy_net, cost_embeddings,
                    entropy_weight=0.1, ece_weight=0.1, accel_weight=0.1, embed_weight=0.1
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            print('done with sim loop')
            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)

            # Update real-time plot
            line.set_xdata(range(1, len(epoch_losses) + 1))
            line.set_ydata(epoch_losses)
            ax.relim()
            ax.autoscale_view()
            ax.legend(labels)
            plt.draw()
            plt.pause(0.01)
            print(epoch_losses)

            # Print progress
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # Store results
        final_loss = np.mean(epoch_losses[-5:])  # Average of last 5 epochs
        results.append({'cost_embed_dim': k, 'final_loss': final_loss})
        if final_loss < best_loss:
            best_model = (cost_net, dynamics_net, policy_net, cost_embeddings)
            best_loss = final_loss
            best_k = k
        

    # Finalize plot
    plt.ioff()
    plt.show()

    # Print grid search results
    print("\nGrid Search Results:")
    for res in results:
        print(f"cost_embed_dim={res['cost_embed_dim']}: Final Loss = {res['final_loss']:.4f}")
    print(f"\nBest cost_embed_dim: {best_k} with Final Loss: {best_loss:.4f}")

    best_cost_net, best_dynamics_net, best_policy_net, best_cost_embeddings = best_model
    test_sim = data[:, :, 300].T  # (61, 20)
    # True trajectory (61, 2, 4)
    states1 = torch.tensor(test_sim[:, 0:4], dtype=torch.float32)
    states2 = torch.tensor(test_sim[:, 4:8], dtype=torch.float32)
    true_trajectory = torch.stack([states1, states2], dim=1)  # (61, 2, 4)

    # Initial state
    initial_state = true_trajectory[0]  # (2, 4)

    print("Rolling out trajectory from simulation index 300...")
    pred_traj = rollout_trajectory(initial_state, best_policy_net, best_dynamics_net, best_cost_embeddings, horizon=60)
    plot_trajectory(pred_traj, true_trajectory[:61])
    
    return best_k, results, best_model

# Run grid search
if __name__ == "__main__":
    filename = "MultiAgentIRL-main/cioc_data/twoplayer.h5"
    best_k, results = train_with_grid_search(filename, cost_dims=[8], num_epochs=30, batch_size=1)
