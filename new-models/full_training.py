import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt

m = 2 
N = 2
class CostNet(nn.Module):
    def __init__(self, state_dim=4*N, action_dim=m, cost_embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + cost_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action, cost_embed):
        x = torch.cat([state, action, cost_embed], dim=-1)
        return self.net(x)  # (batch_size, 1)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, cost_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + cost_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, s_t, c_i):
        x = torch.cat([s_t, c_i], dim=-1)
        return self.net(x)  # returns a_t^i


class DynamicsNet(nn.Module):
    def __init__(self, state_dim=4*N, action_dim=m*N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)
        return self.net(x)

def training_step(trajectory_batch, cost_net, dynamics_net, policy_net, cost_embeddings, entropy_weight, ece_weight, accel_weight, embed_weight):
    """
    trajectory_batch: dict of tensors with keys:
        - states: (B, T, N, 8)
        - next_states: (B, T, N, 8)
    """
    states = trajectory_batch['states']      # (B, T, N, 8)
    next_states = trajectory_batch['next_states']  # (B, T, N, 8)
    actions = trajectory_batch['actions']      # (B, T, N, 4)

    B, T, N, _ = states.shape
    device = states.device

    total_dyn_loss = 0.
    total_ece_loss = 0.
    total_accel_loss = 0.

    for t in range(T - 1):
        s_t = states[:, t]
        s_tp1 = next_states[:, t]

        predicted_actions = []
        for i in range(N):
            s_i_t = s_t[:, i]                  # (B, 8)
            c_i = cost_embeddings[i].unsqueeze(0).expand(B, -1)  # (B, k)

            a_i_t = policy_net(s_i_t, c_i)     # (B, m)
            predicted_actions.append(a_i_t)

        actions = torch.stack(predicted_actions, dim=1)  # (B, N, m)
        actions_flat = actions.view(B, -1)               # (B, N*m)

        s_t_flat = s_t.view(B, -1)
        s_pred_tp1 = dynamics_net(s_t_flat, actions_flat)
        s_true_tp1 = s_tp1.view(B, -1)
        dyn_loss = F.mse_loss(s_pred_tp1, s_true_tp1)
        total_dyn_loss += dyn_loss

        for i in range(N):
            s_i_t = s_t[:, i]         # (B, 8)
            a_i_t = actions[:, i]     # (B, m)
            c_i = cost_embeddings[i].unsqueeze(0).expand(B, -1)  # (B, k)

            s_i_t.requires_grad_(True)
            a_i_t.requires_grad_(True)
            c_i.requires_grad_(True)

            cost = cost_net(s_t_flat, a_i_t, c_i).squeeze(-1)  # (B,)

            grad_a = torch.autograd.grad(cost.sum(), a_i_t, create_graph=True)[0]  # (B, m)

            hess_diag = []
            for j in range(a_i_t.shape[1]):
                grad2 = torch.autograd.grad(grad_a[:, j].sum(), a_i_t, retain_graph=True)[0][:, j]
                hess_diag.append(grad2)
            hess_diag = torch.stack(hess_diag, dim=1)  # (B, m)

            log_det_Q = torch.log(torch.clamp(hess_diag, min=1e-6)).sum(dim=1)  # (B,)
            ece_loss = cost + entropy_weight * log_det_Q
            total_ece_loss += ece_loss.mean()

        v_t = s_t[:, :, 2:]
        v_tp1 = s_tp1[:, :, 2:]
        accel = (v_tp1 - v_t)
        accel_loss = accel.square().sum(dim=-1).mean()
        total_accel_loss += accel_loss

    embed_loss = (cost_embeddings.norm(dim=-1) - 1).square().mean()

    total_loss = (
        total_dyn_loss +
        ece_weight * total_ece_loss +
        accel_weight * total_accel_loss +
        embed_weight * embed_loss
    )

    return total_loss

# Grid search training function
def train_with_grid_search(filename, cost_embed_dims=[8, 16, 32, 64], num_epochs=100, batch_size=32):
    """
    Perform grid search over cost_embed_dim values, training models and plotting losses.
    """
    # Load data
    with h5py.File(filename, "r") as f:
        print("Keys: %s" % list(f.keys()))
        trajs = np.array(f.get("demo_data"))  # (num_trajs, T, N, 4)
        print(f"Trajectory data shape: {trajs.shape}")

    filename = "MultiAgentIRL-main/cioc_data/twoplayer.h5"
    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())
        data = np.array(f.get("demo_data"))

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

    # Grid search over cost_embed_dim
    for k in cost_embed_dims:
        print(f"\nTraining with cost_embed_dim = {k}")
        
        # Initialize models
        cost_net = CostNet(cost_embed_dim=k)
        dynamics_net = DynamicsNet()
        policy_net = PolicyNet(cost_dim=k, state_dim=4, action_dim=m)
        cost_embeddings = nn.Parameter(torch.randn(N, k))
        
        optimizer = torch.optim.Adam(
            list(cost_net.parameters()) +
            list(dynamics_net.parameters()) +
            list(policy_net.parameters()) +
            [cost_embeddings],
            lr=1e-3
        )

        # Training loop
        epoch_losses = []
        line, = ax.plot([], [], label=f'k={k}')
        lines.append(line)
        labels.append(f'k={k}')

        for epoch in range(num_epochs):
            total_loss = 0.
            num_batches = data.shape[2]
            for sim_index in range(num_batches):

                # Select single simulation and transpose
                batch = data[:, :, sim_index]  # Shape: (20, 61)
                batch = data.T  # Shape: (61, 20)

                # Extract states, refs, and actions
                states = torch.tensor(batch[:, 0:8], dtype=torch.float32)  # (61, 8)
                # refs = torch.tensor(batch[:, 8:16], dtype=torch.float32)   # (61, 8)
                actions = torch.tensor(batch[:, 16:20], dtype=torch.float32)  # (61, 4)
 
                states = states[np.newaxis, :]
                batch = {'states': states[:, :, :, :-1],
                         'next_states': states[:, :, :, 1:],
                         'actions': actions}

                loss = training_step(
                    batch, cost_net, dynamics_net, policy_net, cost_embeddings,
                    entropy_weight=0.1, ece_weight=1.0, accel_weight=0.1, embed_weight=0.1
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

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

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # Store results
        final_loss = np.mean(epoch_losses[-5:])  # Average of last 5 epochs
        results.append({'cost_embed_dim': k, 'final_loss': final_loss})
        if final_loss < best_loss:
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

    return best_k, results

# Run grid search
if __name__ == "__main__":
    filename = "MultiAgentIRL-main/cioc_data/twoplayer.h5"  # Update with your path
    best_k, results = train_with_grid_search(filename, cost_embed_dims=[8, 16, 32, 64], num_epochs=100, batch_size=32)
