import torch
import torch.nn as nn
import torch.nn.functional as F


class CostNet(nn.Module):
    def __init__(self, state_dim=4*N, action_dim=m, cost_embed_dim=k):
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

def training_step(trajectory_batch, cost_net, dynamics_net, policy_net, cost_embeddings, λ=1.0, α=1.0, β=1.0, γ=1.0):
    """
    trajectory_batch: dict of tensors with keys:
        - states: (B, T, N, 4)
        - next_states: (B, T, N, 4)
    """
    states = trajectory_batch['states']      # (B, T, N, 4)
    next_states = trajectory_batch['next_states']  # (B, T, N, 4)

    B, T, N, _ = states.shape
    device = states.device

    total_dyn_loss = 0.
    total_ece_loss = 0.
    total_accel_loss = 0.

    for t in range(T - 1):
        s_t = states[:, t]       # (B, N, 4)
        s_tp1 = next_states[:, t]  # (B, N, 4)

        # === Use PolicyNet to predict actions ===
        predicted_actions = []  # (B, N, m)
        for i in range(N):
            s_i_t = s_t[:, i]                  # (B, 4)
            c_i = cost_embeddings[i].unsqueeze(0).expand(B, -1)  # (B, k)

            a_i_t = policy_net(s_i_t, c_i)     # (B, m)
            predicted_actions.append(a_i_t)

        actions = torch.stack(predicted_actions, dim=1)  # (B, N, m)
        actions_flat = actions.view(B, -1)               # (B, N*m)

        # === Forward Dynamics Loss ===
        s_t_flat = s_t.view(B, -1)
        s_pred_tp1 = dynamics_net(s_t_flat, actions_flat)
        s_true_tp1 = s_tp1.view(B, -1)
        dyn_loss = F.mse_loss(s_pred_tp1, s_true_tp1)
        total_dyn_loss += dyn_loss

        # === ECE Loss ===
        for i in range(N):
            s_i_t = s_t[:, i]         # (B, 4)
            a_i_t = actions[:, i]     # (B, m)
            c_i = cost_embeddings[i].unsqueeze(0).expand(B, -1)  # (B, k)

            # Enable gradients
            s_i_t.requires_grad_(True)
            a_i_t.requires_grad_(True)
            c_i.requires_grad_(True)

            cost = cost_net(s_t_flat, a_i_t, c_i).squeeze(-1)  # (B,)

            # First derivative wrt action
            grad_a = torch.autograd.grad(cost.sum(), a_i_t, create_graph=True)[0]  # (B, m)

            # Second derivative (Hessian diagonal approx)
            hess_diag = []
            for j in range(a_i_t.shape[1]):
                grad2 = torch.autograd.grad(grad_a[:, j].sum(), a_i_t, retain_graph=True)[0][:, j]
                hess_diag.append(grad2)
            hess_diag = torch.stack(hess_diag, dim=1)  # (B, m)

            log_det_Q = torch.log(torch.clamp(hess_diag, min=1e-6)).sum(dim=1)  # (B,)
            ece_loss = cost + λ * log_det_Q
            total_ece_loss += ece_loss.mean()

        # === Acceleration Penalty ===
        v_t = s_t[:, :, 2:]         # (B, N, 2)
        v_tp1 = s_tp1[:, :, 2:]
        accel = (v_tp1 - v_t)       # assume unit timestep
        accel_loss = accel.square().sum(dim=-1).mean()
        total_accel_loss += accel_loss

    # === Regularization ===
    embed_loss = (cost_embeddings.norm(dim=-1) - 1).square().mean()

    # === Combine all losses ===
    total_loss = (
        total_dyn_loss +
        α * total_ece_loss +
        β * total_accel_loss +
        γ * embed_loss
    )

    return total_loss


cost_net = CostNet()
dynamics_net = DynamicsNet()
cost_embeddings = nn.Parameter(torch.randn(N, k))
policy_net = PolicyNet(state_dim=4, cost_dim=k, action_dim=m)

optimizer = torch.optim.Adam(list(cost_net.parameters()) +
                             list(dynamics_net.parameters()) +
                             list(policy_net.parameters()) +
                             [cost_embeddings], lr=1e-3)


num_epochs = 100

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = training_step(batch, cost_net, dynamics_net, cost_embeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
