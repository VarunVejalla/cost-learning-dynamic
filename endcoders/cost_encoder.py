import torch
import torch.nn as nn

class CostEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, embedding_dim):
        super(CostEncoder, self).__init__()
        input_dim = state_dim + num_agents * action_dim  # S_t + all u_{i,t}
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)  # Outputs w_i
        )
    
    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)  # Concatenate S_t and all u_{i,t}
        return self.layers(x)