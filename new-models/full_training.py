import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import interp1d

from nuscenes.nuscenes import NuScenes

DATA = os.environ.get('DATA', '../../data')
data_path = os.path.join(DATA, 'nuscenes')  # e.g. /data/nuscenes
version = 'v1.0-trainval'

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from pyquaternion import Quaternion
from tqdm import tqdm

m = 4 # 2 for position, 2 for velocity
# ks = (4, 8, 16, 32, 64) # number of cost functions 
k = 16 # number of cost functions

N = 2
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

def training_step(trajectory_batch, cost_net, dynamics_net, policy_net, cost_embeddings, entropy_weight, ece_weight, accel_weight, embed_weight):
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
        s_t = states[:, t]
        s_tp1 = next_states[:, t]

        predicted_actions = []
        for i in range(N):
            s_i_t = s_t[:, i]                  # (B, 4)
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
            s_i_t = s_t[:, i]         # (B, 4)
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

def get_agents_trajectory(nusc, can_bus, scene):
    # sample_token = scene['first_sample_token']
    # x_states = []
    # times = []
    
    # while sample_token:
    #     sample = nusc.get('sample', sample_token)
    #     ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    #     cs_record = nusc.get('calibrated_sensor', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['calibrated_sensor_token'])
    #     
    #     # Ego car position in world frame
    #     pos = np.array(ego_pose['translation'][:2])
    #     rot = Quaternion(ego_pose['rotation'])
    #     
    #     # Velocity approximation
    #     if len(times) >= 2:
    #         dt = (ego_pose['timestamp'] - times[-1]) / 1e6  # convert from microsec to sec vel = (pos - x_states[-1][:2]) / dt
    #     else:
    #         vel = np.array([0., 0.])
    #     
    #     x_state = np.hstack((pos, vel))
    #     x_states.append(x_state)
    #     times.append(ego_pose['timestamp'])
    #     
    #     sample_token = sample['next']
    
    # x_states = np.array(x_states)  # [T, 4]
    # times = np.array(times)
    
    # # Approximate control inputs via finite difference on velocity
    # dt = np.diff(times) / 1e6
    # acc = np.diff(x_states[:, 2:4], axis=0) / dt[:, None]
    
    # return x_states[:-1], acc, times[:-1]  # [T-1, 4], [T-1, 2], [T-1]

    scene_name = scene['name']
    data = []

    # Load CAN bus data for the scene
    try:
        can_data = can_bus.get_messages(scene_name, 'vehicle_monitor')
        if not can_data:
            print(f"No CAN bus data for {scene_name}, skipping.")
        
        # Extract timestamps and accelerations from CAN bus
        can_timestamps = np.array([msg['utime'] for msg in can_data])  # Microseconds
        can_acc = np.array([msg['acceleration'][:2] for msg in can_data])  # [ax, ay] in m/s²

        # Create interpolation function for ax, ay
        interp_ax = interp1d(can_timestamps, can_acc[:, 0], bounds_error=False, fill_value=0)
        interp_ay = interp1d(can_timestamps, can_acc[:, 1], bounds_error=False, fill_value=0)
        
    except Exception as e:
        print(f"Error loading CAN bus for {scene_name}: {e}, skipping.")
    
    # Iterate through samples in the scene
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        sample_time = sample['timestamp']  # Microseconds
        
        # Get agent states: [xstate, ystate, vx, vy] for each agent
        agent_states = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            # Extract position (x, y) and velocity (vx, vy)
            xstate, ystate, _ = ann['translation']  # Ignore z
            vx, vy = ann['velocity'][:2]  # 2D velocity (x, y)
            if np.isnan(vx) or np.isnan(vy):  # Handle missing velocities
                vx, vy = 0.0, 0.0
            agent_states.append([xstate, ystate, vx, vy])
        
        # Convert to numpy array (num_agents x 4)
        agent_states = np.array(agent_states) if agent_states else np.zeros((0, 4))
        
        # Get control input: [ax, ay] from interpolated CAN bus data
        ax = float(interp_ax(sample_time))
        ay = float(interp_ay(sample_time))
        control = np.array([ax, ay])
        
        # Store data
        data.append({
            'scene_name': scene_name,
            'sample_token': sample_token,
            'timestamp': sample_time,
            'agent_states': agent_states,  # Shape: (num_agents, 4)
            'control': control  # Shape: (2,)
        })
        
        # Move to next sample
        sample_token = sample['next']

    return data


# cost_net = CostNet()
# dynamics_net = DynamicsNet()
# cost_embeddings = nn.Parameter(torch.randn(N, k))
# policy_net = PolicyNet(state_dim=4, cost_dim=k, action_dim=m)
# 
# optimizer = torch.optim.Adam(list(cost_net.parameters()) +
#                              list(dynamics_net.parameters()) +
#                              list(policy_net.parameters()) +
#                              [cost_embeddings], lr=1e-3)
# 
# 
# num_epochs = 100

num_agents = 2
nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
can_bus = NuScenesCanBus(dataroot=data_path)

interactive_keywords = ['merge', 'crosswalk', 'intersection', 'vehicle cut', 'yield', 'overtake']

interactive_scenes = []

for scene in nusc.scene:
    # Get first sample in the scene
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    # Count agents (vehicles and pedestrians) in the sample
    agent_count = 0
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        category = ann['category_name']
        if category.startswith('vehicle') or category.startswith('human.pedestrian'):
            agent_count += 1
    
    if any(kw in scene['name'].lower() or kw in scene['description'].lower() for kw in interactive_keywords):
        # Include scene if exactly num_agents agents
        # if agent_count == num_agents:
        interactive_scenes.append(scene)

    x_traj, u_traj, t_traj = get_agents_trajectory(nusc, can_bus, scene)

print(f"Found {len(interactive_scenes)} interactive scenes")

# for epoch in range(num_epochs):
#     for batch in data:
#         loss = training_step(batch, cost_net, dynamics_net, cost_embeddings)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
