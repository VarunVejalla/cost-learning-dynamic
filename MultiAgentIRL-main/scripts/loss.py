import torch
import torch.nn as nn
import torch.distributions as dist

class GMMLoss(nn.Module):
    def __init__(self, n_gaussians=3):
        super(GMMLoss, self).__init__()
        self.n_gaussians = n_gaussians
    
    def forward(self, means, covs, weights, targets):
        # Expand dimensions to make shapes compatible for batched operations
        batch_size = means.shape[0]
        means = means.view(batch_size, self.n_gaussians, -1) # [B, n_gaussians, 2*self.num_waypoints]
        covs = covs.view(batch_size, self.n_gaussians, -1) # [B, n_gaussians, 2*self.num_waypoints]
        # Expand dimensions of targets to make shapes compatible for batched operations
        targets_exp = targets.unsqueeze(1).repeat(1, self.n_gaussians, 1) # [B, n_gaussians, 2*self.num_waypoints]
        # Compute the Gaussian distributions for all components at once
        gaussian_dists = dist.MultivariateNormal(means, covariance_matrix=torch.diag_embed(covs))
        # Compute log probabilities
        log_probs = gaussian_dists.log_prob(targets_exp) # [B, n_gaussians, 2*self.num_waypoints]
        # clip this to ensure numerical stability
        log_probs = torch.clamp(log_probs, min=-100, max=0)
        # add log weights
        log_probs_weighted = log_probs + torch.log(weights)
        # compute the total log likelihood using logsumexp for numerical stability
        total_log_likelihood = torch.logsumexp(log_probs_weighted, dim=1) # [B, 2*self.num_waypoints]
        return -1.0 * total_log_likelihood
